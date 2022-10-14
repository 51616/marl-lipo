import random

import numpy as np
from gym.spaces import Discrete, Box
import torch
from torch import nn
from torch import optim
import torch.distributions as D

from coop_marl.utils import Arrdict, arrdict, compute_gae, dict_to_tensor,\
                            FLOAT_MIN, FLOAT_MAX, get_logger, get_discount_coef, safe_log
from coop_marl.agents import Agent
from coop_marl.models.modules import FCLayers

logger = get_logger()

class MAPPOTrajeDiAgent(Agent):
    def __init__(self, config):
        self.obs_space = config.obs_space
        self.act_space = config.act_space
        self.use_gpu = config.use_gpu
        self.config = config
        self.device = 'cpu'
        self.training_device = getattr(config, 'training_device', 'cpu')
        self.grad_clip = getattr(config, 'grad_clip', 0.5)
        assert isinstance(self.act_space, (Discrete, Box)), \
        f'Only support Discrete and Box action space, got {type(self.act_space)}'
        assert getattr(config,'mb_size',False) ^ getattr(config,'num_mb',False), \
        f'Use either mb_size or num_mb'

        self.value_net = CentralizedValueNet(config).to(self.device)
        self.policy_net = PolicyNet(self.obs_space, self.act_space, config).to(self.device)
        self.nets = [self.value_net, self.policy_net]
        self.opt = optim.Adam(list(self.value_net.parameters()) + list(self.policy_net.parameters()), lr=config.lr, eps=1e-5)
        self.cur_iter = 0

    def get_param(self):
        return {'policy_param':self.policy_net.state_dict(),
                'critic_param':self.value_net.state_dict()}

    def set_param(self, param):
        self.value_net.load_state_dict(param['critic_param'])
        self.policy_net.load_state_dict(param['policy_param'])

    def get_state(self):
        state = self.get_param()
        state['config'] = self.config
        return state

    def load_state(self, state):
        self.set_param(state)
        self.config = state['config']

    def calc_action_dist(self, obs_batch, *args , **kwargs):
        if isinstance(self.act_space, Discrete):
            logits = self.policy_net(obs_batch)
            dist = D.Categorical(logits=logits)
        else:
            mean, logstd = self.policy_net(obs_batch)
            std = torch.exp(logstd)
            # support Box with more than 1 dimension
            dist = D.MultivariateNormal(mean, scale_tril=torch.diag(std))
        return dist

    def get_prev_decision_view(self):
        return Arrdict()

    @torch.no_grad()
    def act(self, inp):
        obs_batch = dict_to_tensor(inp.data.obs, device=self.device)
        state_batch = dict_to_tensor(inp.data.state, device=self.device)
        dist = self.calc_action_dist(obs_batch)
        action_mask = getattr(inp.data, 'action_mask', None)
        if action_mask is not None:
            mask_tensor = dict_to_tensor(action_mask, device=self.device)
            dist.logits += torch.clamp(torch.log(mask_tensor), FLOAT_MIN, FLOAT_MAX)

        actions = dist.sample()
        logprobs = dist.log_prob(actions).detach().cpu().numpy()
        actions = actions.detach().cpu().numpy()
        values = self.value_net(state_batch).detach().cpu().numpy()
        if getattr(self.config, 'get_act_dist', False):
            if isinstance(self.act_space, Discrete):
                dists = dist.probs.detach().cpu().numpy()
                return [Arrdict(action=a, logprob=l, value=v,
                                act_dist=dist) \
                                for a,l,v,dist in zip(actions,logprobs,values,dists)]
            else:
                raise NotImplementedError
        else:
            return [Arrdict(action=a, logprob=l, value=v,) \
                            for a,l,v in zip(actions,logprobs,values)]


    @torch.no_grad()
    def _preprocess(self, traj):
        next_state = traj.outcome.state[-1:]
        next_value = self.value_net(torch.tensor(next_state, dtype=torch.float, device=self.device)).squeeze().detach().cpu().numpy()
        # print(next_value.shape)
        compute_gae(traj, self.config.gamma, self.config.gae_lambda, traj.decision.value, next_value)

    def compute_jsd_loss(self, ep_list, agent_idx, ep_pad_mask, time_pad_mask):
        ep_len = ep_list[0].inp.data.obs.shape[0]
        assert all(np.array([ep.inp.data.obs.shape[0] for ep in ep_list]) == ep_len)
        i = agent_idx

        sampled_idx = random.sample(range(len(ep_list)), k=len(ep_list)//self.config.num_mb)
        sampled_ep = [ep_list[i] for i in sampled_idx]
        n_ep = len(sampled_ep)

        mu_i = torch.tensor([ep.pi[i] for ep in sampled_ep], dtype=torch.float, device=self.training_device)
        pi_hat = torch.tensor([ep.pi_hat for ep in sampled_ep], dtype=torch.float, device=self.training_device)
        ep_pad_mask = torch.tensor(ep_pad_mask[sampled_idx], dtype=torch.bool, device=self.training_device).view(n_ep, 1)
        time_pad_mask = torch.tensor(time_pad_mask[sampled_idx], dtype=torch.bool, device=self.training_device).view(n_ep, ep_len)

        delta_hat = torch.stack([ep.delta_hat for ep in sampled_ep]).float().to(self.training_device)

        obs_batch = torch.tensor([ep.inp.data.obs for ep in sampled_ep], dtype=torch.float, device=self.training_device)
        obs_batch = obs_batch.view(n_ep*ep_len, *obs_batch.shape[2:]) # remove ep dimension
        action_batch = torch.tensor([ep.decision.action for ep in sampled_ep], dtype=torch.float, device=self.training_device)
        action_batch = action_batch.view(n_ep*ep_len, *action_batch.shape[2:])

        act_dist = self.calc_action_dist(obs_batch)
        act_logprob = act_dist.log_prob(action_batch)
        act_logprob = act_logprob.reshape(n_ep, ep_len)
        act_logprob = act_logprob * ep_pad_mask * time_pad_mask
        
        d = get_discount_coef(self.config.kernel_gamma, ep_len, device=act_logprob.device).unsqueeze(0).repeat(n_ep,1,1) # [n_ep, ep_len, ep_len]

        delta_i = torch.sum(d * act_logprob.unsqueeze(1), axis=2).exp() # [n_ep, ep_len] (delta[t]) is delta_t


        pi_i  = torch.sum(act_logprob, axis=1).exp() + 1e-8 # sum over time dimension

        pi_i = pi_i.unsqueeze(1)
        pi_hat = pi_hat.unsqueeze(1)
        mu_i = mu_i.unsqueeze(1)


        jsd_vec = pi_hat / (mu_i + 1e-8) * delta_i.detach() * safe_log(delta_i) + \
        1 / ep_len * (pi_i / (mu_i + 1e-8) * (delta_hat - 1 / self.config.pop_size * safe_log(delta_i))).detach() * safe_log(pi_i)
        jsd_loss_vec = torch.mean(torch.sum(jsd_vec, axis=-1), axis=0)
        return jsd_loss_vec

    def train(self, batch, ep_list=None, agent_idx=None, ep_pad_mask=None, time_pad_mask=None):
        self.cur_iter += 1
        if self.config.anneal_lr:
            frac = min(1, (self.cur_iter - 1.0) / (self.config.num_anneal_iter))
            lrnow = self.config.lr - frac * (self.config.lr - self.config.min_anneal_lr)  # max(frac * self.config.lr, self.config.min_anneal_lr)
            for g in self.opt.param_groups:
                g['lr'] = lrnow
            self.cur_lr = lrnow
            logger.debug(f'current learning rate: {lrnow}')

        self._preprocess(batch)
        assert 'adv' in batch.outcome, f'Must call `preprocess()` before calling `train()`'
        batch = arrdict.torchify(batch).to(self.device)
        batch_size = batch.inp.data.obs.shape[0]
        mb_size = getattr(self.config,'mb_size',None) or int(np.ceil(batch_size/self.config.num_mb))
        for net in self.nets:
            net.to(self.training_device)

        for ep in range(self.config.epochs):
            idx = np.arange(batch_size)
            np.random.shuffle(idx)
            for start in range(0, batch_size, mb_size):
                end = start + mb_size
                mb_idx = idx[start:end]
                mb = batch[mb_idx].to(self.training_device)
                mb_adv = mb.outcome.adv
                # normalize adv
                mb_adv = (mb_adv - mb_adv.mean())/(mb_adv.std()+1e-8)

                dist = self.calc_action_dist(mb.inp.data.obs)
                logprob = dist.log_prob(mb.decision.action)
                # calc entropy and ratio
                entropy = dist.entropy()
                ratio = (logprob - mb.decision.logprob).exp()

                # ppo loss
                pg_loss1 = mb_adv * ratio
                pg_loss2 = mb_adv * torch.clamp(ratio, 1-self.config.clip_param, 1+self.config.clip_param)
                pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()

                # value loss
                cur_values = self.value_net(mb.inp.data.state)
                v_loss_unclipped = torch.pow(mb.outcome.ret-cur_values, 2)
                v_clipped = mb.decision.value + torch.clamp(cur_values - mb.decision.value, -self.config.clip_param, self.config.clip_param)
                v_loss_clipped = torch.pow(mb.outcome.ret-v_clipped, 2)
                v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef

                if self.config.diverse_coef!=0:
                    jsd_loss = self.config.diverse_coef * self.compute_jsd_loss(ep_list, agent_idx, ep_pad_mask, time_pad_mask)
                    loss += jsd_loss

                self.opt.zero_grad()
                loss.backward()
                # clip grad
                nn.utils.clip_grad_norm_(list(self.value_net.parameters()) + list(self.policy_net.parameters()), self.grad_clip)
                self.opt.step()

        for net in self.nets:
            net.to(self.device)
    
    def reset(self):
        pass

class CentralizedValueNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_dim = int(np.prod(config.state_shape))
        self.inp_dim = self.state_dim
        self.joint_action_size = self.config.act_space.n if isinstance(self.config.act_space, Discrete) else np.prod(self.config.act_space.shape)
        hidden_size = getattr(config,'hidden_size',64)
        num_hidden = 2
        self.layers = FCLayers(self.state_dim, hidden_size, num_hidden, 1, head_std=1.0)

    def forward(self, inp):
        return self.layers(inp).view(-1)
        
class PolicyNet(nn.Module):
    def __init__(self, obs_space, action_space, config):
        super().__init__()
        obs_size = obs_space.n if isinstance(obs_space, Discrete) else np.prod(obs_space.shape)
        hidden_size = getattr(config,'hidden_size',64)
        num_hidden = 2
        self.logstd = None
        
        if isinstance(action_space, Discrete):
            act_size = action_space.n
        else:
            act_size = np.prod(action_space.shape)
            self.logstd = nn.Parameter(torch.zeros(act_size))
        self.layers = FCLayers(obs_size, hidden_size, num_hidden, act_size, head_std=0.01)

    def forward(self, inp):
        if self.logstd is None:
            return self.layers(inp)
        else:
            mean = self.layers(inp)
            logstd = self.logstd
            return mean, logstd
