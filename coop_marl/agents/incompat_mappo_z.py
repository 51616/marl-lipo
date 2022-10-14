import numpy as np
from gym.spaces import Discrete, Box
import torch
from torch import nn
from torch import optim
import torch.distributions as D

from coop_marl.utils import Arrdict, arrdict, compute_gae, dict_to_tensor,\
                            FLOAT_MIN, FLOAT_MAX, get_logger
from coop_marl.agents import Agent
from coop_marl.models.modules import FCLayers

logger = get_logger()

class IncompatMAPPOZ(Agent):
    def __init__(self, config): 
        self.config = config
        self.obs_space = config.obs_space
        self.act_space = config.act_space
        self.epochs = config.epochs
        self.device = 'cpu' 
        self.training_device = getattr(config, 'training_device', 'cpu')
        self.xp_coef = config.xp_coef
        self.use_gpu = config.use_gpu
        self.cur_iter = 0
        self.z = None
        self.grad_clip = getattr(config, 'grad_clip', 0.5)
        self.policy_net = PolicyNet(self.obs_space, self.act_space, config) 
        self.sp_critic = CentralizedValueNet(config)
        # team of param_j with agent_i
        self.xp_critics = [CentralizedValueNet(config) for _ in range(config.pop_size)] 
        self.discrim = Discriminator(config) 
        self.nets = [self.policy_net, self.sp_critic, *self.xp_critics, self.discrim]

        self.opts = [optim.Adam(net.parameters(), lr=config.lr, eps=1e-5) for net in [self.policy_net, self.sp_critic,
                                                                                    *self.xp_critics, self.discrim]]
        assert (getattr(config,'mb_size',False)>0) ^ (getattr(config,'num_mb',False)>0), \
        f'Use either mb_size or num_mb'

    def get_param(self):
        return {'policy_param':self.policy_net.state_dict(),
                'sp_critic_param':self.sp_critic.state_dict(),
                'xp_critic_param_list':[net.state_dict() for net in self.xp_critics],
                }

    def set_param(self, param):
        self.sp_critic.load_state_dict(param['sp_critic_param'])
        self.policy_net.load_state_dict(param['policy_param'])
        assert len(self.xp_critics) == len(param['xp_critic_param_list']), f'{len(self.xp_critics)}, {len(param["xp_critic_param_list"])}'
        for net, param in zip(self.xp_critics, param['xp_critic_param_list']):
            net.load_state_dict(param)

    def get_state(self):
        # could save the critics and the optimizers as well if needed...
        state = self.get_param()
        state['config'] = self.config
        state['cur_iter'] = self.cur_iter
        return state

    def load_state(self, state, load_z=True):
        # could load the discriminator the optimizers as well if needed...
        self.set_param(state)
        self.config = state['config']
        # set_z
        # self.z is only used in act
        # calc_action_dist and train still use z_batch
        if ('z' in state) and load_z:
            self.z = state['z']

    def calc_action_dist(self,obs_batch,z_batch):
        if self.z is not None:
            z_batch = self.z.repeat(obs_batch.shape[0],1)
        if isinstance(self.act_space, Discrete):
            logits = self.policy_net(obs_batch,z_batch)
            dist = D.Categorical(logits=logits)
        elif isinstance(self.act_space, Box):
            mean, logstd = self.policy_net(obs_batch,z_batch)
            std = torch.exp(logstd)
            # support Box with more than 1 dimension
            dist = D.MultivariateNormal(mean, scale_tril=torch.diag(std))
        else:
            raise NotImplementedError
        return dist

    def get_prev_decision_view(self):
        return Arrdict()

    @torch.no_grad()
    def act(self, inp):
        obs_batch = dict_to_tensor(inp.data.obs, device=self.device)
        z_batch = dict_to_tensor(inp.data.z, device=self.device)
        state_batch = dict_to_tensor(inp.data.state, device=self.device)
        dist = self.calc_action_dist(obs_batch, z_batch)
        action_mask = getattr(inp.data, 'action_mask', None)
        if action_mask is not None:
            mask_tensor = dict_to_tensor(action_mask, device=self.device)
            dist.logits += torch.clamp(torch.log(mask_tensor), FLOAT_MIN, FLOAT_MAX)

        actions = dist.sample()
        logprobs = dist.log_prob(actions).detach().cpu().numpy()
        actions = actions.detach().cpu().numpy()
        values = np.zeros(state_batch.shape[0])
        assert getattr(self, 'value_net', None) is not None
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
    def preprocess(self, traj):
        pass

    def compute_loss(self, minibatch, critic, minimize_reward=False, pg_coef=1.0):
        
        mb_adv = minibatch.outcome.adv
        # normalize adv
        mb_adv = (mb_adv - mb_adv.mean())/(mb_adv.std()+1e-8)

        dist = self.calc_action_dist(minibatch.inp.data.obs, minibatch.inp.data.z)
        logprob = dist.log_prob(minibatch.decision.action)
        # calc entropy and ratio
        entropy = dist.entropy()
        ratio = (logprob - minibatch.decision.logprob).exp()

        # ppo losst
        pg_loss1 = mb_adv * ratio
        pg_loss2 = mb_adv * torch.clamp(ratio, 1-self.config.clip_param, 1+self.config.clip_param)
        pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

        entropy_loss = entropy.mean()

        # value loss
        cur_values = critic(minibatch.inp.data.state)
        v_loss_unclipped = torch.pow(minibatch.outcome.ret-cur_values, 2)
        v_clipped = minibatch.decision.value + torch.clamp(cur_values - minibatch.decision.value, -self.config.clip_param, self.config.clip_param) 
        v_loss_clipped = torch.pow(minibatch.outcome.ret-v_clipped, 2)
        v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

        # discrim loss
        discrim_loss = 0.0
        if self.config.discrim_coef>0:
            z = minibatch.inp.data.z
            int_z = z.argmax(-1)
            if isinstance(self.act_space, Discrete):
                pred_z = self.discrim(minibatch.inp.data.obs, dist.probs)
            elif isinstance(self.act_space, Box):
                pred_z = self.discrim(minibatch.inp.data.obs, torch.cat([dist.mean, dist.scale_tril.diagonal(dim1=1,dim2=2)],axis=1))
            else:
                raise NotImplementedError
            discrim_loss = nn.functional.cross_entropy(pred_z, int_z)

        loss = (-1)**minimize_reward * pg_coef * pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef + self.config.discrim_coef * discrim_loss
        return loss

    def compute_critic_loss(self, minibatch, critic):
        cur_values = critic(minibatch.inp.data.state)
        v_loss_unclipped = torch.pow(minibatch.outcome.ret-cur_values, 2)
        v_clipped = minibatch.decision.value + torch.clamp(cur_values - minibatch.decision.value, -self.config.clip_param, self.config.clip_param) 
        v_loss_clipped = torch.pow(minibatch.outcome.ret-v_clipped, 2)
        v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
        return v_loss * self.config.vf_coef

    def compute_pg_loss(self, minibatch, minimize_reward=False, pg_coef=1.0):
        mb_adv = minibatch.outcome.adv
        # normalize adv
        mb_adv = (mb_adv - mb_adv.mean())/(mb_adv.std()+1e-8)

        dist = self.calc_action_dist(minibatch.inp.data.obs, minibatch.inp.data.z)
        logprob = dist.log_prob(minibatch.decision.action)
        # calc entropy and ratio
        entropy = dist.entropy()
        ratio = (logprob - minibatch.decision.logprob).exp()

        # ppo losst
        pg_loss1 = mb_adv * ratio
        pg_loss2 = mb_adv * torch.clamp(ratio, 1-self.config.clip_param, 1+self.config.clip_param)
        pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

        entropy_loss = entropy.mean()

        # discrim loss
        discrim_loss = 0.0
        if self.config.discrim_coef>0:
            z = minibatch.inp.data.z
            int_z = z.argmax(-1)
            if isinstance(self.act_space, Discrete):
                pred_z = self.discrim(minibatch.inp.data.obs, dist.probs)
            elif isinstance(self.act_space, Box):
                pred_z = self.discrim(minibatch.inp.data.obs, torch.cat([dist.mean, dist.scale_tril.diagonal(dim1=1,dim2=2)],axis=1))
            else:
                raise NotImplementedError
            discrim_loss = nn.functional.cross_entropy(pred_z, int_z)

        loss = (-1)**minimize_reward * pg_coef * pg_loss - self.config.ent_coef * entropy_loss + self.config.discrim_coef * discrim_loss
        return loss

    def learn(self, batches, critics, pg_mask, value_mask):
        # sample xp_batch here (not separately)
        # sample from sp_batch (mb_size)
        # sample from all xp_batches (num_xp_batches * mb_size)
        # compute loss for each sample

        indices = [np.arange(batch.inp.data.obs.shape[0]) if batch is not None else [None] for batch in batches]
        [np.random.shuffle(ind) for ind in indices]
        batch_size = max([batch.inp.data.obs.shape[0] if batch is not None else 0 for batch in batches])
        mb_size = getattr(self.config,'mb_size',None) or int(np.ceil(batch_size/self.config.num_mb))

        for start in range(0, batch_size, mb_size):
            losses = []
            for i, (batch,critic,pm,vm) in enumerate(zip(batches,critics,pg_mask,value_mask)):
                if batch is None:
                    continue
                if not (pm or vm):
                    continue
                # sample a minibatch for each batch
                start_i = start
                if len(indices[i]) <= start:
                    start_i = 0
                end = start + mb_size
                mb_idx = indices[i][start_i:end]
                mb = batch[mb_idx].to(self.training_device)

                loss = 0.0
                if vm:
                    loss += vm * self.compute_critic_loss(mb, critic)
                # assuming the last batch is sp_batch and others are from xp_batches
                if pm:
                    if i!=len(batches)-1:
                        minimize_reward = True
                        pg_coef = self.xp_coef
                    else:
                        minimize_reward = False
                        pg_coef = 1.0
                    if pg_coef != 0:
                        loss += self.compute_pg_loss(mb, minimize_reward, pg_coef)
                losses.append(loss)

            for opt in self.opts:
                opt.zero_grad(set_to_none=True)

            stacked_losses = torch.stack(losses)
            xp_loss = torch.sum(stacked_losses[:-1])
            sp_loss = stacked_losses[-1]
            loss = sp_loss + xp_loss
            loss.backward()

            (nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip) for net in [self.policy_net, self.sp_critic, *self.xp_critics])
            for opt in self.opts:
                opt.step()
            
    def train(self, sp_rollout, *, away_rollouts=None, home_rollouts=None, pg_mask=None, value_mask=None, **kwargs):
        self.cur_iter += 1
        if self.use_gpu:
            for net in self.nets:
                net.to('cuda')
                self.device = 'cuda'

        # centralized critic uses states instead of obs
        # traj is an agent view of an episode
        # need to flatten the rollout to agent view traj first
        with torch.no_grad():
            sp_batch = None
            for p in sp_rollout.inp.data:
                batch = getattr(sp_rollout, p)
                next_state = batch.outcome.state[-1:]
                next_value = self.sp_critic(torch.tensor(next_state, dtype=torch.float, device=self.device)).squeeze().detach().cpu().numpy()
                compute_gae(batch, self.config.gamma, self.config.gae_lambda, batch.decision.value, next_value)
                if sp_batch is None:
                    sp_batch = batch
                else:
                    sp_batch = arrdict.cat([sp_batch, batch],axis=0)
            sp_batch = arrdict.torchify(sp_batch).to(self.device)

            # convert home_rollouts (list) and away_rollouts (list) to a batch data
            # then compute the advantage
            xp_batches = [None] * self.config.pop_size
            if (away_rollouts is not None) or (home_rollouts is not None):
                for r in [away_rollouts, home_rollouts]:
                    for i, batch in enumerate(r):
                        if batch is None:
                            continue
                        next_state = batch.outcome.state[-1:]
                        next_value = self.xp_critics[i](torch.tensor(next_state, dtype=torch.float32, device=self.device)).squeeze().detach().cpu().numpy()
                        compute_gae(batch, self.config.gamma, self.config.gae_lambda, batch.decision.value, next_value)
                        batch = arrdict.torchify(batch).to(self.device)
                        if xp_batches[i] is None:
                            xp_batches[i] = batch
                        else:
                            xp_batches[i] = arrdict.cat([xp_batches[i], batch], axis=0)

        batches = xp_batches + [sp_batch]
        critics = self.xp_critics + [self.sp_critic]

        if pg_mask is None:
            pg_mask = [None] * len(xp_batches)
        if value_mask is None:
            value_mask = [None] * len(xp_batches)
        
        pg_mask = np.concatenate([pg_mask, np.array([1.])])
        value_mask = np.concatenate([value_mask, np.array([1.])])

        assert len(batches)==len(critics)==len(pg_mask)==len(value_mask), f'{len(batches)},{len(critics)},{len(pg_mask)},{len(value_mask)}'

        for net in self.nets:
            net.to(self.training_device)
        for _ in range(self.epochs):
            self.learn(batches, critics, pg_mask, value_mask)

        for net in self.nets:
            net.to('cpu')
        if self.use_gpu:
            self.device = 'cpu'
    
    def reset(self):
        pass

class PolicyNet(nn.Module):
    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.config = config
        obs_size = obs_space.n if isinstance(obs_space, Discrete) else np.prod(obs_space.shape)
        hidden_size = getattr(config,'hidden_size',64)
        num_hidden = getattr(2, 'num_hidden', 2)
        self.logstd = None
        self.use_cnn = False
        
        if isinstance(action_space, Discrete):
            act_size = action_space.n
        elif isinstance(action_space, Box):
            act_size = np.prod(action_space.shape)
            self.logstd = nn.Parameter(torch.zeros(act_size) + np.log(config.pol_init_var))
        else:
            raise NotImplementedError
        
        self.layers = FCLayers(obs_size+config.z_dim, hidden_size, num_hidden, act_size, head_std=0.01)

    def forward(self, inp, z):
        inp = torch.cat([inp, z],axis=-1)
        out = self.layers(inp)
        return out if self.logstd is None else (out, self.logstd)


class CentralizedValueNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.joint_action_size = self.config.act_space.n if isinstance(self.config.act_space, Discrete) else np.prod(self.config.act_space.shape)
        hidden_size = getattr(config, 'hidden_size', 64)
        num_hidden = getattr(config, 'num_hidden', 2)
        self.use_cnn = False
        self.layers = FCLayers(config.state_shape[0], hidden_size, num_hidden, 1, head_std=1.0)

    def forward(self, inp):
        if self.use_cnn:
            x = self.cnn(inp).flatten(start_dim=1)
            out = self.fc(x)
            return out.view(-1)
        return self.layers(inp).view(-1)

class Discriminator(nn.Module):
    '''
    Per agent discriminator only looks at local observation
    (or action distribution) from each agent
    '''
    def __init__(self,config):
        super().__init__()
        self.config = config
        hidden_size = getattr(config, 'discrim_hidden_size', 64)
        obs_size = config.obs_space.n if isinstance(config.obs_space, Discrete) else np.prod(config.obs_space.shape)
        n_actions = config.act_space.n if isinstance(config.act_space, Discrete) else np.prod(config.act_space.shape) * 2
        inp_dim = obs_size + n_actions
        self.layers = FCLayers(inp_dim, hidden_size, 2, output_size=config.z_dim)
    
    def forward(self, obs, a_probs):
        # obs -> [bs, feature]
        # probs -> [bs, n_actions]
        inp = torch.cat([obs, a_probs], axis=-1)
        z = self.layers(inp)
        return z
    