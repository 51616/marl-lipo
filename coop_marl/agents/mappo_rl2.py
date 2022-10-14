import numpy as np
from gym.spaces import Discrete, Box
import torch
from torch import nn
from torch import optim
import torch.distributions as D
import torch.nn.functional as F

from coop_marl.utils import Arrdict, arrdict, compute_gae, ortho_layer_init, dict_to_tensor,\
                            FLOAT_MIN, FLOAT_MAX, get_logger
from coop_marl.agents import Agent

logger = get_logger()

class MAPPORL2Agent(Agent):
    def __init__(self, config):
        self.obs_space = config.obs_space
        self.act_space = config.act_space
        self.act_size = config.act_space.n if isinstance(config.act_space, Discrete) else np.prod(config.act_space.shape)
        self.config = config
        self.device = getattr(config, 'device', 'cpu')
        self.training_device = getattr(config, 'training_device', 'cpu')
        self.n_grad_cum = getattr(config, 'n_grad_cum', 0)
        self.grad_clip = getattr(config, 'grad_clip', 0.5)
        assert getattr(config,'num_seq_mb',False) ^ getattr(config,'num_mb',False), \
        f'Use either mb_size or num_mb'
        self.policy_net = PolicyNet(self.obs_space, self.act_space, config) 
        self.value_net = CentralizedValueNet(config) 

        self.opt = optim.Adam(list(self.value_net.parameters()) + list(self.policy_net.parameters()), lr=config.lr, eps=1e-5)
        self.cur_iter = 0

    def get_param(self):
        return {'critic_param':self.value_net.state_dict(),
                'policy_param':self.policy_net.state_dict()}

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

    def get_prev_decision_view(self):
        return Arrdict(action=self.act_space.sample(),
                       h_in_actor=self.policy_net.get_rnn_init_state(),
                       h_in_critic=self.value_net.get_rnn_init_state(),
                       h_out_actor=self.policy_net.get_rnn_init_state(),
                       h_out_critic=self.value_net.get_rnn_init_state(),
                       )

    def calc_action_dist(self, inp, h_in_actor):
        if isinstance(self.act_space, Discrete):
            logits, h_out_actor = self.policy_net(inp, h_in_actor)
            dist = D.Categorical(logits=logits)

        elif isinstance(self.act_space, Box):
            mean, logstd, h_out_actor = self.policy_net(inp, h_in_actor)
            std = torch.exp(logstd)
            dist = D.MultivariateNormal(mean, scale_tril=torch.diag(std))
        
        return dist, h_out_actor

    @torch.no_grad()
    def act(self, inp):
        obs_batch = dict_to_tensor(inp.data.obs, device=self.device)
        state_batch = dict_to_tensor(inp.data.state, device=self.device)
        prev_act_batch = dict_to_tensor(inp.prev_decision.action, device=self.device, dtype=torch.long)
        if isinstance(self.act_space, Discrete):
            prev_act_batch = F.one_hot(prev_act_batch, num_classes=self.act_size)
        prev_rew_batch = dict_to_tensor(inp.data.reward, device=self.device) # .unsqueeze(-1)
        if len(prev_rew_batch.shape)==1:
            prev_rew_batch = prev_rew_batch.unsqueeze(-1)

        policy_inp = torch.cat([obs_batch,prev_act_batch,prev_rew_batch], axis=-1)
        value_inp = torch.cat([state_batch, prev_act_batch,prev_rew_batch], axis=-1)
        if self.config.critic_use_local_obs:
            value_inp = torch.cat([obs_batch, value_inp], axis=-1)

        h_in_actor = dict_to_tensor(inp.prev_decision.h_out_actor, device=self.device, axis=2)
        h_in_critic = dict_to_tensor(inp.prev_decision.h_out_critic, device=self.device, axis=2)

        dist, h_out_actor = self.calc_action_dist(policy_inp, h_in_actor)
        values, h_out_critic = self.value_net(value_inp, h_in_critic)

        action_mask = getattr(inp.data, 'action_mask', None)
        if action_mask is not None:
            mask_tensor = dict_to_tensor(action_mask, device=self.device)
            dist.logits += torch.clamp(torch.log(mask_tensor), FLOAT_MIN, FLOAT_MAX)
        actions = dist.sample()
        logprobs = dist.log_prob(actions).detach().cpu().numpy()
        actions = actions.detach().cpu().numpy()

        values = values.detach().cpu().numpy()
        # values -> [B,1]
        # h_critic -> [num_layers,B,hidden_dim]

        # move agent dim to the first dim
        h_out_critic = h_out_critic.detach().cpu().numpy().transpose([2,0,1,3])
        h_out_actor = h_out_actor.detach().cpu().numpy().transpose([2,0,1,3])
        h_in_actor = h_in_actor.cpu().numpy().transpose([2,0,1,3])
        h_in_critic = h_in_critic.cpu().numpy().transpose([2,0,1,3])
        if getattr(self.config, 'get_act_dist', False):
            if isinstance(self.act_space, Discrete):
                dists = dist.probs.detach().cpu().numpy()
                return [Arrdict(action=a, logprob=l, value=v,
                                h_in_actor=hia, h_in_critic=hic,
                                h_out_actor=hoa, h_out_critic=hoc,
                                act_dist=dist) \
                                for a,l,v,hia,hic,hoa,hoc,dist in zip(actions,logprobs,values,h_in_actor,h_in_critic,
                                                            h_out_actor,h_out_critic,dists)]
            else:
                raise NotImplementedError
        else:
            return [Arrdict(action=a, logprob=l, value=v,
                                h_in_actor=hia, h_in_critic=hic,
                                h_out_actor=hoa, h_out_critic=hoc) \
                                for a,l,v,hia,hic,hoa,hoc in zip(actions,logprobs,values,h_in_actor,h_in_critic,
                                                            h_out_actor,h_out_critic)]

    @torch.no_grad()
    def preprocess(self,traj):
        next_value = None
        obs_batch = torch.tensor(traj.inp.data.obs[-1:], device=self.device, dtype=torch.float)
        state_batch = torch.tensor(traj.inp.data.state[-1:], device=self.device, dtype=torch.float)
        prev_act_batch = torch.tensor(traj.inp.prev_decision.action[-1:], device=self.device, dtype=torch.long)
        if isinstance(self.act_space, Discrete):
            prev_act_batch = F.one_hot(prev_act_batch, num_classes=self.act_size)
        prev_rew_batch = torch.tensor(traj.inp.data.reward[-1:], device=self.device)
        if len(prev_rew_batch.shape)==1:
            prev_rew_batch = prev_rew_batch.unsqueeze(-1)
        net_inp = torch.cat([state_batch,prev_act_batch,prev_rew_batch], axis=-1)
        if self.config.critic_use_local_obs:
            net_inp = torch.cat([obs_batch, net_inp], axis=-1)

        h_in_critic = torch.tensor(traj.inp.prev_decision.h_out_critic[-1:], device=self.device).permute([1,2,0,3]).contiguous()

        next_value, _ = self.value_net(net_inp, h_in_critic)
        next_value = next_value.squeeze().detach().cpu().numpy()
        compute_gae(traj, self.config.gamma, self.config.gae_lambda, traj.decision.value, next_value)

    @torch.no_grad()
    def create_rnn_batch(self,traj,max_len=None):
        # takes batch with original batchsize
        # turn it to [new_batch_size,seq_len,-1]
        # the timesteps could be padded (bigger than the inputs)
        # e.g. take 50 timesteps with max_len 20 will produce 3 rows of 20 timesteps seq [50x...] -> [3x20x...]
        # (non-overlapping sequences)
        if max_len is None:
            # calculate the longest traj in this batch
            max_len = 0
            last_t = -1
            t_end = np.nonzero(traj.outcome.done)[0]
            for t in t_end:
                if (t-last_t)>max_len:
                    max_len = t - last_t
                last_t = t

        rnn_batch = []
        masks = []
        done = traj.outcome.done.copy()
        # set last time step to be True
        # used to compute the last seq
        done[-1] = True
        end_indices = iter(done.nonzero()[0])
        end_idx = next(end_indices)
        start_idx = 0
        while True:
            # create a chunk that has max_len that does not cross between episodes
            # towards the end to traj the chunk would be smaller than max_len, need padding there
            if (start_idx+max_len-1) > end_idx:
                # create the last seq with padding
                seq = traj[start_idx:end_idx+1]
                seq_size = seq.inp.data.obs.shape[0]
                pad_size = max_len - seq_size
                seq = arrdict.postpad(seq,max_len,dim=0)
                padded_seq_size = seq.inp.data.obs.shape[0]
                rnn_batch.append(seq)
                # add false masks the padded tokens
                mask = np.zeros([padded_seq_size],dtype=bool)
                mask[:-pad_size] = True
                masks.append(mask)
            else:
                # create a seq with max_len length
                seq = traj[start_idx:start_idx+max_len]
                seq_size = seq.inp.data.obs.shape[0]
                rnn_batch.append(seq)
                mask = np.ones(seq_size,dtype=bool)
                masks.append(mask)

            if start_idx+max_len>=end_idx:
                start_idx = end_idx + 1
                end_idx = next(end_indices,None)
                if end_idx is None:
                    return arrdict.stack(rnn_batch), np.stack(masks)
            else:
                start_idx += max_len

    def train(self, batch):
        with torch.no_grad():
            self.cur_iter += 1
            self.preprocess(batch)
            if self.config.anneal_lr:
                frac = min(1, (self.cur_iter - 1.0) / (self.config.num_anneal_iter))
                lrnow = self.config.lr - frac * (self.config.lr - self.config.min_anneal_lr)  # max(frac * self.config.lr, self.config.min_anneal_lr)
                for g in self.opt.param_groups:
                    g['lr'] = lrnow
                self.cur_lr = lrnow
                logger.debug(f'current learning rate: {lrnow}')
            rnn_batch, masks = self.create_rnn_batch(batch, getattr(self.config,'max_len',None))
            rnn_batch = arrdict.torchify(rnn_batch).to('cpu').detach()
            masks = arrdict.torchify(masks).to('cpu').detach()
            
            batch_size = rnn_batch.inp.data.obs.shape[0]
            logger.debug(f'batch size: {batch_size}')
            # mb_size is the number of seqs in a minibatch
            num_seq_mb = getattr(self.config,'num_seq_mb',None) or getattr(self.config,'mb_size',None) or int(np.ceil(batch_size/self.config.num_mb))
        
        self.policy_net.to(self.training_device)
        self.value_net.to(self.training_device)
        for ep in range(self.config.epochs):
            idx = torch.randperm(batch_size)
            # randomly choose sub sequences from rnn_batch
            for mb_i, start in enumerate(range(0, batch_size, num_seq_mb)):
                with torch.no_grad():
                    end = start + num_seq_mb
                    mb_idx = idx[start:end]
                    rnn_mb = rnn_batch[mb_idx].to(self.training_device)
                    mb_masks = masks[mb_idx]
                    mb_adv = rnn_mb.outcome.adv
                    # normalize adv
                    mb_adv = (mb_adv - mb_adv[mb_masks].mean())/(mb_adv[mb_masks].std()+1e-8)
                    mb_adv = mb_adv.view(-1)
                    obs_batch = rnn_mb.inp.data.obs
                    state_batch = rnn_mb.inp.data.state

                    prev_act_batch = rnn_mb.inp.prev_decision.action
                    if isinstance(self.act_space, Discrete):
                        prev_act_batch = F.one_hot(prev_act_batch.long(), num_classes=self.act_size)
                    prev_rew_batch = rnn_mb.inp.data.reward
                    if len(prev_rew_batch.shape)==2:
                        prev_rew_batch = prev_rew_batch.unsqueeze(-1)
                    policy_inp = torch.cat([obs_batch,prev_act_batch,prev_rew_batch], axis=-1)
                    value_inp = torch.cat([state_batch, prev_act_batch,prev_rew_batch], axis=-1)
                    if self.config.critic_use_local_obs:
                        value_inp = torch.cat([obs_batch, value_inp], axis=-1)
                    h_in_actor = rnn_mb.decision.h_in_actor[:,0].permute([1,2,0,3]).contiguous()
                    h_in_critic = rnn_mb.decision.h_in_critic[:,0].permute([1,2,0,3]).contiguous()
                    mb_act = rnn_mb.decision.action # N, max_len, action_dim

                dist, _ = self.calc_action_dist(policy_inp, h_in_actor)
                cur_values, _ = self.value_net(value_inp, h_in_critic)

                if len(mb_act.shape)==2:
                    logprob = dist.log_prob(mb_act.view(-1))
                elif len(mb_act.shape)==3:
                    logprob = dist.log_prob(mb_act.view(-1, mb_act.shape[2]))
                # calc entropy and ratio
                ratio = (logprob - rnn_mb.decision.logprob.view(-1)).exp()
                # ppo loss
                pg_loss1 = mb_adv * ratio
                pg_loss2 = mb_adv * torch.clamp(ratio, 1-self.config.clip_param, 1+self.config.clip_param)
                pg_loss = -torch.min(pg_loss1, pg_loss2)[mb_masks.view(-1)].mean()
                entropy = dist.entropy()
                entropy_loss = entropy[mb_masks.view(-1)].mean()
                # value loss
                v_loss_unclipped = torch.pow(rnn_mb.outcome.ret.view(-1) - cur_values, 2)
                v_clipped = rnn_mb.decision.value.view(-1) + torch.clamp(cur_values - rnn_mb.decision.value.view(-1), -self.config.vf_clip_param, self.config.vf_clip_param)
                v_loss_clipped = torch.pow(rnn_mb.outcome.ret.view(-1) - v_clipped, 2)
                v_loss = torch.max(v_loss_unclipped, v_loss_clipped)[mb_masks.view(-1)].mean()

                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                if self.n_grad_cum:
                    loss = loss / self.n_grad_cum
                loss.backward()
                # clip grad
                if (not self.n_grad_cum) or (self.n_grad_cum and (mb_i % self.n_grad_cum) == 0):
                    nn.utils.clip_grad_norm_(list(self.value_net.parameters()) + list(self.policy_net.parameters()), self.grad_clip)
                    self.opt.step()
                    self.opt.zero_grad()
        self.policy_net.to(self.device)
        self.value_net.to(self.device)
        
    def reset(self):
        pass

class PolicyNet(nn.Module):
    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.config = config
        obs_size = obs_space.n if isinstance(obs_space, Discrete) else np.prod(obs_space.shape)
        self.hidden_size = getattr(config,'hidden_size',64)
        self.logstd = None
        
        if isinstance(action_space, Discrete):
            self.act_size = action_space.n
        elif isinstance(action_space, Box):
            self.act_size = np.prod(action_space.shape)
            self.logstd = nn.Parameter(torch.zeros(self.act_size) + np.log(config.pol_init_var))
        else:
            raise NotImplementedError

        a_input_size = [1, self.act_size] # prev_rew and prev_action
        rnns = []
        for i in range(0,2):
            aux_inp_size = a_input_size[i]
            if i>0:
                aux_inp_size += self.hidden_size 
            rnn = nn.GRU(input_size=obs_size + aux_inp_size, hidden_size=self.hidden_size, batch_first=True)
            for name, param in rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param, 0.01)

            rnns.append(rnn)
        self.rnns = nn.ModuleList(rnns)
        self.head = ortho_layer_init(nn.Linear(self.hidden_size,self.act_size), std=0.01)

    def forward(self, inp, rnn_state):
        if len(inp.shape)==2:
            # add time dimension
            inp = inp.unsqueeze(1)

        obs, prev_a, prev_r = inp[...,:-self.act_size-1], inp[...,-self.act_size-1:-1], inp[...,-1:]
        rnn_out = None
        rnn_states = []
        for i in range(len(self.rnns)):
            rnn_inp = torch.cat([obs,prev_r],axis=-1) if i==0 else torch.cat([obs,rnn_out,prev_a],axis=-1)
            rnn_out, h = self.rnns[i](rnn_inp, rnn_state[i])
            rnn_states.append(h)

        out = self.head(rnn_out).view(-1, self.act_size)

        rnn_state_out = torch.stack(rnn_states, axis=0)
        return (out, rnn_state_out) if self.logstd is None else (out, self.logstd, rnn_state_out)

    def get_rnn_init_state(self,batch_size=1):
        # first dim is num_rnns,second dim is num_layers of each rnn
        return np.zeros([2, 1, self.hidden_size],dtype=np.float32)


class CentralizedValueNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_dim = int(np.prod(config.state_shape))
        self.obs_size = config.obs_space.n if isinstance(config.obs_space, Discrete) else np.prod(config.obs_space.shape)
        self.inp_dim = self.state_dim
        if config.critic_use_local_obs: 
            self.inp_dim = self.state_dim + self.obs_size

        self.joint_action_size = self.config.act_space.n if isinstance(self.config.act_space, Discrete) else np.prod(self.config.act_space.shape)
        self.hidden_size = getattr(config,'hidden_size',64)

        if isinstance(config.act_space, Discrete):
            self.act_size = config.act_space.n
        elif isinstance(config.act_space, Box):
            self.act_size = np.prod(config.act_space.shape)
            self.logstd = nn.Parameter(torch.zeros(self.act_size) + np.log(config.pol_init_var))
        else:
            raise NotImplementedError

        a_input_size = [1, self.act_size] # prev_rew and prev_action
        rnns = []
        for i in range(0,2):
            aux_inp_size = a_input_size[i]
            if i>0:
                aux_inp_size += self.hidden_size 
            rnn = nn.GRU(input_size=self.inp_dim + aux_inp_size, hidden_size=self.hidden_size, batch_first=True)
            for name, param in rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param, 0.01)

            rnns.append(rnn)
        self.rnns = nn.ModuleList(rnns)
        self.head = ortho_layer_init(nn.Linear(self.hidden_size, 1), std=1.0)

    def forward(self, inp, rnn_state):
        if len(inp.shape)==2:
            # add time dimension
            inp = inp.unsqueeze(1)

        obs, prev_a, prev_r = inp[...,:-self.act_size-1], inp[...,-self.act_size-1:-1], inp[...,-1:]
        rnn_out = None
        rnn_states = []
        for i in range(len(self.rnns)):
            rnn_inp = torch.cat([obs,prev_r],axis=-1) if i==0 else torch.cat([obs,rnn_out,prev_a],axis=-1)
            rnn_out, h = self.rnns[i](rnn_inp, rnn_state[i])
            rnn_states.append(h)

        out = self.head(rnn_out).view(-1)

        rnn_state_out = torch.stack(rnn_states, axis=0)
        return out, rnn_state_out

    def get_rnn_init_state(self,batch_size=1):
        # first dim is num_rnns,second dim is num_layers of each rnn
        return np.zeros([2, 1, self.hidden_size],dtype=np.float32)
