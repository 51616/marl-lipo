import random
from collections import deque

import numpy as np
from gym.spaces import Discrete, Box
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from coop_marl.utils import Arrdict, Dotdict, arrdict, ortho_layer_init, dict_to_tensor,\
                            chop_into_episodes, get_logger
from coop_marl.agents import Agent, PGAgent
from coop_marl.models.modules import FCLayers, HyperNet

logger = get_logger()


'''
This QMIX implementation is based on: https://github.com/ray-project/ray/tree/master/rllib/agents/qmix
and https://github.com/oxwhirl/pymarl
'''

def linear_schedule(start_e, end_e, duration, t):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def normal_loglh(x, mu, var):
    return (-1/2) * ( torch.log(var * 2 * np.pi + 1e-8) + (x-mu)**2 / (var + 1e-8) )

'''
from infoGAN implementation
https://github.com/Natsu6767/InfoGAN-PyTorch/blob/4586919f2821b9b2e4aeff8a07c5003a5905c7f9/utils.py#L15
'''
class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):
        # x, mu, var -> [bs, z_dim]
        assert x.shape==mu.shape==var.shape, f'The shape of the target z ({x.shape}) must match the shape of mean ({mu.shape}) and var ({var.shape})'
        # logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        # -(1/2) * torch.log(np.pi*2) -(1/2) * torch.log(var) - 1/(2 * var) * (x-mu).square()
        # loglh = (-1/2) * ( torch.log(var * 2 * np.pi + 1e-8) + (x-mu)**2 / (var + 1e-8) )
        loglh = normal_loglh(x, mu, var)
        nll = -loglh.sum(1) # [bs]
        # loglh_prior = -normal_loglh(x, torch.zeros_like(x), torch.ones_like(x)).sum(1) # entropy term
        # nll -= loglh_prior
        return nll

class EpisodicReplayBuffer:
    '''
    Similar to ReplayBuffer but one sample is an episode
    '''
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def put(self,episodes):
        assert isinstance(episodes, list)
        self.buffer.extend(episodes)

    def sample(self, num_episodes):
        return random.sample(self.buffer, k=num_episodes)

    @property
    def size(self):
        return len(self.buffer)


class QMIXAgent(Agent):
    '''
    QMIX agent should only be used with PSController (parameter-sharing)
    '''
    def __init__(self, config):
        self.validate_config(config)
        self.config = config
        self.batch_size = config.batch_size
        self.n_agents = config.n_agents
        self.act_space = config.act_space
        self.cur_step = 0
        self.train_iter = 0
        self.explore = True
        self.z = None
        self.eps = linear_schedule(config.start_e, config.end_e,
                                   config.explore_decay_ts, self.cur_step)
        self.grad_clip = getattr(config, 'grad_clip', 0.5)
        self.config['device'] = self.device = getattr(config, 'device', 'cpu')
        self.config['training_device'] = self.training_device = getattr(config, 'training_device', 'cpu')
        # create utility and mixer net
        self.model = DRQN(config).to(self.device)
        self.target_model = DRQN(config).to(self.device)

        self.params = list(self.model.parameters())
        self.mixer =  QMixer(config).to(self.device)
        self.target_mixer = QMixer(config).to(self.device)
        self.params += list(self.mixer.parameters())

        if config.maven:
            self.discrim = Discriminator(config).to(self.device)
            self.params += list(self.discrim.parameters())
            self.discrim_loss = nn.CrossEntropyLoss(reduction='none')
            if not self.config.z_discrete:
                self.discrim_loss = NormalNLLLoss()

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.opt = optim.Adam(params=self.params,
                           lr=config.lr,
                           eps=1e-5
                           )
        self.replay_buffer = EpisodicReplayBuffer(config.buffer_size)

    def get_param(self):
        state = {'model_param': self.model.state_dict(),
                 'target_model_param': self.target_model.state_dict(),
                 'mixer_param': self.mixer.state_dict(),
                 'target_mixer_param': self.target_mixer.state_dict(),}
        if self.config.maven:
            state['discrim_param'] = self.discrim.state_dict()
        return state

    def set_param(self, param):
        self.model.load_state_dict(param['model_param'])
        self.target_model.load_state_dict(param['target_model_param'])
        self.mixer.load_state_dict(param['mixer_param'])
        self.target_mixer.load_state_dict(param['target_mixer_param'])
        if self.config.maven:
            self.discrim.load_state_dict(param['discrim_param'])


    def get_state(self):
        state = self.get_param()
        state['config'] = self.config
        state['cur_step'] = self.cur_step
        state['eps'] = self.eps
        logger.debug('Get state called')
        logger.debug(f'Explore: {self.explore}')
        logger.debug(f'cur_step:{self.cur_step}')
        logger.debug(f'eps:{self.eps}')
        return state

    def load_state(self, state):
        self.set_param(state)
        self.config = state['config']
        self.cur_step = state['cur_step']
        self.eps = linear_schedule(self.config.start_e, self.config.end_e,
                                   self.config.explore_decay_ts, self.cur_step)
        if state['eps'] < self.eps:
            self.eps = state['eps']
        logger.debug('Load state called')
        logger.debug(f'Explore: {self.explore}')
        logger.debug(f'cur_step:{self.cur_step}')
        logger.debug(f'eps:{self.eps}')
        if 'z' in state:
            self.z = state['z']

    @torch.no_grad()
    def act(self, inp):        
        obs_dict = inp.data.obs
        obs_batch = dict_to_tensor(obs_dict, device=self.device)
        n_agents = obs_batch.shape[0]
        prev_act_batch = dict_to_tensor(inp.prev_decision.action, device=self.device, dtype=torch.long)
        prev_act_onehot = F.one_hot(prev_act_batch, num_classes=self.act_space.n)
        net_inp = torch.cat([obs_batch,prev_act_onehot], axis=-1)
        h_in = dict_to_tensor(inp.prev_decision.h_out, device=self.device, axis=1) # [num_rnns, n_agents, hidden_dim]

        z = None
        if getattr(self.config, 'maven', False):
            assert getattr(inp.data,'z',False), f'The data must contain z for maven agent'
            z = dict_to_tensor(inp.data.z, device=self.device).unsqueeze(1) # [bs, ts, z_dim]
            if self.z is not None:
                z = self.z.repeat(obs_batch.shape[0],1,1)

        q,h_out = self.model(net_inp, h_in, z)
        a_max = q.squeeze(1).argmax(dim=-1).tolist()
        if self.explore:
            noise = np.random.random(size=n_agents)
            random_action = np.random.randint(low=0,high=self.act_space.n, size=n_agents)
            actions = np.where(noise<self.eps,random_action,a_max)
            if self.eps != self.config.end_e:
                self.update_eps(1)
        else:
            actions = a_max
        h_in = h_in.cpu().numpy().transpose([1,0,2])
        h_out = h_out.cpu().numpy().transpose([1,0,2])
        if getattr(self.config, 'get_q_values', False):
            q = q.cpu().numpy()
            return [Arrdict(action=a,h_in=hi,h_out=ho,q_values=q_val) for a,hi,ho,q_val in zip(actions,h_in,h_out,q)]
        else:
            return [Arrdict(action=a,h_in=hi,h_out=ho) for a,hi,ho in zip(actions,h_in,h_out)]

    def update_eps(self, taken_step=1):
        self.cur_step += taken_step
        self.eps = linear_schedule(self.config.start_e, self.config.end_e,
                                   self.config.explore_decay_ts, self.cur_step)

    def _preprocess(self, traj):
        '''
        Put traj into EpisodicReplayBuffer
        '''
        episodes = chop_into_episodes(traj)
        self.replay_buffer.put(episodes)

    def train(self, *args, **kwargs):
        self._preprocess(args[0])
        if self.replay_buffer.size < self.batch_size:
            print('skip trainng as the buffer has not reached the batch size')
            return
        self.target_model.to(self.training_device)
        self.target_mixer.to(self.training_device)
        self.model.train().to(self.training_device)
        self.mixer.train().to(self.training_device)
        if self.config.maven:
            self.discrim.train().to(self.training_device)

        self.train_iter += 1
        # sample episodes from replay buffer ...
        episodes = self.replay_buffer.sample(self.batch_size)
        # add mask for cases where episodes' length is not the same
        masks = []
        maxlen = max([max(list(ep.inp.data.obs.shape[0].values())) for ep in episodes])
        for ep in episodes:
            mask = torch.ones(maxlen, dtype=bool)
            padsize = maxlen - max(list(ep.inp.data.obs.shape[0].values()))
            if padsize > 0:
                mask[-padsize:] = False
            masks.append(mask)
        # masks is a stack of masks per episode [n_eps,ts]
        masks = torch.stack(masks).to(self.training_device)
        padded_eps = [arrdict.postpad(ep,maxlen,dim=0) for ep in episodes]
        # change players key to additional dimension of the episodes
        ep_list = []
        for ep in padded_eps:
            ep_list.append(arrdict.stack([getattr(ep,p) for p in ep.inp.data],axis=0))
        n_eps = len(ep_list)
        # input the the drqn net will have this shape: [n_eps * n_agents, ts (padded), obs]
        rnn_batch = arrdict.torchify(arrdict.stack(ep_list, axis=0)).to(self.training_device).view(n_eps*self.n_agents, maxlen, -1) # [n_eps*n_agents, ts, feature]

        assert 'state' in rnn_batch.inp.data, f'QMIX requires state information from the environment'
        rnn_batch.inp.data['state'] = rnn_batch.inp.data.state.view(n_eps,self.n_agents,maxlen,-1)[:,0]
        rnn_batch.outcome['state'] = rnn_batch.outcome.state.view(n_eps,self.n_agents,maxlen,-1)[:,0]

        # h0 shape: [num_layers(1), n_eps*n_agents, hidden_size]
        init_h = torch.tensor(self.model.get_rnn_init_state(), device=self.training_device).view(1,1,-1).expand(1,n_eps*self.n_agents,-1).contiguous()
        prev_act_batch = rnn_batch.inp.prev_decision.action.squeeze(-1)
        prev_act_onehot = F.one_hot(prev_act_batch.long(), num_classes=self.act_space.n)

        net_inp = torch.cat([rnn_batch.inp.data.obs, prev_act_onehot],axis=-1)

        # compute q for each timestep/agent
        z = None
        if getattr(self.config, 'maven', False):
            z = rnn_batch.inp.data.z # [n_eps*n_agents,ts,z_dim]
        q, _ = self.model(net_inp, init_h, z) # [n_eps*n_agents,ts,num_actions]
        q_vals = torch.gather(q,dim=-1,index=rnn_batch.decision.action.long()).squeeze(-1).reshape(n_eps,self.n_agents,maxlen) # [n_eps,n_agents,ts]
        q_vals = q_vals.transpose(1,2).contiguous() # [n_eps, ts, num_agents]

        # compute q target for each ts/agent
        act_batch = rnn_batch.decision.action.squeeze(-1)
        act_onehot = F.one_hot(act_batch.long(), num_classes=self.act_space.n)
        target_net_inp = torch.cat([rnn_batch.outcome.obs, act_onehot],axis=-1)
        
        if 'action_mask' in rnn_batch.inp.data:
            raise NotImplementedError(f'Action mask is not handled currently')

        # compute q_max for q_target
        # compute double-q target
        # Assume the same z for entire episode (thus using the same z as the online model)
        q_target, _ = self.target_model(target_net_inp, init_h, z) # [n_eps*n_agents,ts,num_actions]
        q_target = q_target.detach()
        dq, _ = self.model(target_net_inp, init_h, z)
        cur_max_action = dq.detach().max(dim=-1, keepdim=True)[1]
        q_target_vals = torch.gather(q_target,dim=-1,index=cur_max_action).squeeze(-1).view(n_eps,self.n_agents,maxlen)
        q_target_vals = q_target_vals.transpose(1,2) # [n_eps, ts, num_agents]

        # compute q_tot using mixer
        ep_z = None
        if self.config.maven:
            assert z.shape == (n_eps*self.n_agents, maxlen, self.config.z_dim)
            ep_z = z.reshape(n_eps, self.n_agents, maxlen, self.config.z_dim)[:,0] # [n_eps, maxlen, z_dim]
            ep_z = ep_z.reshape(n_eps*maxlen, self.config.z_dim)
        q_tot_vals = self.mixer(q_vals, rnn_batch.inp.data.state, ep_z).squeeze(-1)


        q_target_tot_vals = self.target_mixer(q_target_vals, rnn_batch.outcome.state, ep_z).squeeze(-1)
        # assume team reward (all agents have the same reward)
        team_reward = rnn_batch.outcome.reward.view(n_eps, self.n_agents, maxlen)[:,0]
        # terminated is the team termination flag and is true only when all agents are done
        terminated = rnn_batch.outcome.done.view(n_eps, self.n_agents, maxlen).prod(1)
        # Calculate 1-step Q-Learning targets
        targets = team_reward + self.config.gamma * (1 - terminated) * q_target_tot_vals

        # Td-error
        td_error = (q_tot_vals - targets.detach())

        # 0-out the targets that came from padded data
        masked_td_error = td_error * masks
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.config.maven and self.config.discrim_coef > 0:
            z_pred, z_mean, z_var = self.discrim(q, rnn_batch.inp.data.state, training=True)
            assert z_pred.shape==(n_eps, maxlen, self.config.z_dim)
            z_pred = z_pred.reshape(-1, self.config.z_dim)
            if self.config.z_discrete:
                ep_z_int = ep_z.argmax(-1)
                discrim_loss = self.discrim_loss(z_pred, ep_z_int)
            else:
                z_mean = z_mean.reshape(-1, self.config.z_dim)
                z_var = z_var.reshape(-1, self.config.z_dim)
                discrim_loss = self.discrim_loss(z_mean, ep_z, z_var)
            discrim_loss = discrim_loss.reshape(n_eps, maxlen)
            masked_discrim_loss = (discrim_loss * masks).sum() / mask.sum()
            loss += self.config.discrim_coef * masked_discrim_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.opt.step()
        self.target_model.to(self.device)
        self.target_mixer.to(self.device)
        self.model.eval().to(self.device)
        self.mixer.eval().to(self.device)
        if self.config.maven:
            self.discrim.eval().to(self.device)

        if self.train_iter % self.config.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def get_prev_decision_view(self):
        return Arrdict(action=self.act_space.sample(),
                       h_in=self.model.get_rnn_init_state(),
                       h_out=self.model.get_rnn_init_state(),
                       )

    def reset(self):
        self.model.eval()

    def validate_config(self, config):
        pass

class DRQN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_agents = config.n_agents
        self.hidden_dim = config.hidden_dim
        input_shape = config.obs_space.n if isinstance(config.obs_space, Discrete) else np.prod(config.obs_space.shape)
        act_size = config.act_space.n

        self.fc1 = ortho_layer_init(nn.Linear(input_shape+act_size, self.hidden_dim))

        self.rnn = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, np.sqrt(2))

        if getattr(config, 'maven', False):
            self.fc2 = HyperNet(self.hidden_dim, config.z_dim, config.hypernet_embed,
                                 num_heads=1,
                                 head_sizes=[act_size],
                                 head_std=[0.05])
        else:
            self.fc2 = ortho_layer_init(nn.Linear(self.hidden_dim, act_size))

    def get_rnn_init_state(self):
        # make hidden states on same device as model
        return np.zeros([1,self.hidden_dim],dtype=np.float32)

    def forward(self, inp, h_in, z=None):
        if len(inp.shape)==2:
            # add time dimension
            inp = inp.unsqueeze(1)
        x = F.elu(self.fc1(inp))
        h, state_out = self.rnn(x, h_in)

        if getattr(self.config, 'maven', False):
            assert z is not None, f'z must be provided for maven agent'
            q, _ = self.fc2(h,z)
        else:
            q = self.fc2(h)
        return q, state_out

class QMixer(nn.Module):
    def __init__(self, config):
        super(QMixer, self).__init__()

        self.config = config
        self.n_agents = config.n_agents
        self.state_dim = int(np.prod(config.state_shape))
        self.inp_dim = self.state_dim
        if config.maven:
            self.inp_dim += config.z_dim
        self.embed_dim = config.mixing_embed_dim

        if getattr(config, "hypernet_layers", 2) == 1:
            self.hyper_w_1 = ortho_layer_init(nn.Linear(self.inp_dim, self.embed_dim * self.n_agents))
            self.hyper_w_final = ortho_layer_init(nn.Linear(self.inp_dim, self.embed_dim), std=0.001)
        elif getattr(config, "hypernet_layers", 2) == 2:
            hypernet_embed = self.config.hypernet_embed
            self.hyper_w_1 = nn.Sequential(ortho_layer_init(nn.Linear(self.inp_dim, hypernet_embed)),
                                           nn.ELU(),
                                           ortho_layer_init(nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
                                           )
            self.hyper_w_final = nn.Sequential(ortho_layer_init(nn.Linear(self.inp_dim, hypernet_embed)),
                                           nn.ELU(),
                                           ortho_layer_init(nn.Linear(hypernet_embed, self.embed_dim), std=0.001)
                                           )
        elif getattr(config, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = ortho_layer_init(nn.Linear(self.inp_dim, self.embed_dim))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(ortho_layer_init(nn.Linear(self.inp_dim, self.embed_dim)),
                               nn.ELU(),
                               ortho_layer_init(nn.Linear(self.embed_dim, 1))
                               )

    def forward(self, agent_qs, states, ep_z=None):
        # agnet_qs -> [bs, ts, n_agents]
        # states -> [bs, ts, features]
        # ep_z -> [bs, ts, z_dim]
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim) # [bs*ts, features]
        if ep_z is not None:
            ep_z = ep_z.reshape(-1, self.config.z_dim)
            states = torch.cat([states, ep_z], axis=-1)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents) # [bs*ts, 1, n_agents]
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

class VDNMixer(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, agent_qs, states, ep_z=None):
        return torch.sum(agent_qs,axis=-1,keepdim=True)

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        stack_shape = config.n_agents * config.act_space.n + int(np.prod(config.state_shape))
        self.rnn = nn.GRU(input_size=stack_shape, hidden_size=config.discrim_hidden_dim, batch_first=True) # stack_shape -> rnn_out
        self.layers = FCLayers(config.discrim_hidden_dim,
                               config.discrim_hidden_dim,
                               2,
                               output_size=config.z_dim) # rnn_out_shape -> z_dim
        if not self.config.z_discrete:
            self.log_var_layers = nn.Sequential(ortho_layer_init(nn.Linear(config.discrim_hidden_dim, config.discrim_hidden_dim)),
                                                nn.ELU(),
                                                ortho_layer_init(nn.Linear(config.discrim_hidden_dim, config.discrim_hidden_dim)),
                                                nn.ELU(),
                                                ortho_layer_init(nn.Linear(config.discrim_hidden_dim, config.z_dim), bias_const=1.0),
                                                )

    def forward(self, q, states, training=False):
        # q -> [n_eps*n_agents, ts, act_size]
        # states -> [n_eps, ts, state_size]
        n_eps, ts, _ = states.shape
        assert q.shape[0]%n_eps==0
        n_agents = q.shape[0]//n_eps
        act_size = q.shape[-1]
        # compute softmax_q 
        # cat all softmax_q and states
        sm_q = F.softmax(q, dim=-1)
        sm_q = sm_q.reshape(n_eps, n_agents, ts, act_size).permute(0,2,1,3)
        sm_q = sm_q.reshape(n_eps, ts, n_agents * act_size)

        rnn_inp = torch.cat([sm_q, states],axis=-1) # [n_eps, ts, n_agents*act_size + state_size]
        state_in = self.get_rnn_init_state(n_eps).to(self.config.device if not training else self.config.training_device)
        x, state_out = self.rnn(rnn_inp, state_in)
        mean = None
        var = None
        # x -> [n_eps, ts, act_size]
        if self.config.z_discrete:
            # forward
            z = self.layers(x) # [n_eps, ts, z_dim]
        else:
            # continuous z
            mean = self.layers(x) # [n_eps, ts, z_dim]
            log_var = self.log_var_layers(x) # [n_eps, ts, z_dim]
            var = log_var.exp()
            std = torch.sqrt(var)
            z = mean + std * torch.normal(0, 1, size=std.shape).to(self.config.device)

        return z, mean, var

    def get_rnn_init_state(self, bs):
        return torch.zeros(1, bs, self.config.discrim_hidden_dim)

if __name__ == '__main__':
    from gym import spaces
    from coop_marl.envs import registered_envs
    from coop_marl.controllers import PSController
    from coop_marl.runners import StepsRunner, EpisodesRunner
    from coop_marl.envs.wrappers import StateWrapper, AgentIDWrapper, ZWrapper
    from tqdm import tqdm

    horizon = 25
    n_iter = 1000
    num_episodes = 10
    num_eval_eps = 10
    eval_interval = 50
    n_agents = 1
    num_landmarks = 2
    z_dim = 8

    env = registered_envs['simple_fixed'](horizon=horizon, num_landmarks=num_landmarks, n_agents=n_agents)
    eval_env = registered_envs['simple_fixed'](horizon=horizon, num_landmarks=num_landmarks, n_agents=n_agents)

    env = AgentIDWrapper(StateWrapper(ZWrapper(env, z_dim=z_dim, z_discrete=True)))
    eval_env = AgentIDWrapper(StateWrapper(ZWrapper(eval_env, z_dim=z_dim, z_discrete=True)))

    obs_space = list(env.observation_spaces.values())[0]['obs'] 
    if isinstance(obs_space, spaces.Box):
        obs_space.shape = (obs_space.shape[0] + n_agents,)  # concat agent id
    else:
        obs_space.n = obs_space.n + n_agents
    print('obs_space:',obs_space)

    act_space = list(env.action_spaces.values())[0]
    state_shape = env.get_state_shape(num_landmarks, n_agents)
    print('state_shape:',state_shape)

    config = Dotdict(obs_space=obs_space,
                 act_space=act_space,
                 hidden_dim=64,
                 mixing_embed_dim=32,
                 hypernet_embed=128,
                 n_agents=n_agents,
                 buffer_size=1000,
                 batch_size=128,
                 state_shape=state_shape,
                 maven=True,
                 discrim_coef=10,
                 z_dim=z_dim,
                 discrim_hidden_dim=64,
                 lr=1e-3,
                 gamma=0.99,
                 start_e=1,
                 end_e=0.05,
                 explore_decay_ts=100000,
                 target_update_freq=25,
                 )

    agent = QMIXAgent(config)
    controller = PSController(env.action_spaces, agent)
    runner = EpisodesRunner(env, controller)
    eval_runner = EpisodesRunner(eval_env, controller)

    with tqdm(range(n_iter)) as t:
        for it in t:
            traj, *_ = runner.rollout(num_episodes)
            controller.train(traj, flatten=False)
            
            if it % eval_interval==0:
                eval_traj, *_ = eval_runner.rollout(num_eval_eps, render=True)
                eval_mean_rew = np.sum(eval_traj.outcome.agent_0.reward)/num_eval_eps
                t.set_postfix(eval_mean_rew=eval_mean_rew)
