import random
from copy import deepcopy
import numpy as np

from gym import spaces
import torch

def wrap(*, env, wrappers, **kwargs):
    for wrapper in reversed(wrappers):
        if isinstance(wrapper, str):
            env = eval(wrapper)(env, **kwargs)
        elif isinstance(wrapper, Wrapper):
            env = wrapper(env, **kwargs)
    return env

class Wrapper:
    def __init__(self, env, *args, **kwargs):
        self.env = env
        
    def __getattr__(self, key):

        if key in self.__dict__:
            return self.__dict__[key]
        elif key.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{key}'")
        else:
            return getattr(self.env, key)

    # for NormalizedWrapper that keeps the running stats
    def get_stats(self):
        try:
            return self.env.get_stats()
        except AttributeError:
            return {}

    def set_stats(self, *args, **kwargs):
        try:
            return self.env.set_stats(*args, **kwargs)
        except AttributeError:
            pass


class SARDConsistencyChecker(Wrapper): #sub-class MultiagentEnv
    # check if obs,action have a correspoind pair of agents
    # (done agents and waiting agents (do not present in obs dict) should not have action in that timestep)
    # agents that get obs must act immediately in the next timestep
    def reset(self):
        self.last_data = self.env.reset()
        return self.last_data

    def step(self, decision):
        assert getattr(self.last_data, 'obs', None) is not None, \
        f'data must contain obs field. Found {self.last_data}'
        assert decision.keys()==self.last_data.keys(), \
        f'Incoming actions contain more or less agents that get observation from the last timestep: \n\
        get {decision.keys()} when expected {self.last_data.keys()}'

        data, info = self.env.step(decision)

        assert data.reward.keys()==decision.keys(), \
        f'Each agent that took an action must get a reward: \n\
        get {data.reward.keys()} when expected {decision.keys()}'

        assert data.done.keys()==decision.keys(), \
        f'Each agent that took an action must get a done flag: \n\
        get {data.done.keys()} when expected {decision.keys()}'        

        self.last_data = data
        return data, info

class TurnBasedConsistencyChecker(SARDConsistencyChecker):
    def step(self, decision):
        assert getattr(self.last_data, 'obs', None) is not None, \
        f'data must contain obs field. Found {self.last_data}'
        assert decision.keys()==self.last_data.keys(), \
        f'Incoming actions contain more or less agents that get observation from the last timestep: \n\
        get {decision.keys()} when expected {self.last_data.keys()}'
        for k,v in self.last_data.items():
            assert ('obs' in v) and ('done' in v) and ('reward' in v), \
            f'Each agent that gets obs must also gets done and reward: player {k} got {v}'

        data, info = self.env.step(decision)
        self.last_data = data
        return data, info

class ZWrapper(Wrapper):
    # add z key to data dict
    def __init__(self, env, z_dim, z_discrete, z_range=None, shared_z=True, *args, **kwargs):
        super().__init__(env)
        # all possible players shouold be in env.players
        self.players = deepcopy(env.players)
        self.shared_z = shared_z
        self.z_dim = z_dim
        self.z_discrete = z_discrete
        if not z_discrete:
            assert z_range is not None, f'z_range must be defined for continuous z'
            self.z_range = z_range
        self.persistent = False

    def _resample_z(self):
        self.z = dict()
        if isinstance(self.z_dim, int):
            if self.z_discrete:
                z = random.choices(np.eye(self.z_dim, dtype=np.float32), k=len(self.players))
            else:
                z = np.random.uniform(*self.z_range, size=(len(self.players),self.z_dim)).astype(np.float32)
            if self.shared_z:
                # use only one value of z
                z = np.tile(z[0], (len(z),1))

            for i,p in enumerate(self.players):
                self.z[p] = z[i]

        elif isinstance(self.z_dim, dict):
            for p,z_dim in self.z_dim.items():
                if self.z_discrete:
                    self.z[p] = random.choice(np.eye(z_dim, dtype=np.float32))
                else:
                    self.z[p] = np.random.uniform(*self.z_range, size=(1,self.z_dim)).astype(np.float32)[0]

    def _add_z(self, data):
        for p in data:
            data[p]['z'] = self.z[p]
        return data

    def set_z_dim(self, player, size):
        if not isinstance(self.z_dim, dict):
            self.z_dim = dict()    
        self.z_dim[player] = size

    def set_z(self, z):
        self.z = dict()
        if isinstance(z, dict):
            self.z = z
        elif isinstance(z,  (list, tuple)):
            assert len(z)==len(self.players)
            for i, p in enumerate(self.players):
                self.z[p] = torch.as_tensor(z[i])
        else:
            z = torch.as_tensor(z)
            for p in self.players:
                self.z[p] = z
        self.persistent = True

    def unset_z(self):
        self.persistent = False

    def reset(self):
        if not self.persistent:
            self._resample_z()
        data = self.env.reset()
        data = self._add_z(data)
        return data

    def step(self, decision):
        data, info = self.env.step(decision)
        data = self._add_z(data)
        return data, info

class StateWrapper(Wrapper):
    '''
    Duplicate obs field as global state
    '''

    def _add_state(self, data):
        for p in data:
            data[p]['state'] = data[p]['obs']
        return data

    def get_state_shape(self, *args, **kwargs):
        obs_space = self.env.get_observation_space(*args, **kwargs)
        obs = obs_space['obs']
        return obs.n if isinstance(obs, spaces.Discrete) else obs.shape

    def reset(self):
        data = self.env.reset()
        return self._add_state(data)

    def step(self, decision):
        data, info = self.env.step(decision)
        return self._add_state(data), info


class AgentIDWrapper(Wrapper):
    def __init__(self, env, *args, **kwargs):
        assert np.array([len(v['obs'].shape)==1 for k,v in env.observation_spaces.items()]).all()
        self.env = env
        self.name_id_map = dict()
        self.num_players = len(self.env.players)
        for p,v in self.env.observation_spaces.items():
            v['obs'].shape = (v['obs'].shape[0] + self.num_players, )

    def get_observation_space(self):
        obs_space = self.env.get_observation_space()
        obs_space['obs'].shape = (obs_space['obs'].shape[0] + self.num_players, )
        return obs_space

    def _add_id(self, data):
        for p in data.keys():
            if p not in self.name_id_map:
                self.name_id_map[p] = np.eye(self.num_players)[int(p.split('_')[-1])]
            data[p]['obs'] = np.concatenate([data[p]['obs'],self.name_id_map[p]],axis=-1)
        return data

    def reset(self):
        data = self.env.reset()
        return self._add_id(data)

    def step(self, decision):
        data, info = self.env.step(decision)
        return self._add_id(data), info

class RescaleActionsWrapper(Wrapper):
    def __init__(self, env, min_action, max_action, *args, **kwargs):
        """
        min_action: new low values
        max_action: new high values
        This wrapper first clips the action to [min_action, max_action] interval then rescale to the environment's range.
        """
        for p in env.players:
            assert isinstance(
                env.action_spaces[p], spaces.Box
            ), f"expected Box action space, got {type(env.action_space)}"

        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        super().__init__(env)
        self.min_action = {p:np.zeros(env.action_spaces[p].shape, dtype=env.action_spaces[p].dtype) + min_action for p in env.players}
        self.max_action = {p:np.zeros(env.action_spaces[p].shape, dtype=env.action_spaces[p].dtype) + max_action for p in env.players}

    def step(self, decision):
        out = deepcopy(decision)
        for p in out:
            action = out[p]['action']
            action = np.clip(action, self.min_action[p], self.max_action[p])
            low = self.env.action_spaces[p].low
            high = self.env.action_spaces[p].high
            action = low + (high - low) * (
                (action - self.min_action[p]) / (self.max_action[p] - self.min_action[p])
            )
            action = np.clip(action, low, high)
            out[p]['action'] = action
        return self.env.step(out)


class ClipActionsWrapper(Wrapper):
    def step(self, decision):
        out = deepcopy(decision)
        for p in out:
            action = out[p]['action']
            out[p]['action'] = np.clip(action, self.action_spaces[p].low, self.action_spaces[p].high)
        return self.env.step(out)

# taken from https://github.com/vwxyzjn/cleanrl/blob/c43554a36e7dcdcdb994a8e75134b459fb426cf0/cleanrl/brax/ppo_continuous_action.py
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0 

    def update(self, x):
        batch_mean = np.mean([x], axis=0, dtype=np.float64)
        batch_var = np.var([x], axis=0, dtype=np.float64)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def __str__(self):
        return f'Mean: {self.mean}, Var: {self.var}, count: {self.count}, shape: {self.mean.shape}'
        
    def __repr__(self):
        return f'Mean: {self.mean}, Var: {self.var}, count: {self.count}, shape: {self.mean.shape}'

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count

class NormalizedWrapper(Wrapper):
    def __init__(self, env, norm_obs=True, norm_ret=True, norm_state=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, *args, **kwargs):
        self.env = env
        self.ob_rms = {p:RunningMeanStd(shape=self.get_observation_space().obs.shape) for p in env.players} if norm_obs else None
        self.state_rms = None
        if norm_state:
            self.state_rms = {p:RunningMeanStd(shape=self.get_state_shape()) for p in env.players}
        self.ret_rms = {p:RunningMeanStd(shape=(1,)) for p in env.players} if norm_ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = {p:np.zeros(()) for p in env.players}
        self.gamma = gamma
        self.epsilon = epsilon

    def get_stats(self):
        out = {'ob_rms':deepcopy(self.ob_rms),
                'ret_rms':deepcopy(self.ret_rms)}
        if self.state_rms is not None:
            out['state_rms'] = deepcopy(self.state_rms)
        return out

    def set_stats(self, *args, ob_rms, ret_rms, **kwargs):
        self.ob_rms = ob_rms
        self.ret_rms = ret_rms
        if 'state_rms' in kwargs:
            self.state_rms = kwargs['state_rms']

    def step(self, decision):
        data, info = self.env.step(decision)
        for p in data:
            if p in self.ob_rms:
                data[p]['obs_unnorm'] = data[p]['obs']
                data[p]['obs'] = self._obfilt(data[p]['obs'], p)
            if p in self.ret_rms:
                data[p]['reward_unnorm'] = data[p]['reward']
                data[p]['reward'] = self._rewfilt(data[p]['reward'], p)
            if 'state' in data[p]:
                if p in self.state_rms:
                    data[p]['state'] = self._statefilt(data[p]['state'], p)
        return data, info

    def _rewfilt(self, rew, p):
        if self.state_rms and (p in self.state_rms):
            self.ret[p] = self.ret[p] * self.gamma + rew
            self.ret_rms[p].update(np.array([self.ret[p]].copy()))
            rew = np.clip(rew / np.sqrt(self.ret_rms[p].var + self.epsilon), -self.cliprew, self.cliprew).astype(np.float32)
        return rew

    def _obfilt(self, obs, p):
        if self.state_rms and (p in self.state_rms):
            self.ob_rms[p].update(obs)
            obs = np.clip((obs - self.ob_rms[p].mean) / np.sqrt(self.ob_rms[p].var + self.epsilon), -self.clipob, self.clipob).astype(np.float32)
        return obs

    def _statefilt(self, state, p):
        if self.state_rms and (p in self.state_rms):
            self.state_rms[p].update(state)
            state = np.clip((state - self.state_rms[p].mean) / np.sqrt(self.state_rms[p].var + self.epsilon), -self.clipob, self.clipob).astype(np.float32)
        return state

    def reset(self):
        self.ret = {p:np.zeros(()) for p in self.env.players}
        data = self.env.reset()
        for p in data:
            if p in self.ob_rms:
                data[p]['obs_unnorm'] = data[p]['obs']
                data[p]['obs'] = self._obfilt(data[p]['obs'], p)
            if p in self.ret_rms:
                data[p]['reward_unnorm'] = data[p]['reward']
                data[p]['reward'] = self._rewfilt(data[p]['reward'], p)
            if 'state' in data[p]:
                if p in self.state_rms:
                    data[p]['state'] = self._statefilt(data[p]['state'], p)
        return data
