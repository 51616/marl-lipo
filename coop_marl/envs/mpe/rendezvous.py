import numpy as np
from scipy.spatial.distance import cdist
from gym.spaces import Discrete, Box
from pettingzoo.mpe._mpe_utils.simple_env import make_env # SimpleEnv, 
from pettingzoo.mpe._mpe_utils.core import World, Agent, Landmark
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.utils.conversions import parallel_wrapper_fn

from coop_marl.envs.mpe._mpe_utils.simple_env import SimpleEnv
from coop_marl.envs.wrappers import SARDConsistencyChecker
from coop_marl.utils import Arrdict, Dotdict

def equid_points_circle(n,r, rot=True):
    if n <= 0:
        return np.array([])
    pi = np.pi
    thetas = [i*(2*pi/n) for i in range(n)]
    points = np.stack([r*np.cos(thetas), r*np.sin(thetas)],axis=-1)
    # rotate every point by pi/n rad
    if rot:
        rot_angle = pi/n
        rot = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
                          [-np.sin(rot_angle), np.cos(rot_angle)]])
        points = rot @ points.T
        points = points.T
    return points

class Scenario(BaseScenario):
    def make_world(self, n_landmarks, n_agents, partner_obs, mode):
        assert mode in ['easy','medium','hard']
        self.n_landmarks = n_landmarks
        self.partner_obs = partner_obs
        self.n_agents = n_agents
        self.mode = mode
        self.lm_size = 0.2
        
        world = World()
        # add agents
        world.agents = [Agent() for i in range(n_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_{}'.format(i)
            agent.collide = False
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(n_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = self.lm_size
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([244, 177, 131])/255
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([161, 184, 225])/255
        # set random initial states
        start_pos = {'easy':np.array([[0.3,0.0],[-0.3,0]]),
                     'hard':np.array([[1.0,0.0],[-1.0,0.0]])
                     }

        for i,agent in enumerate(world.agents):
            if self.n_agents > 1:
                agent.state.p_pos = start_pos[self.mode][i]
            else:
                agent.state.p_pos = np.zeros(world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        space = 1.5
        hard_lms = np.array([[0,1.5*space],[0,0.5*space],[0,-0.5*space],[0,-1.5*space]])
        lm_pos = {
                  'easy': equid_points_circle(n=self.n_landmarks, r=2.25),
                  'hard': hard_lms,
                  }
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = lm_pos[self.mode][i]
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        reward = 0.0
        centroid = np.array([a.state.p_pos for a in world.agents]).mean(axis=0)
        reward += -np.linalg.norm(agent.state.p_pos - centroid)

        """
        Dense reward
        """
        lm_pos = np.array([lm.state.p_pos for lm in world.landmarks])
        lm_a_dist = cdist(lm_pos, centroid[np.newaxis,:]) # [n_landmarks, 1]
        dist = np.min(lm_a_dist,axis=0) # get the closest landmark
        reward += 1.0 - dist
        return reward

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        if self.partner_obs:
            for a in world.agents:
                if a is not agent:
                    entity_pos.append(a.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel, agent.state.p_pos] + entity_pos)

class raw_env(SimpleEnv):
    def __init__(self, max_cycles, n_landmarks, partner_obs, mode, continuous_actions=False):
        scenario = Scenario()
        n_agents = 2
        world = scenario.make_world(n_landmarks, n_agents, partner_obs, mode)
        super().__init__(scenario, world, max_cycles)
        self.metadata['name'] = "rendezvous"

    def get_closest_landmark(self, agent_name):
        agent_idx = self._index_map[agent_name]
        agent = self.world.agents[agent_idx]
        dist = [np.linalg.norm(agent.state.p_pos - lm.state.p_pos) for lm in self.world.landmarks]
        return np.argmin(dist)

    def get_dist_from_landmarks(self):
        out = np.zeros(len(self.world.landmarks))
        for i, lm in enumerate(self.world.landmarks):
            dist = sum([np.linalg.norm(a.state.p_pos - lm.state.p_pos) for a in self.world.agents])
            out[i] = dist
        return out

env_maker = make_env(raw_env)
parallel_env_maker = parallel_wrapper_fn(env_maker)

class Rendezvous:
    def __init__(self,horizon=25, n_landmarks=4, partner_obs=True, mode='easy', *args,**kwargs):
        self._env = parallel_env_maker(max_cycles=horizon,
                                       n_landmarks=n_landmarks,
                                       partner_obs=partner_obs,
                                       mode=mode)
        self.n_landmarks = n_landmarks
        self.n_agents = 2
        self.partner_obs = partner_obs
        self.players = self._env.possible_agents
        self.action_spaces = Dotdict({agent: self.get_action_space() for agent in self.players})
        self.observation_spaces = Dotdict({agent: self.get_observation_space() for agent in self.players})

    def get_action_space(self):
        return Discrete(5)

    def get_observation_space(self):
        # agent observation size
        # only use keys that will be used to forward pass in the agents' act method
        if self.partner_obs:
            return Dotdict(obs=Box(-np.inf,np.inf,shape=[4+2*self.n_landmarks+2*(self.n_agents-1)]))
        else:
            return Dotdict(obs=Box(-np.inf,np.inf,shape=[4+2*self.n_landmarks]))

    def reset(self,switch=False,soft=False):
        obs = self._env.reset()
        data = Arrdict()
        for p,k in zip(self.players,obs):
            data[p] = Arrdict(obs=obs[k],
                             reward=0.,
                             done=False
                             ) 
        return data

    def step(self,decision):
        actions = {}
        for a,p in zip(self._env.agents,decision.action):
            actions[a] = decision.action[p]

        obs, reward, done, info = self._env.step(actions)
        info = Dotdict()
        if sum(done.values()):
            for p in self._env.agents:
                info['dist_from_landmarks'] = self._env.unwrapped.get_dist_from_landmarks()
        data= Arrdict()
        for k in obs.keys():
            data[k] = Arrdict(obs=obs[k].astype(np.float32),
                             reward=np.float32(reward[k]), 
                             done=done[k]
                            )
        return data, Dotdict(info)

    
    def render(self, mode='human'):
        return self._env.render(mode)

    @staticmethod
    def make_env(*args, **kwargs):
        env = Rendezvous(*args, **kwargs)
        env = SARDConsistencyChecker(env)
        return env

if __name__=='__main__':
    env = Rendezvous.make_env(n_landmarks=4, mode='easy')
    data = env.reset()
    print(data.agent_0.obs)
    # x = env.render(mode='rgb_array')
    # plt.imshow(x)
    # plt.xticks([], [])
    # plt.yticks([], [])
    # plt.show()
    done = False
    while not done:
        data, infos = env.step(Dotdict({p:Dotdict(action=space.sample()) for p,space in env.action_spaces.items()}))
        done = sum(data.done.values())        
    # x = env.render(mode='rgb_array')
    # plt.imshow(x)
    # plt.show()




