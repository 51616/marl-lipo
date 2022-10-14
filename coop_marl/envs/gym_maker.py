import numpy as np
import gym
from copy import copy

from coop_marl.utils import Arrdict, Dotdict
from coop_marl.envs.wrappers import SARDConsistencyChecker

class GymMaker:
    def __init__(self, env_name):
        self._env = gym.make(env_name)
        self.players = ['player_0']
        self.action_spaces = Dotdict({self.players[0]:copy(self._env.action_space)})
        self.observation_spaces = Dotdict({self.players[0]:Dotdict(obs=copy(self._env.observation_space))})
        self.total_steps = 0

    def get_action_space(self):
        return self._env.action_space

    def get_observation_space(self):
        return Dotdict(obs=self._env.observation_space)

    def reset(self):
        obs = self._env.reset()
        data = Arrdict()
        data[self.players[0]] = Arrdict(obs=obs.astype(np.float32), reward=np.float32(0), done=False)
        return data

    def step(self,decision):
        self.total_steps += 1
        action = decision[self.players[0]]['action']
        obs, reward, done, info = self._env.step(action)
        data = Arrdict()
        data[self.players[0]] = Arrdict(obs=obs.astype(np.float32), reward=reward.astype(np.float32), done=done)
        return data, Dotdict(info)

    def render(self, mode):
        return self._env.render(mode)

    @staticmethod
    def make_env(*args,**kwargs):
        env = GymMaker(*args,**kwargs)
        env = SARDConsistencyChecker(env)
        return env

if __name__ == '__main__':
    from coop_marl.controllers import RandomController
    from coop_marl.runners import StepsRunner
    import argparse

    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--env_name', type=str, default='CartPole-v1',
                        help='name of the env')
    args = parser.parse_args()

    env = GymMaker(args.env_name)
    action_spaces = env.action_spaces
    controller = RandomController(action_spaces)
    runner = StepsRunner(env, controller)
    for i in range(20):
        traj, *_ = runner.rollout(1)
        print(traj.data.player_1)
