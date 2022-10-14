import numpy as np
from gym.spaces import Box, Discrete

from coop_marl.utils import Arrdict, Dotdict
from coop_marl.envs.wrappers import SARDConsistencyChecker

class OneStepMatrixGame:
    def __init__(self,n_conventions,payoffs,k,*args,**kwargs):
        self.players = ['player_0','player_1']
        if isinstance(k, list):
            assert len(k)==n_conventions
        if isinstance(k, int):
            k = [k] * n_conventions
            
        self.n_actions = sum(k)
        print(f'N actions: {self.n_actions}')
        self.payoff_matrix = np.zeros([self.n_actions, self.n_actions], dtype=np.float32)
        if isinstance(payoffs, str):
            payoffs = eval(payoffs)
        for m in range(n_conventions):
            start = sum(k[:m])
            end = sum(k[:m+1])
            self.payoff_matrix[start:end, start:end] = payoffs[m]

        self.action_spaces = Dotdict({player: self.get_action_space() for player in self.players})
        self.observation_spaces = Dotdict({player: self.get_observation_space() for player in self.players})

    def get_action_space(self):
        return Discrete(self.n_actions)

    def get_observation_space(self):
        return Dotdict(obs=Box(low=0, high=1, shape=[1], dtype=np.float32))

    def set_players(self, players):
        self.players = players

    def reset(self):
        data = Arrdict()
        for p in self.players:
            data[p] = Arrdict(obs=np.array([1.0], dtype=np.float32),
                              reward=np.float32(0),
                              done=False)
        return data

    def step(self, decision):
        done = True
        joint_actions = list(decision.action.values())
        a1, a2 = joint_actions
        reward = self.payoff_matrix[a1,a2]

        data = Arrdict()
        for p in self.players:
            data[p] = Arrdict(obs=np.array([0.0], dtype=np.float32),
                              reward=np.float32(reward),
                              done=done)
        return data, {}

    def render(self, *args, **kwargs):
        return 

    @staticmethod
    def make_env(*args,**kwargs):
        env = OneStepMatrixGame(*args,**kwargs)
        env = SARDConsistencyChecker(env)
        return env

if __name__ == '__main__':
    env = OneStepMatrixGame.make_env(n_conventions=3,k=[3,2,1],payoffs='np.linspace(1,0.5,n_conventions)')
    data = env.reset()
    action_spaces = env.action_spaces

    while True:
        decision = Arrdict()
        for p in action_spaces:
            a = action_spaces[p].sample()
            decision[p] = Arrdict(action=a)
        for p in decision:
            print(f'{p} obs: {data.obs[p]}')
            print(f'{p} action: {decision[p].action}')
        data, *_= env.step(decision)
        for p in data:
            print(f'{p} reward: {data.reward[p]}')
        print()
        if data.player_0.done:
            for p in data:
                print(f'{p} obs: {data.obs[p]}')
            break
