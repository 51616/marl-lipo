from copy import deepcopy
import random

from coop_marl.utils import Arrdict, arrdict

'''
All runner assume simultaneous action except RoundRobinTurnBasedEpisodesRunner
'''

class Runner():
    def __init__(self,env,controller,*,possible_teams=None):
        self.env = env
        self.controller = controller
        self.possible_teams = possible_teams
        self._setup()

    def _setup(self):
        self.total_timestep = 0

    def rollout(self, *args, **kwargs):
        # each runner should have its own rollout logic
        raise NotImplementedError

class EpisodesRunner(Runner):
    def _setup(self):
        super()._setup()
        # each player might have different keys in the decision
        # make sure that the controller takes care of this
        self.dummy_decision = self.controller.get_prev_decision_view()

    def rollout(self, num_episodes, render=False, render_mode='rgb_array', resample_players=False):
        frames = []
        buffer = []
        infos = []
        for _ in range(num_episodes):
            if resample_players:
                team = random.choice(self.possible_teams)
                self.env.set_players(team)
            self.controller.reset()
            outcome = self.env.reset()
            decision = Arrdict({p:self.dummy_decision[p] for p in outcome})
            while True:
                if render:
                    frames.append(self.env.render(mode=render_mode))
                inp = Arrdict(data=outcome, prev_decision=decision)
                decision = self.controller.select_actions(inp)
                transition = Arrdict(inp=inp, decision=decision)
                # env step
                outcome, info = self.env.step(decision)
                # use s_t, r_t, a_t, d_t convention
                transition['outcome'] = outcome
                # add transition to buffer
                buffer.append(transition)
                infos.append(info)
                self.total_timestep += 1
                # last time step data will not be collected
                # check terminal condition
                done_agents = set(k for k,v in outcome.done.items() if v)
                if done_agents==set(outcome.keys()):
                    break
        traj = arrdict.stack(buffer)
        return traj, infos, frames

class StepsRunner(Runner):
    def _setup(self):
        super()._setup()
        self.last_outcome = self.env.reset()
        # each player might have different keys in the decision
        # make sure that the controller takes care of this
        self.dummy_decision = self.controller.get_prev_decision_view()
        self.last_decision = Arrdict({p:self.dummy_decision[p] for p in self.last_outcome}) # None

    # has to keep env state between calls
    def rollout(self, num_timesteps, render=False, render_mode='rgb_array', resample_players=False):
        # keep rolling-out until num_timesteps frams are collected (through multiple resets if needed)
        frames = []
        buffer = []
        infos = []

        outcome = self.last_outcome
        decision = deepcopy(self.last_decision)
        # remove keys that're not gonna be used in prev_decision
        for p,d in self.last_decision.items():
            for k in d:
                if k not in self.dummy_decision[p]:
                    del decision[p][k]

        for _ in range(num_timesteps):
            if render:
                frames.append(self.env.render(mode=render_mode))

            inp = Arrdict(data=outcome, prev_decision=decision)
            decision = self.controller.select_actions(inp)
            transition = Arrdict(inp=inp, decision=decision)
            # env step
            outcome, info = self.env.step(decision)
            # use s_t, r_t, a_t, d_t convention
            transition['outcome'] = outcome
            # add transition to buffer
            buffer.append(transition)
            infos.append(info)
            self.total_timestep += 1
            # last time step data will not be collected
            # check terminal condition
            done_agents = set(k for k,v in outcome.done.items() if v)
            if done_agents==set(outcome.keys()):
                if resample_players:
                    team = random.choice(self.possible_teams)
                    self.env.set_players(team)
                self.controller.reset()
                outcome = self.env.reset()
                decision = Arrdict({p:self.dummy_decision[p] for p in outcome}) # None

        traj = arrdict.stack(buffer)
        self.last_outcome = outcome
        self.last_decision = decision

        return traj, infos, frames


