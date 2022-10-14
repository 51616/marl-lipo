import numpy as np
from gym.spaces import Discrete

from coop_marl.utils import Arrdict, arrdict, reverse_dict
from coop_marl.agents import Agent

class Controller:
    def select_actions(self, inp):
        raise NotImplementedError
    def train(self, *args, **kwargs):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    def get_prev_decision_view(self):
        raise NotImplementedError

# Agent-less controller, e.g., random agent could implement without calling each agent explicitly
class RandomController(Controller):
    # this factors out the agents
    def __init__(self, action_spaces):
        self.action_spaces = action_spaces

    def select_actions(self, inp):
        # data is arrdict
        action_dict = Arrdict()
        for k in inp.data.keys():
            if 'action_mask' in inp.data[k].keys() and isinstance(self.action_spaces[k],Discrete):
                # handle action mask in discrete action space
                a = np.random.choice(np.where(inp.data[k].action_mask==1)[0])
                action_dict[k] = Arrdict(action=a)
            else:
                action_dict[k] = Arrdict(action=self.action_spaces[k].sample())
        return action_dict

    def train(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def get_prev_decision_view(self):
        return {p:Arrdict() for p,space in self.action_spaces.items()}


# Agent-based controller needs to call each agent independently to construct action_dict
class PSController(Controller):
    '''
    Parameter sharing controller: one agent controlls all the players in the game
    '''
    def __init__(self, action_spaces, agent):
        assert isinstance(agent, Agent), f'agent must be an instance of Agent. Got {type(agent)}'
        self.action_spaces = action_spaces
        self.agent = agent

    def select_actions(self, inp):
        # act returns list of arrdict (each arrdict represents one player's action)
        actions = self.agent.act(inp)
        action_dict = Arrdict((k,v) for k,v in zip(inp.data.keys(), actions))
        return action_dict

    def train(self, traj, *args, **kwargs):
        if kwargs.get('flatten',True):
            # concat and preprocess all agents traj to a single batch
            batch = []
            for player in traj.inp.data:
                player_traj = getattr(traj,player)
                self.agent.preprocess(player_traj)
                batch.append(player_traj)
            batch = arrdict.cat(batch)
            self.agent.train(batch, *args, **kwargs)
        else:
            self.agent.preprocess(traj)
            self.agent.train(traj, *args, **kwargs)

    def reset(self):
        # called at the begining of an episode
        self.agent.reset()

    def get_prev_decision_view(self):
        dummy_decision = self.agent.get_prev_decision_view()
        return Arrdict({p:dummy_decision for p in self.action_spaces})
        

class MappingController(Controller):
    '''
    Maps player to agent using policy_mapping_fn
    '''
    # currently does not support stochastic mapping (training logic assumes mapping is static between episodes)
    def __init__(self, action_spaces, agents_dict, policy_mapping_fn, possible_teams=None):
        self.action_spaces = action_spaces
        self.agents_dict = agents_dict
        self.policy_mapping_fn = policy_mapping_fn
        self.possible_teams = possible_teams
        self._player_to_agent = {}

    # ray/rllib/evaluation/episode.py
    def agent_for(self, player):
        if player not in self._player_to_agent:
            self._player_to_agent[player] = self.policy_mapping_fn(player)
        return self._player_to_agent[player]

    def select_actions(self, inp):
        player_to_agent = {p:self.agent_for(p) for p in inp.data.keys()}
        agent_to_player = reverse_dict(player_to_agent)

        # create inp_dict for each agent to use as inputs
        agent_in = Arrdict()
        for agent, player_list in agent_to_player.items():
            for player in player_list:
                if agent not in agent_in:
                    agent_in[agent] = Arrdict()
                    agent_in[agent][player] = getattr(inp,player)
                else:
                    assert player not in agent_in[agent], f'each player has its own dict key\
                    and no one player should appear twice'
                    agent_in[agent][player] = getattr(inp,player)

        # act
        action_dict = Arrdict()
        for agent, player_list in agent_to_player.items():
            actions = self.agents_dict[agent].act(agent_in[agent])
            for a,player in zip(actions, player_list):
                action_dict[player] = a
        return action_dict
        
    def train(self, traj, *args, **kwargs):
        # preprocess traj
        player_to_agent = {p:self.agent_for(p) for p in traj.inp.data.keys()}
        agent_batch = Arrdict()
        for player,agent in player_to_agent.items():
            player_traj = getattr(traj,player)
            # per agent perprocess
            self.agents_dict[agent].preprocess(player_traj)
            if agent not in agent_batch:
                agent_batch[agent] = player_traj
            else:
                agent_batch[agent] = arrdict.cat([agent_batch[agent], (player_traj)])

        for agent in agent_batch.keys():
            self.agents_dict[agent].train(agent_batch[agent], *args, **kwargs)

    def reset(self):
        self._player_to_agent = {}
        for a in self.agents_dict.values():
            a.reset()

    def get_prev_decision_view(self):
        decision = Arrdict()
        if self.possible_teams is None:
            for p in self.action_spaces:
                decision[p] = self.agents_dict[self.agent_for(p)].get_prev_decision_view()
        else:
            for team in self.possible_teams:
                for p in team:
                    decision[p] = self.agents_dict[self.agent_for(p)].get_prev_decision_view()
        return decision
