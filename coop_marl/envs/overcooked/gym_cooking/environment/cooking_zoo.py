# Other core modules
import copy

from gym_cooking.cooking_world.cooking_world import CookingWorld
from gym_cooking.cooking_world.world_objects import *
from gym_cooking.cooking_world.abstract_classes import *
from gym_cooking.cooking_book.recipe_drawer import RECIPES, NUM_GOALS

import numpy as np
from collections import namedtuple, defaultdict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

import gym


CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")
COLORS = ['blue', 'magenta', 'yellow', 'green']


def env(*, level, num_agents, record, max_steps, recipes, obs_spaces,
        interact_reward, progress_reward, complete_reward,
        step_cost):
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env_init = CookingEnvironment(level, num_agents, record, max_steps,
                                  recipes, obs_spaces,
                                  interact_reward, progress_reward, complete_reward,
                                  step_cost)
    env_init = wrappers.CaptureStdoutWrapper(env_init)
    env_init = wrappers.AssertOutOfBoundsWrapper(env_init)
    env_init = wrappers.OrderEnforcingWrapper(env_init)
    return env_init


parallel_env = parallel_wrapper_fn(env)


class CookingEnvironment(AECEnv):
    """Environment object for Overcooked."""

    metadata = {'render.modes': ['human'], 'name': "cooking_zoo"}

    def __init__(self, level, num_agents, record, max_steps, recipes, obs_spaces=["numeric"],
                 interact_reward=0.5, progress_reward=1.0, complete_reward=10.0,
                 step_cost=0.1):
        super().__init__()

        self.allowed_obs_spaces = ["symbolic", "numeric", "dense", "2d", "2d_flatten"]
        assert len(obs_spaces)==1
        assert len(set(obs_spaces + self.allowed_obs_spaces)) == len(self.allowed_obs_spaces), \
            f"Selected invalid obs spaces. Allowed {self.allowed_obs_spaces}"
        assert len(obs_spaces) != 0, f"Please select an observation space from: {self.allowed_obs_spaces}"
        self.obs_spaces = obs_spaces
        # self.allowed_objects = allowed_objects or []
        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]

        self.level = level
        self.record = record
        self.max_steps = max_steps
        self.interact_reward = interact_reward
        self.progress_reward = progress_reward
        self.complete_reward = complete_reward
        self.step_cost = step_cost
        print(f'Interact reward: {self.interact_reward}')
        print(f'Progress reward: {self.progress_reward}')
        print(f'Complete reward: {self.complete_reward}')
        print(f'Step cost: {self.step_cost}')
        self.t = 0
        self.filename = ""
        self.set_filename()
        self.world = CookingWorld()
        self.recipes = recipes
        self.game = None
        self.recipe_graphs = [RECIPES[recipe]() for recipe in recipes]
        # sort the recipe so that the longest recipe will be checked for completion first
        self.recipe_graphs = sorted(self.recipe_graphs, key=lambda x: -len(x.node_list))
        self.objects_in_recipes = set()
        for recipe in self.recipe_graphs:
            for node in recipe.node_list:
                self.objects_in_recipes.add(node.root_type.__name__)
        print(f'Objects in recipes: {self.objects_in_recipes}')

        self.termination_info = ""
        self.world.load_level(level=self.level, num_agents=num_agents)
        self.init_world_objs = copy.deepcopy(self.world.world_objects)
        self.graph_representation_length = sum([tup[1] for tup in GAME_CLASSES_STATE_LENGTH]) + self.num_agents
        self.has_reset = True

        self.recipe_mapping = dict(zip(self.possible_agents, self.recipe_graphs))
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.world_agent_mapping = dict(zip(self.possible_agents, self.world.agents))
        self.world_agent_to_env_agent_mapping = dict(zip(self.world.agents, self.possible_agents))
        self.agent_selection = None
        self._agent_selector = agent_selector(self.agents)
        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.current_tensor_observation = dict(zip(self.agents, [np.zeros((self.world.width, self.world.height,
                                                                           self.graph_representation_length))
                                                                 for _ in self.agents]))
        if "dense" in self.obs_spaces:
            self.obs_size = self.get_obs_size()
            self.observation_spaces = {agent: gym.spaces.Box(low=-1, high=1, shape=self.obs_size) for agent in self.possible_agents}
        elif "2d_flatten" in self.obs_spaces:
            self.obs_size = (self.world.width * self.world.height * (self.graph_representation_length), )
            self.observation_spaces = {agent: gym.spaces.Box(low=0, high=10, shape=self.obs_size) for agent in self.possible_agents}
        elif "2d" in self.obs_spaces:
            self.obs_size = (self.graph_representation_length, self.world.width, self.world.height)
            self.observation_spaces = {agent: gym.spaces.Box(low=0, high=10, shape=self.obs_size)
                                                                            for agent in self.possible_agents}
        
        self.action_spaces = {agent: gym.spaces.Discrete(6) for agent in self.possible_agents}
        self.held_obj = []

    def get_obs_size(self):
        all_objs = copy.deepcopy(self.world.world_objects)
        
        n_obj = 0
        n_state = 0
        for obj_name, obj_list in all_objs.items():
            for obj in obj_list:
                if (not isinstance(obj, (Floor, Counter))):
                    n_obj += 1
                    if isinstance(obj, Food):
                        n_state += 1        
        return (6 * len(self.possible_agents) + 2 * n_obj + \
                 n_state + 4 + len(GAME_CLASSES_HOLDABLE_IDX) + len(GAME_CLASSES) - 1 + \
                 len(FOOD_CLASSES) + len(FOOD_CLASSES), )

    def set_filename(self):
        self.filename = f"{self.level}_agents{self.num_agents}"

    def state(self):
        pass

    def reset(self):
        self.world = CookingWorld()
        self.t = 0
        self.held_obj = []

        # For tracking data during an episode.
        self.termination_info = ""

        # Load world & distances.
        self.world.load_level(level=self.level, num_agents=self.num_agents)
        for obj1,obj2 in zip(self.init_world_objs, self.world.world_objects):
            assert type(obj1)==type(obj2)

        for recipe in self.recipe_graphs:
            recipe.update_recipe_state(self.world)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.recipe_mapping = dict(zip(self.possible_agents, self.recipe_graphs))
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.world_agent_mapping = dict(zip(self.possible_agents, self.world.agents))
        self.world_agent_to_env_agent_mapping = dict(zip(self.world.agents, self.possible_agents))

        self.current_tensor_observation = dict(zip(self.agents, [np.zeros((self.world.width, self.world.height,
                                                                           self.graph_representation_length))
                                                                 for _ in self.agents]))
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []

    def close(self):
        return

    def step(self, action):
        agent = self.agent_selection
        self.accumulated_actions.append(action)
        for idx, agent in enumerate(self.agents):
            self.rewards[agent] = 0
        if self._agent_selector.is_last():
            self.accumulated_step(self.accumulated_actions)
            self.accumulated_actions = []
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0

    def accumulated_step(self, actions):
        # Track internal environment info.
        self.t += 1
        self.world.perform_agent_actions(self.world.agents, actions)

        # Visualize.
        if self.record:
            self.game.on_render()

        if self.record:
            self.game.save_image_obs(self.t)

        for agent in self.agents:
            self.current_tensor_observation[agent] = self.get_tensor_representation(agent)

        done, rewards, goals = self.compute_rewards()
        info = {"t": self.t, "termination_info": self.termination_info}
        for idx, agent in enumerate(self.agents):
            self.dones[agent] = done
            self.rewards[agent] = max(rewards) - self.step_cost # use the highest progressed recipe as a reward
            self.infos[agent] = info

    def observe(self, agent):
        observation = []
        if "numeric" in self.obs_spaces:
            num_observation = {'numeric_observation': self.current_tensor_observation[agent],
                               'agent_location': np.asarray(self.world_agent_mapping[agent].location, np.int32),
                               'goal_vector': self.recipe_mapping[agent].goals_completed(NUM_GOALS)}
            observation.append(num_observation)
        if "symbolic" in self.obs_spaces:
            objects = defaultdict(list)
            objects.update(self.world.world_objects)
            objects["Agent"] = self.world.agents
            sym_observation = copy.deepcopy(objects)
            observation.append(sym_observation)
        if "2d" in self.obs_spaces:
            obs = self.current_tensor_observation[agent].transpose(2,1,0)
            observation.append(obs)
        if "2d_flatten" in self.obs_spaces:
            obs = self.current_tensor_observation[agent]
            obs = obs.reshape(-1)
            observation.append(obs)

        ### Add numeric representation for symbolic data
        if "dense" in self.obs_spaces:
            # pass
            # all objects [Counter, Floor, *static_objs, *dynamic_objs, *agents]
            dense_obs = []
            world_shape = np.array([self.world.width, self.world.height], np.float32)
            all_objs = self.world.world_objects # self.all_objs
            
            state_feat = []
            next_to_counter_feat = np.zeros(4)
            pos_feat = []
            dir_feat = []
            agent_obj = self.world_agent_mapping[agent]
            own_pos = np.asarray(agent_obj.location)
            eye = np.eye(4)
            dir_feat.append(eye[agent_obj.orientation-1])
            pos_feat.append(own_pos)
            # other agents' pos
            for a in self.possible_agents:
                if a != agent:
                    other_pos = np.array(self.world_agent_mapping[a].location)
                    rel_pos = other_pos - own_pos
                    pos_feat.append(rel_pos)
                    dir_feat.append(eye[self.world_agent_mapping[a].orientation-1])
            dir_feat = np.array(dir_feat).reshape(-1)
            # static obj pos
            for obj_name, obj_list in all_objs.items():
                for obj in obj_list:
                    if (not isinstance(obj, Floor)):
                        if not isinstance(obj, Counter):
                            rel_pos = np.array(obj.location) - own_pos
                            pos_feat.append(rel_pos)
                        if isinstance(obj, Food):
                            state_feat.append(obj.done())
                        elif isinstance(obj, StaticObject):
                            if all(np.array(obj.location) == (own_pos + np.array([1,0]))):
                                next_to_counter_feat[0]=1
                            elif all(np.array(obj.location) == (own_pos + np.array([0,1]))):
                                next_to_counter_feat[1]=1
                            elif all(np.array(obj.location) == (own_pos + np.array([-1,0]))):
                                next_to_counter_feat[2]=1
                            elif all(np.array(obj.location) == (own_pos + np.array([0,-1]))):
                                next_to_counter_feat[3]=1

                            
            pos_feat = np.array(pos_feat, np.float32) / world_shape
            pos_feat = pos_feat.reshape(-1) 
            # what is in the held obj
            holding_feat = np.zeros(len(GAME_CLASSES_HOLDABLE_IDX))
            holding_state_feat = np.zeros(len(FOOD_CLASSES))
            held_obj = agent_obj.holding
            if held_obj is not None:
                held_obj_name = type(held_obj).__name__
                holding_feat[GAME_CLASSES_HOLDABLE_IDX[held_obj_name]] = 1
                if isinstance(held_obj, Food):
                    holding_state_feat[FOOD_CLASSES_IDX[held_obj_name]] = held_obj.done()
                if isinstance(held_obj, Container):
                    if held_obj.content:
                        for obj in held_obj.content:
                            obj_name = type(obj).__name__
                            holding_feat[GAME_CLASSES_HOLDABLE_IDX[obj_name]] = 1
                            if isinstance(obj, Food):
                                holding_state_feat[FOOD_CLASSES_IDX[obj_name]] = obj.done()

            # the cell in-front is interactable/pickable
            front_obj_feat = np.zeros(len(GAME_CLASSES)-1)
            front_pos = self.world.get_target_location(agent_obj, agent_obj.orientation)
            front_obj_list = self.world.get_objects_at(front_pos) 
            front_obj_state_feat = np.zeros(len(FOOD_CLASSES))
            for obj in front_obj_list:
                if not isinstance(obj, Floor):
                    obj_name = type(obj).__name__
                    front_obj_feat[OBJ_IDX[obj_name]] = 1
                    if isinstance(obj, Food):
                        front_obj_state_feat[FOOD_CLASSES_IDX[obj_name]] = obj.done()

            # Agents
            dense_obs = np.concatenate([pos_feat, dir_feat, state_feat,
                                       next_to_counter_feat,
                                       holding_feat, holding_state_feat,
                                       front_obj_feat, front_obj_state_feat])
            observation.append(dense_obs)
            
        returned_observation = observation if not len(observation) == 1 else observation[0]
        return returned_observation

    def compute_rewards(self):
        done = False
        rewards = np.zeros(len(self.recipes), dtype=np.float32) # [0] * len(self.recipes)
        open_goals = [[0]] * len(self.recipes)
        # Done if the episode maxes out
        if self.t >= self.max_steps and self.max_steps:
            self.termination_info = f"Terminating because passed {self.max_steps} timesteps"
            done = True

        for i,agent in enumerate(self.world.agents):
            if agent.holding is not None:
                state = [None]
                holding_obj = agent.holding
                if type(holding_obj).__name__ in self.objects_in_recipes:
                    if isinstance(agent.holding , Food):
                        state = [agent.holding.done()]
                    elif isinstance(agent.holding, Container):
                        if agent.holding.content:
                            food_names = [type(food).__name__ for food in agent.holding.content]
                            state = food_names

                    obj_with_state = set({agent, holding_obj, *state})
                    if obj_with_state not in self.held_obj:
                        self.held_obj.append(obj_with_state)
                        rewards += self.interact_reward

        for idx, recipe in enumerate(self.recipe_graphs):
            goals_before = recipe.goals_completed(NUM_GOALS)
            recipe.update_recipe_state(self.world)
            open_goals[idx] = recipe.goals_completed(NUM_GOALS)
            bonus = recipe.completed() * self.complete_reward
            rewards[idx] += bonus if bonus else ((sum(goals_before) - sum(open_goals[idx])) * self.progress_reward) # * 5
            if rewards[idx] < 0:
                print(f"Goals before: {goals_before}")
                print(f"Goals after: {open_goals}")
        
        for recipe in self.recipe_graphs:
            if recipe.completed():
                self.termination_info = f"Terminating because {recipe.name} is completed"
                done = True
                break
        return done, rewards, open_goals

    def get_tensor_representation(self, agent):
        tensor = np.zeros(
            (self.world.width, self.world.height, self.graph_representation_length))
        objects = defaultdict(list)
        objects.update(self.world.world_objects)
        idx = 0
        for game_class in GAME_CLASSES:
            if game_class is Agent:
                continue
            stateful_class = self.get_stateful_class(game_class)
            if stateful_class:
                n = 1
                for obj in objects[ClassToString[game_class]]:
                    representation = self.handle_stateful_class_representation(obj, stateful_class)
                    n = len(representation)
                    x, y = obj.location
                    for i in range(n):
                        tensor[x, y, idx + i] += representation[i]
                idx += n
            else:
                for obj in objects[ClassToString[game_class]]:
                    x, y = obj.location
                    tensor[x, y, idx] += 1
                idx += 1

        ego_agent = self.world_agent_mapping[agent]
        x, y = ego_agent.location
        # location map for all agents, location maps for separate agent and four orientation maps shared
        # between all agents
        tensor[x, y, idx] = 1
        tensor[x, y, idx + 1] = 1
        tensor[x, y, idx + self.num_agents + ego_agent.orientation] = 1

        agent_idx = 1
        for world_agent in self.world.agents:
            if agent != world_agent:
                x, y = world_agent.location
                # location map for all agents, location maps for separate agent and four orientation maps shared
                # between all agents
                tensor[x, y, idx] = 1
                tensor[x, y, idx + agent_idx + 1] = 1
                tensor[x, y, idx + self.num_agents + world_agent.orientation] = 1
                agent_idx += 1
        return tensor

    def get_agent_names(self):
        return [agent.name for agent in self.world.agents]

    def render(self, mode='human'):
        pass

    @staticmethod
    def get_stateful_class(game_class):
        for stateful_class in STATEFUL_GAME_CLASSES:
            if issubclass(game_class, stateful_class):
                return stateful_class
        return None

    @staticmethod
    def handle_stateful_class_representation(obj, stateful_class):
        if stateful_class is ChopFood:
            return [int(obj.chop_state == ChopFoodStates.CHOPPED)]
        if stateful_class is BlenderFood:
            return [obj.current_progress]
        raise ValueError(f"Could not process stateful class {stateful_class}")
