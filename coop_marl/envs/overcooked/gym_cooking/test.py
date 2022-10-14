from gym_cooking.environment import cooking_zoo

n_agents = 1
num_humans = 1
max_steps = 100
render = False

level = 'open_room_salad_easy'
seed = 1
record = False
max_num_timesteps = 1000
recipes = ["LettuceSalad", 'LettuceSalad']

env = parallel_env = cooking_zoo.parallel_env(level=level, num_agents=n_agents, record=record,
                                        max_steps=max_num_timesteps, recipes=recipes, obs_spaces=["dense"],
                                        interact_reward=0.5, progress_reward=1.0, complete_reward=10.0,
                                        step_cost=0.05)
obs = env.reset()
print(obs)
