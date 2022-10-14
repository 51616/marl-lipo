from abc import ABC, abstractmethod

from coop_marl.agents import registered_agents
from coop_marl.envs import registered_envs
from coop_marl.envs.wrappers import wrap
from coop_marl.controllers import MappingController
from coop_marl.runners import registered_runners, EpisodesRunner, StepsRunner
from coop_marl.utils import get_logger, Dotdict, get_traj_info, auto_assign

logger = get_logger()

def population_based_setup(trainer, conf):
    agent_cls = registered_agents[conf.agent_name]
    trainer.agent_pop = [agent_cls(conf) for _ in range(conf.pop_size)]
    trainer.agents_dict = {p:agent_cls(conf) for p in trainer.env.players}
    trainer.controller = MappingController(action_spaces=trainer.env.action_spaces,
                               agents_dict=trainer.agents_dict,
                               policy_mapping_fn=lambda name: name,
                               )


def trainer_setup(trainer, conf, env_conf, eval_env_conf=None):
    trainer.iter = 0
    reg_env_name = env_conf.name
    del env_conf['name']
    env = registered_envs[reg_env_name](**env_conf)
    if eval_env_conf is not None:
        eval_env = registered_envs[reg_env_name](**eval_env_conf)
    else:
        eval_env = registered_envs[reg_env_name](**env_conf)

    if 'env_wrappers' in conf:
        env = wrap(env=env, wrappers=conf.env_wrappers, **conf)
        eval_env = wrap(env=eval_env, wrappers=conf.env_wrappers, **conf)

    conf['obs_space'] = env.get_observation_space()['obs'] 
    conf['act_space'] = env.get_action_space() 
    conf['n_agents'] = len(env.players)
    try:
        conf['state_shape'] = env.get_state_shape()
    except AttributeError:
        conf['state_shape'] = None

    trainer.env = env
    trainer.eval_env = eval_env
    trainer.agent = registered_agents[conf.agent_name](conf)


def population_evaluation(trainer, render=False, render_mode='rgb_array'):
    pop_size = trainer.conf.pop_size
    trajs = [[None]* pop_size for _ in range(pop_size)]
    infos = [[None]* pop_size for _ in range(pop_size)]
    frames = [[None]* pop_size for _ in range(pop_size)]
    metrics = [[None]* pop_size for _ in range(pop_size)]

    for i in range(pop_size):
        param_i = trainer.agent_pop[i].get_param()
        for a in trainer.controller.agents_dict.values():
            a.set_param(param_i)
        trajs[i][i], infos[i][i], frames[i][i], metrics[i][i] = evaluate(trainer, trainer.conf.n_eval_ep, render=render, render_mode=render_mode)

        for j in range(pop_size):
            if i!=j:
                param_j = trainer.agent_pop[j].get_param()
                for k in range(trainer.conf.n_agents):
                    for a in trainer.controller.agents_dict.values():
                        a.set_param(param_j)
                    player_k = list(trainer.controller.agents_dict.values())[k]
                    player_k.set_param(param_i)
                    trajs[i][j], infos[i][j], frames[i][j], metrics[i][j] = evaluate(trainer, trainer.conf.n_eval_ep, render=render, render_mode=render_mode)

    return trajs, infos, frames, metrics

def evaluate(trainer, n=None, render=False, render_mode='rgb_array'):
    eval_traj, infos, frames, eval_metrics = trainer._collect_data(n, eval=True, render=render, render_mode=render_mode)
    return eval_traj, infos, frames, eval_metrics

def collect_data(runner, n=None, render=False, render_mode='rgb_array'):
    traj, infos, frames = runner.rollout(n, render=render, render_mode=render_mode)
    metrics = Dotdict({})
    traj_info = get_traj_info(traj)
    metrics.update(traj_info)
    return traj, infos, frames, metrics


class Trainer(ABC):
    @auto_assign
    def __init__(self, conf, env_conf, eval_env_conf=None):
        self._setup(conf, env_conf, eval_env_conf)

    def _setup(self, conf, env_conf, eval_env_conf):
        self.iter = 0
        reg_env_name = env_conf.name
        del env_conf['name']
        env = registered_envs[reg_env_name](**env_conf)
        if eval_env_conf is not None:
            eval_env = registered_envs[reg_env_name](**eval_env_conf)
        else:
            eval_env = registered_envs[reg_env_name](**env_conf)

        if 'env_wrappers' in conf:
            env = wrap(env=env, wrappers=conf.env_wrappers, **conf)
            eval_env = wrap(env=eval_env, wrappers=conf.env_wrappers, **conf)

        conf['obs_space'] = env.get_observation_space()['obs'] 
        conf['act_space'] = env.get_action_space() 
        conf['n_agents'] = len(env.players)
        try:
            conf['state_shape'] = env.get_state_shape()
        except AttributeError:
            conf['state_shape'] = None

        self.env = env
        self.eval_env = eval_env

    def _runner_setup(self):
        runner_cls = registered_runners[self.conf.runner]
        self.runner = runner_cls(env=self.env, controller=self.controller)
        self.eval_runner = EpisodesRunner(env=self.env, controller=self.controller)

    def _collect_data(self, n=None, eval=False, render=False, render_mode='rgb_array'):
        runner = self.runner
        if eval:
            runner = self.eval_runner    
        if n is None:
            if isinstance(runner, EpisodesRunner):
                n = self.conf.n_ep
            elif isinstance(runner, StepsRunner):
                n = self.conf.n_ts
        return collect_data(runner, n, render, render_mode)

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, n=None, render=False, render_mode='rgb_array'):
        eval_traj, infos, frames, eval_metrics = self._collect_data(n, eval=True, render=render, render_mode=render_mode)
        return eval_traj, infos, frames, eval_metrics
