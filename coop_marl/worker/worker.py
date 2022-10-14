from functools import partial

import ray

from coop_marl.agents import registered_agents
from coop_marl.envs import registered_envs
from coop_marl.envs.wrappers import wrap
from coop_marl.controllers import MappingController
from coop_marl.runners import registered_runners, EpisodesRunner, StepsRunner
from coop_marl.utils import Dotdict, arrdict, save_gif, merge_dict
from coop_marl.utils.metrics import get_traj_info

p_save_gif = partial(save_gif, fps=30, size=(200,200))

def collect_data(runner, n=None, render=False, render_mode='rgb_array'):
    traj, infos, frames = runner.rollout(n, render=render, render_mode=render_mode)
    metrics = Dotdict({})
    traj_info = get_traj_info(traj)
    metrics.update(traj_info)
    return traj, infos, frames, metrics

def setup_xp(worker, controller, home_conf, away_conf,
             home_state, away_state, 
             home_env_stats, away_env_stats,
             env,
             home_id, away_id, k, eval_mode):

    # merge home and away keys
    keys = list(home_env_stats.keys()) + list(away_env_stats.keys())
    env_stats = dict({s:dict() for s in keys})
    for i,(p,a) in enumerate(controller.agents_dict.items()):
        conf = home_conf
        state = home_state
        e_stats = home_env_stats
        partner_id = away_id
        if i==k:
            conf = away_conf
            state = away_state
            e_stats = away_env_stats
            partner_id = home_id
        if not isinstance(a, registered_agents[conf.agent_name]):
            controller.agents_dict[p] = registered_agents[conf.agent_name](conf) # construct a new agent
        for s in env_stats:
            if s in e_stats:
                env_stats[s][p] = e_stats[s][p]
        if 'ZWrapper' in conf.env_wrappers:
            env.set_z_dim(p, conf.z_dim)
        controller.agents_dict[p].load_state(state)
        if isinstance(controller.agents_dict[p], registered_agents['IncompatMAPPOZ']):
            controller.agents_dict[p].value_net = controller.agents_dict[p].sp_critic
            if partner_id < len(controller.agents_dict[p].xp_critics):
                controller.agents_dict[p].value_net = controller.agents_dict[p].xp_critics[partner_id]
        if isinstance(controller.agents_dict[p], registered_agents['QMIXAgent']):
            if eval_mode:
                controller.agents_dict[p].explore = False
            else:
                controller.agents_dict[p].explore = True
            
    env.set_stats(**env_stats)
    worker.runner = worker.runner_cls(env=env, controller=controller) if not eval_mode \
                    else EpisodesRunner(env=env, controller=controller)


def setup_sp(worker, controller, state, conf, env_stats, env, eval_mode):
    # check a type
    for p,a in controller.agents_dict.items():
        if not isinstance(a, registered_agents[conf.agent_name]):
            controller.agents_dict[p] = registered_agents[conf.agent_name](conf)
        controller.agents_dict[p].load_state(state)
        if isinstance(controller.agents_dict[p], registered_agents['IncompatMAPPOZ']):
            controller.agents_dict[p].value_net = a.sp_critic
        if isinstance(controller.agents_dict[p], registered_agents['QMIXAgent']):
            if eval_mode:
                controller.agents_dict[p].explore = False
            else:
                controller.agents_dict[p].explore = True
    env.set_stats(**env_stats)
    worker.runner = worker.runner_cls(env=env, controller=controller) if not eval_mode \
                    else EpisodesRunner(env=env, controller=controller)

@ray.remote
class RolloutWorker:
    """
    Two-agent runner
    """
    def __init__(self, conf, env_conf, eval_env_conf):
        self.conf = conf
        self._setup(conf, env_conf, eval_env_conf)

        agent_cls = registered_agents[conf.agent_name]
        self.agents_dict = {p:agent_cls(conf) for p in self.env.players}
        self.controller = MappingController(action_spaces=self.env.action_spaces,
                                   agents_dict=self.agents_dict,
                                   policy_mapping_fn=lambda name: name,
                                   )

        self.runner_cls = registered_runners[self.conf.runner]

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


    def set_param(self, home_param: dict, away_param: dict) -> None:
        self.set_home_param(home_param)
        self.set_away_param(away_param)

    def set_home_id(self, id):
        self.home_id = id

    def set_away_id(self, id):
        self.away_id = id

    def set_sp(self, flag):
        self.sp = flag

    def update(self, home_conf, away_conf,
                           home_state, away_state,
                           home_env_stats, away_env_stats):
        self.home_conf = home_conf
        self.away_conf = away_conf
        self.home_state = home_state
        self.away_state = away_state
        self.home_env_stats = home_env_stats
        self.away_env_stats = away_env_stats


    def get_home_env_stats(self):
        return self.home_env_stats

    def get_away_env_stats(self):
        return self.away_env_stats

    def get_home_id(self):
        return self.home_id

    def get_away_id(self):
        return self.away_id

    def get_env_stats(self):
        return self.env.get_stats()

    def get_controller(self):
        return self.controller

    def set_home_param(self, param):
        self.home_param = param

    def set_away_param(self, param):
        self.away_param = param

    def set_z(self, z, eval_mode=False):
        for env in [self.env, self.eval_env]:
            env.set_z(z)

    def unset_z(self):
        for env in [self.env, self.eval_env]:
            env.unset_z()

    def _collect_data(self, runner, n=None, eval=False, render=False, render_mode='rgb_array'):
        if n is None:
            if isinstance(runner, EpisodesRunner):
                n = self.conf.n_ep
            elif isinstance(runner, StepsRunner):
                n = self.conf.n_ts
        return collect_data(runner, n, render, render_mode)

    def _sp_rollout(self, n=None, render=None, eval_mode=False):
        if render is None:
            render = self.conf.render
        env = self.env
        if eval_mode:
            env = self.eval_env
        setup_sp(self, self.controller, self.home_state, self.home_conf, self.home_env_stats, env, eval_mode)
        if n is None:
            n = self.conf.n_sp_episodes if not eval_mode else self.conf.n_eval_ep
            if not eval_mode and isinstance(self.runner, StepsRunner):
                n = self.conf.n_sp_ts
        sp_rollout, infos, frames, metrics = self._collect_data(self.runner, n, eval_mode, render * eval_mode, self.conf.render_mode)
        return Dotdict({'rollouts':sp_rollout, 'infos':infos, 'frames':frames, 'metrics':metrics})

    def _xp_rollout(self, n=None, render=None, eval_mode=False):
        home_rollout, away_rollout = None, None
        infos, frames, metrics = [[None] * self.conf.n_agents for _ in range(3)]
        if render is None:
            render = self.conf.render
        env = self.env
        if eval_mode:
            env = self.eval_env
        for k in range(self.conf.n_agents):
            setup_xp(self, self.controller, self.home_conf, self.away_conf,
                     self.home_state, self.away_state,
                     self.home_env_stats, self.away_env_stats,
                     env,
                     self.home_id, self.away_id, k, eval_mode)
            if n is None:
                n = self.conf.n_sp_episodes if not eval_mode else self.conf.n_eval_ep
                if not eval_mode and isinstance(self.runner, StepsRunner):
                    n = self.conf.n_xp_ts
            xp_rollout, infos[k], frames[k], metrics[k] = self._collect_data(self.runner, n//self.conf.n_agents,
                                          eval_mode, render * eval_mode, self.conf.render_mode)
            # infos is arbitrary dict -> create a list for non-iterable data, concat iterable data
            # frames is list of images -> can concat directly
            # metrics is arbitrary dict -> create a list for non-iterable data, concat iterable data

            agent_k_name = list(self.controller.agents_dict.keys())[k]
            if away_rollout is None:
                away_rollout = getattr(xp_rollout, agent_k_name)
            else:
                away_rollout = arrdict.cat([away_rollout, getattr(xp_rollout, agent_k_name)], axis=0)

            for p in xp_rollout.inp.data:
                if p!=agent_k_name:
                    if home_rollout is None:
                        home_rollout = getattr(xp_rollout,p)
                    else:
                        home_rollout = arrdict.cat([home_rollout, getattr(xp_rollout, p)], axis=0)
        f_out = []
        for f in frames:
            f_out.extend(f)
        i_out = []
        for i in infos:
            i_out.extend(i)
        m_out = merge_dict(metrics)
        
        return Dotdict({'rollouts':(home_rollout, away_rollout), 'infos':i_out, 'frames':f_out, 'metrics':m_out})


    def rollout(self, n, render=None, eval_mode=False, render_only_sp=False):
        if self.sp:
            out = self._sp_rollout(n, render, eval_mode)
        else:
            out = self._xp_rollout(n, render * (1 - render_only_sp), eval_mode)
        return out
