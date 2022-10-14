import pickle
import os
import gc
from collections import defaultdict
from collections.abc import Iterable
from functools import partial

import numpy as np
import torch
import ray

from coop_marl.trainers import Trainer
from coop_marl.agents import registered_agents
from coop_marl.envs import registered_envs
from coop_marl.envs.wrappers import wrap, RunningMeanStd
from coop_marl.utils import get_logger, pblock, Dotdict, arrdict, merge_dict, save_gif
from coop_marl.utils.rl import flatten_traj
from coop_marl.worker import RolloutWorker
from coop_marl.evaluation import update_workers
from coop_marl.utils.metrics import get_avg_metrics

logger = get_logger()
p_save_gif = partial(save_gif, fps=30, size=(200,200))

class SimplePSTrainer(Trainer):
    """
    Plain PS Controller + EpisodesRunner
    """
    def __init__(self, conf, env_conf, eval_env_conf=None):
        self._setup(conf, env_conf, eval_env_conf)
        self.conf = conf
        self.env_conf = env_conf
        self.eval_env_conf = eval_env_conf
        agent_cls = registered_agents[conf.agent_name]
        if not conf.checkpoint:
            self.agent_pop = [agent_cls(conf)]

        else:
            checkpoint = torch.load(f'{conf.checkpoint}/trainer_checkpoint.pt')
            self.agent_pop = checkpoint['agent_pop']
            self.iter = checkpoint['iter']

        logger.info('Initializing ray...')
        ray.init(local_mode=conf.debug, log_to_driver=conf.debug)
        
        self.n_workers = conf.n_workers
        if conf.algo_name in ['maven', 'sp_mi']:
            self.n_workers = min(conf.n_workers, conf.z_dim, conf.n_sp_episodes)
        self.workers = [RolloutWorker.remote(conf, env_conf, eval_env_conf) for _ in range(self.n_workers)]

    def _setup(self, conf, env_conf, eval_env_conf=None):
        self.iter = 0
        self.count = 0
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

        self.env_stats_list = [env.get_stats()]
        env_conf['name'] = reg_env_name

        conf['obs_space'] = env.get_observation_space()['obs']
        conf['act_space'] = env.get_action_space()
        conf['n_agents'] = len(env.players)
        try:
            conf['state_shape'] = env.get_state_shape()
        except AttributeError:
            conf['state_shape'] = None

    def _log_gif(self, frames):
        if isinstance(frames[0], np.ndarray):
            # save only one gif
            path = f'{self.conf.save_dir}/it_{self.iter:05d}/renders/0.gif'
            p_save_gif(frames, path)
        else:
            # save as multiple gifs
            fs = [frames[i] for i in range(len(frames))]
            paths = [f'{self.conf.save_dir}/it_{self.iter:05d}/renders/0-{i}.gif' for i in range(len(frames))]
            for f,p in zip(fs, paths):
                p_save_gif(f,p)
            del fs
        del frames
        gc.collect()

    def _collect(self, eval_mode=False):
        def unpack(res):
            rollouts, infos, frames, metrics = [[None] * len(res) for _ in range(4)]
            for i, r in enumerate(res):
                rollouts[i], infos[i], frames[i], metrics[i] = r['rollouts'], r['infos'], r['frames'], r['metrics']
            return rollouts, infos, frames, metrics

        def update_count(count, n):
            self.count += n

        update_workers(self.workers, [(0,0)] * self.n_workers, [{'agent_conf': self.conf}] * self.n_workers,
                        [self.agent_pop[0].get_state()] * self.n_workers, self.env_stats_list * self.n_workers)
        n = self.conf.n_sp_episodes//self.n_workers if self.conf.runner=='EpisodesRunner' else self.conf.n_sp_ts//self.n_workers
        if eval_mode:
            n = self.conf.n_eval_ep
        res = ray.get([w.rollout.remote(n, self.conf.render, eval_mode) for w in self.workers])
        update_count(self.count, n*self.n_workers)
        sp_rollouts, sp_infos, sp_frames, sp_metrics = unpack(res)

        env_stats_list = [defaultdict(lambda:defaultdict(list))]

        if not eval_mode:
            env_stats = ray.get([w.get_home_env_stats.remote() for w in self.workers]) 

            for env_stat in env_stats:
                for k in env_stat:
                    for p in env_stat[k]:
                        env_stats_list[0][k][p].append(env_stat[k][p])

            for i in range(len(env_stats_list)):
                for k in env_stats_list[i]:
                    for p in env_stats_list[i][k]:
                        mean = np.mean([rms.mean for rms in env_stats_list[i][k][p]], axis=0)
                        var = np.mean([rms.var for rms in env_stats_list[i][k][p]], axis=0)
                        self.env_stats_list[i][k][p] = RunningMeanStd(shape=mean.shape)
                        self.env_stats_list[i][k][p].mean = mean
                        self.env_stats_list[i][k][p].var = var
                        self.env_stats_list[i][k][p].count = self.count
            logger.debug(f'env stats: {self.env_stats_list}')

        return Dotdict({'rollouts':sp_rollouts, 'infos':sp_infos, 'frames':sp_frames, 'metrics':sp_metrics})

    def train(self):
        self.iter += 1
        data = self._collect()

        rollouts = arrdict.merge_and_cat(data.rollouts)
        metrics = merge_dict(data.metrics)
        avg_metrics = get_avg_metrics([metrics])
        for k,v in avg_metrics.items():
            if isinstance(v, Iterable):
                logger.add_scalars(k, {i:v[i] for i in range(len(v))}, self.iter)
            logger.add_scalar(k, np.mean(v), self.iter)
        if self.conf.flatten_traj:
            rollouts = flatten_traj(rollouts)
        train_info = self.agent_pop[0].train(rollouts)
        del data
        gc.collect()
        return train_info

    def evaluate(self):
        self.agent_pop[0].explore = False
        if self.conf.vary_z_eval:
            if self.conf.z_discrete:
                z_list = np.eye(self.conf.z_dim)

            data, frames, metrics, avg_ret = [[None] * len(z_list) for _ in range(4)]
            for i in range(0, len(z_list), self.n_workers):
                start = i
                end = i + self.n_workers
                ray.get([w.set_z.remote(z, eval_mode=True) for w,z in zip(self.workers, z_list[start:end])])

                d = self._collect(eval_mode=True)

                for j in range(self.n_workers):
                    data[start+j] = Dotdict({k:v[j] for k,v in d.items()})
                    frames[start+j] = d['frames'][j]
                    metrics[start+j] = d['metrics'][j]
                    avg_ret[start+j] = np.mean(list(metrics[j].avg_ret.values()))

            ray.get([w.unset_z.remote() for w in self.workers])            
            data = merge_dict(data)

        else:
            data = self._collect(eval_mode=True)
            frames = data['frames']
            metrics = data['metrics']
            avg_ret = np.mean([list(metrics[i].avg_ret.values()) for i in range(len(metrics))])
            logger.add_scalar('eval/avg_return', avg_ret, self.iter)

        self.agent_pop[0].explore = True 
        logger.info(pblock(avg_ret, 'Average return...'))
        if self.conf.render:
            self._log_gif(frames)
        del data['frames']
        data = Dotdict({'sp_data':data})
        os.makedirs(f'{self.conf.save_dir}/it_{self.iter:05d}', exist_ok=True)
        pickle.dump(data, open(f'{self.conf.save_dir}/it_{self.iter:05d}/data.pkl','wb'))
        del data
        del frames
        del avg_ret
        del metrics
        gc.collect()
        
        return None, None, None, None

    def save(self, save_dir=None):
        if save_dir is None:
            save_dir = self.conf.save_dir
        path = f'{save_dir}/it_{self.iter:05d}'
        os.makedirs(path, exist_ok=True)
        env_tar = dict(env_conf=self.env_conf, eval_env_conf=self.eval_env_conf)
        env_tar['env_stats'] = ray.get([w.get_env_stats.remote() for w in self.workers])
        
        torch.save(env_tar, f'{path}/env_data.tar')
        def _save(agent, env_stat, i, env_conf, eval_env_conf):
            agent_tar = dict()
            agent_state = agent.get_state()
            if self.conf.algo_name=='sp_mi':
                agent_state['z'] = torch.eye(self.conf.z_dim)[i%self.conf.z_dim]
            agent_tar['agent_state'] = agent_state
            agent_tar['config'] = self.conf
            agent_tar['agent_no'] = i
            # each worker has its own mean, std stats
            # save only sp env
            agent_tar['env_stats'] = env_stat
            agent_tar['env_conf'] = env_conf
            agent_tar['eval_env_conf'] = eval_env_conf
            torch.save(agent_tar, f'{path}/agent_{i}.tar')

        if self.conf.algo_name=='sp_mi' or self.conf.algo_name=='maven':
            assert self.conf.z_discrete, f'Currently only handle z discrete'
            # repeat z_dim times
            len_repeated = len(self.agent_pop)*self.conf.z_dim
            env_stats = [env_tar['env_stats'][i//self.conf.z_dim] for i in range(len_repeated)]
            p_save = partial(_save, env_conf=self.env_conf, eval_env_conf=self.eval_env_conf)
            for agent, env_stat, i in zip([self.agent_pop[i//self.conf.z_dim] for i in range(len_repeated)],\
                                      env_stats, range(len_repeated)):
                p_save(agent, env_stat, i)
        else:
            env_stats_sp = [env_tar['env_stats'][i] for i in range(self.conf.pop_size)]
            p_save = partial(_save, env_conf=self.env_conf, eval_env_conf=self.eval_env_conf)
            for (agent, env_stat, i) in zip(self.agent_pop, env_stats_sp, range(len(self.agent_pop))):
                p_save(agent, env_stat, i)

        torch.save({'agent_pop': self.agent_pop,
                    'iter': self.iter}, 
                    f'{path}/trainer_checkpoint.pt')

        gc.collect()


