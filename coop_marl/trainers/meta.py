import gc
import os
import pickle
from collections import defaultdict
from itertools import product
from functools import partial
from glob import glob

import numpy as np
import ray
import torch

from coop_marl.trainers import Trainer
from coop_marl.agents import registered_agents
from coop_marl.envs import registered_envs
from coop_marl.envs.wrappers import wrap, RunningMeanStd
from coop_marl.utils import get_logger, pblock, Dotdict, arrdict, save_gif
from coop_marl.evaluation import load_config, load_state, load_env_stats, update_workers, get_last_iter_dir
from coop_marl.worker import RolloutWorker

logger = get_logger()
p_save_gif = partial(save_gif, fps=30, size=(200,200))

def get_avg_metrics(metrics):
    # assume a list of dotdicts of structure {player_name: {metric: value}}
    # each corresponds to one agent 
    # take a mean over all players (e.g., avg return for an episodes)
    temp = [defaultdict(list) for _ in range(len(metrics))] # list of dicts of lists
    for i, m in enumerate(metrics):
        for p in m:
            for k,v in m[p].items():
                if isinstance(v, list):
                    temp[i][k].extend(v)
                else:
                    temp[i][k].append(v)

    out = defaultdict(list)
    for t in temp:
        for k in t:
            out[k].append(sum(t[k])/len(t[k])) # mean
    # out[k] is a list of means
    # each element corresponds to each element in the input metrics
    # (if metrics has 10 elem, out also has 10)
    return Dotdict(out)

def _get_tar_dir(out, p, iterations=None):
    if iterations is None:
        folder = get_last_iter_dir(p, complete_only=False)
        iterations = [int(folder.split('_')[-1].strip('/'))]
    for it in iterations:
        folder = f'{p}/it_{it:05d}/'
        out.extend(glob(f'{folder}/agent_*.tar'))

def get_tar_dir(paths, iterations=None):
    out = []
    for p in paths:
        if glob(f'{p}/*.yaml'):
            _get_tar_dir(out, p, iterations)
        else:
            for p in glob(f'{p}/*/'):
                _get_tar_dir(out, p ,iterations)
    return out

class MetaTrainer(Trainer):
    """
    Meta-trainer
    """
    def __init__(self, conf, env_conf, eval_env_conf=None):

        assert len(conf.partner_dir), f'Must have at least one trainer partner'
        # partner_dir is a list of agent_.tar
        self._setup(conf, env_conf, eval_env_conf)
        self.conf = conf
        self.env_conf = env_conf
        self.eval_env_conf = eval_env_conf
        agent_cls = registered_agents[conf.agent_name]
        self.meta_agent = agent_cls(conf)
        self.n = conf.n_ts if conf.runner=='StepsRunner' else conf.n_episodes
        self.partner_tar_dir = get_tar_dir(conf.partner_dir, conf.partner_iterations)
        self.n_partners = len(self.partner_tar_dir)
        logger.debug(f'Partner tar dir: {self.partner_tar_dir}')
        logger.info(f'N partners: {self.n_partners}')
        logger.info('Initializing ray...')
        ray.init(include_dashboard=False,
                 local_mode=conf.debug, log_to_driver=conf.debug)

        combined_conf = Dotdict(agent_conf=conf, env_conf=env_conf, eval_env_conf=eval_env_conf)
        self.config_list = [combined_conf] + [load_config(d) for d in self.partner_tar_dir]
        self.count = [0,0] # training and eval count

        # DO NOT CHANGE self.pairs
        self.pairs = [(0,i) for i in range(1,len(self.config_list))] + [(i,0) for i in range(1,len(self.config_list))]
        self.n_pairs = len(self.pairs)

        self.n_workers = min(self.n_pairs, self.conf.n_workers)
        self.workers = [RolloutWorker.remote(conf, env_conf, eval_env_conf) for _ in range(self.n_workers)]

        self.partner_state_list = [load_state(p) for p in self.partner_tar_dir]
        self.partner_env_stats_list = [load_env_stats(p) for p in self.partner_tar_dir]
        # the trainer has to keep tack of the env_stats itself
        # to sync all workers' env_stats for the meta-agent
        self.env_stats = ray.get(self.workers[0].get_env_stats.remote()) # aggregated stats
        logger.debug(f'initial env_stats: {self.env_stats}')
        
    @property
    def agent_state_list(self):
        return [self.meta_agent.get_state()] + self.partner_state_list

    @property
    def env_stats_list(self):
        return [self.env_stats] + self.partner_env_stats_list

    def get_env_stats(self):
        return self.env_stats

    def _setup(self, conf, env_conf, eval_env_conf=None):
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

        env_conf['name'] = reg_env_name
        conf['obs_space'] = env.get_observation_space()['obs']
        conf['act_space'] = env.get_action_space()
        conf['n_agents'] = len(env.players)
        try:
            conf['state_shape'] = env.get_state_shape()
        except AttributeError:
            conf['state_shape'] = None

    def _log_gif(self, frames, xp=True):
        f = [frames[i][j] for i,j in product(range(len(frames)),range(len(frames[0])))] # assume a 2d list
        paths = [f'{self.conf.save_dir}/it_{self.iter:05d}/renders/{i}-{j}.gif' for i,j in product(range(len(frames)),range(len(frames[0])))]
        for frame,p in zip(f,paths):
            p_save_gif(frame,p)
        del f
        del frames
        gc.collect()

    def _get_home_away_rollouts(self, xp_rollouts):
        # the input will be a 1d list
        pop_size = self.conf.pop_size
        for i in range(pop_size):
            # pad the list to have the size of pop_size**2
            xp_rollouts.insert(i*pop_size+i, None)
        home_rollouts = [[None] * pop_size for _ in range(pop_size)] # pop_size x pop_size list
        away_rollouts = [[None] * pop_size for _ in range(pop_size)]
        for k, r in enumerate(xp_rollouts):
            i = k // pop_size
            j = k % pop_size
            if self.conf.parent_only:
                if i>j:
                    assert r[0] is not None
                    home_rollouts[i][j] = r[0]
                if j>i:
                    assert r[1] is not None
                    away_rollouts[j][i] = r[1]
            else:
                if i!=j:
                    assert r[0] is not None
                    assert r[1] is not None
                    home_rollouts[i][j] = r[0]
                    away_rollouts[j][i] = r[1]
        return home_rollouts, away_rollouts

    def _collect(self, eval_mode=False):
        res = []
        env_stats = []
        meta_env_stats = defaultdict(lambda:defaultdict(list))
        def unpack(res):
            rollouts, infos, frames, metrics = [[None] * len(res) for _ in range(4)]
            for i, r in enumerate(res):
                rollouts[i], infos[i], frames[i], metrics[i] = r['rollouts'], r['infos'], r['frames'], r['metrics']
            return rollouts, infos, frames, metrics
        

        for i in range(0, self.n_pairs, self.n_workers):
            # loop over all pairs
            start = i
            end = i + self.n_workers
            update_workers(self.workers, self.pairs[start:end], self.config_list,
                       self.agent_state_list, self.env_stats_list)
            if not eval_mode:
                n = self.n//self.n_pairs
            else:
                n = self.conf.n_eval_ep
            
            res.extend(ray.get([w.rollout.remote(n, self.conf.render, eval_mode) for w in self.workers]))
            self.count[eval_mode] += n * self.n_workers

            # get env_stats for meta_agent
            if not eval_mode:
                home_env_stats = ray.get([w.get_home_env_stats.remote() for w in self.workers])
                away_env_stats = ray.get([w.get_away_env_stats.remote() for w in self.workers])
                env_stats = list(zip(home_env_stats,away_env_stats))
                
                for home_away_stat, pair in zip(env_stats, self.pairs[start:end]):
                    pos = int(pair[1]==0)
                    stat = home_away_stat[pos] # dict[stat_name: dict[player_name:RMS]]
                    for k in stat:
                        for p in stat[k]:
                            meta_env_stats[k][p].append(stat[k][p])
        if not eval_mode:
            for k in meta_env_stats:
                for p in meta_env_stats[k]:
                    mean = np.mean([rms.mean for rms in meta_env_stats[k][p]], axis=0)
                    var = np.mean([rms.var for rms in meta_env_stats[k][p]], axis=0)
                    self.env_stats[k][p] = RunningMeanStd(shape=mean.shape)
                    self.env_stats[k][p].mean = mean
                    self.env_stats[k][p].var = var
                    self.env_stats[k][p].count = self.count[eval_mode]

        logger.debug(f'env stats: {self.env_stats}')
        rollouts, infos, frames, metrics = unpack(res) # each is a list of length num_xp_worker
        data = Dotdict({'rollouts':rollouts, 'infos':infos, 'frames':frames, 'metrics':metrics})
        return data

    def train(self):
        self.iter += 1
        data = self._collect()
        # get only meta agent's view
        
        pos = [int(p[1]==0) for p in self.pairs]
        meta_rollouts = []
        for p, rollout in zip(pos, data['rollouts']):
            meta_rollouts.append(rollout[p])
        avg_metrics = get_avg_metrics(data.metrics)
        for k,v in avg_metrics.items():
            logger.add_scalars('xp', {f'{k}_{i}':v[i] for i in range(len(v))}, self.iter)
            logger.add_scalar(f'{k}', np.mean(v), self.iter)

        meta_batch = arrdict.cat(meta_rollouts)
        self.meta_agent.train(meta_batch)

        del data
        del meta_rollouts
        del meta_batch
        gc.collect()
        return {}

    def evaluate(self):
        data = self._collect(eval_mode=True)
        frames = [[None] * self.n_partners for _ in range(2)]
        metrics = data.metrics
        avg_metrics = get_avg_metrics(metrics)
        payoff_matrix = np.zeros([2, self.n_partners])

        for i,j in product(range(2), range(self.n_partners)):
            payoff_matrix[i,j] = avg_metrics['avg_ret'][i*self.n_partners + j]
            frames[i][j] = data['frames'][i*self.n_partners + j]
        logger.info(pblock(payoff_matrix, 'Payoff matrix...'))

        if self.conf.render:
            self._log_gif(frames, xp=not self.conf.vary_z_eval)
        del frames
        # save data from evaluation
        del data['frames']
        data = Dotdict({'data':data})
        os.makedirs(f'{self.conf.save_dir}/it_{self.iter:05d}', exist_ok=True)
        pickle.dump(data, open(f'{self.conf.save_dir}/it_{self.iter:05d}/data.pkl','wb'))
        del data
        gc.collect()
        return None, None, None, None

    def save(self, save_dir=None):
        if save_dir is None:
            save_dir = self.conf.save_dir
        path = f'{save_dir}/it_{self.iter:05d}'
        os.makedirs(path, exist_ok=True)
        env_tar = dict(env_conf=self.env_conf, eval_env_conf=self.eval_env_conf)
        env_tar['env_stats'] = self.env_stats
        torch.save(env_tar, f'{path}/env_data.tar')

        def _save(agent, env_stat, i, env_conf, eval_env_conf):
            agent_tar = dict()
            agent_tar['agent_state'] = agent.get_state()
            agent_tar['config'] = self.conf
            agent_tar['agent_no'] = i
            # each worker has its own mean, std stats
            # save only sp env
            agent_tar['env_stats'] = env_stat
            agent_tar['env_conf'] = env_conf
            agent_tar['eval_env_conf'] = eval_env_conf
            torch.save(agent_tar, f'{path}/agent_{i}.tar')
        _save(self.meta_agent, self.env_stats, 0, self.env_conf, self.eval_env_conf)
        gc.collect()
