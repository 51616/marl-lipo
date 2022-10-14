import gc
import os
import pickle
import random
import psutil 
from collections import defaultdict, deque, Counter
from itertools import product
from functools import partial

import numpy as np
import ray
import torch

from coop_marl.trainers import Trainer
from coop_marl.agents import registered_agents, IncompatMAPPOZ, QMIXAgent
from coop_marl.envs import registered_envs
from coop_marl.envs.wrappers import wrap, RunningMeanStd
from coop_marl.utils import get_logger, pblock, Dotdict, save_gif, merge_dict
from coop_marl.worker import RolloutWorker
from coop_marl.evaluation import update_workers
from coop_marl.utils.metrics import get_avg_metrics

logger = get_logger()
p_save_gif = partial(save_gif, fps=30, size=(200,200))

class SWEpsilonGreedyBandit:
    def __init__(self, n_arms, eps, window_size):
        self.n_arms = n_arms
        self.eps = eps
        self.means = np.full([self.n_arms,], None)
        self.rets = [deque(maxlen=window_size) for _ in range(self.n_arms)]
        self.taken_actions = []
    
    def __repr__(self):
        return f'N arms: {self.n_arms}, Epsilon: {self.eps}, Means: {self.means}, Returns: {self.rets}\n\
                Taken actions: {Counter(self.taken_actions)}'

    def get_best_arms(self,k=1):
        if None in self.means:
            none_pos = np.where(self.means==None)[0]
            n_nones = len(none_pos)
            if k>n_nones:
                # all none will be -inf entries in self.means
                return random.choices(none_pos, k=n_nones) + self.get_best_arms(k-n_nones)
            return random.choices(none_pos, k=k)
        # get best k arms
        return np.argsort(self.means)[:-(1+k):-1]

    def get_random_arms(self, n_best, k):
        random_arms = [i for i in range(self.n_arms) if i not in self.get_best_arms(n_best)]
        if k>len(random_arms):
            return random_arms
        return random.sample(random_arms, k=k)

    def select_arms(self, k=1):
        assert k <= self.n_arms
        if None in self.means:
            arms = self.get_best_arms(k)
            for arm in arms:
                if self.means[arm] is None:
                    self.means[arm] = -np.inf
            # do this iteratively
        else:
            # explore at most (n_arms - k) arms
            noise = np.random.rand(k)
            arms = self.get_best_arms(k)
            explore_arms = self.get_random_arms(n_best=k, k=k) # exclude the best k arms
            j = 0
            for i, n in enumerate(noise):
                if j==len(explore_arms):
                    break
                if n<self.eps:
                    # replace least optimal arm first
                    arms[-(i+1)] = explore_arms[j]
                    j += 1

        self.taken_actions.extend(arms)
        return np.array(arms)

    def update(self, taken_actions, ret):
        assert len(taken_actions)==len(ret)
        for a, r in zip(taken_actions, ret):
            self.rets[a].append(r)
            self.means[a] = np.mean(self.rets[a])

class BanditPartnerSelector:
    def __init__(self, pop_size, idx, eps, window_size, *args, **kwargs):
        # use idx to indicate the corresponding agent
        self.bandit = SWEpsilonGreedyBandit(pop_size-1, eps, window_size)
        self.idx = idx

    def __repr__(self):
        return ' '.join([f'Index: {self.idx}', repr(self.bandit)])

    def select_partners(self, k=1):
        selected_partners = self.bandit.select_arms(k=k)
        # +1 to all partners that comes after or equals self.idx
        return np.where(selected_partners>=self.idx, selected_partners+1, selected_partners)

    def update(self, selected_partners, ret):
        selected_partners = np.array(selected_partners)
        taken_actions = np.where(selected_partners>=self.idx, selected_partners-1, selected_partners)
        return self.bandit.update(taken_actions, ret)

class UniformPartnerSelector:
    def __init__(self, pop_size, idx, keep_last=False, *args, **kwargs):
        # use idx to indicate the corresponding agent
        self.pop_size = pop_size
        self.idx = idx
        self.keep_last = keep_last
        self.last_best = None
        self.possible_partners = list(range(pop_size))
        self.possible_partners.remove(self.idx)
        self.taken_actions = []

    def __repr__(self):
        return f'N arms: {len(self.possible_partners)}, \n\
                Taken actions: {Counter(self.taken_actions)}'

    def select_partners(self, k=1):
        selected_partners = random.sample(self.possible_partners, k=k)
        self.taken_actions.extend(selected_partners)
        if self.keep_last and self.last_best:
            # replace one with the selected partners in the last iteration
            selected_partners[0] = self.last_best
        return selected_partners

    def update(self, selected_partners, ret):
        assert len(selected_partners)==len(ret)
        if self.keep_last:
            self.last_best = selected_partners[np.argmax(ret)]
        return 

class IncompatTrainer(Trainer):
    """
    LIPO specific trainer
    """
    def __init__(self, conf, env_conf, eval_env_conf=None):
        self._setup(conf, env_conf, eval_env_conf)
        self.conf = conf
        self.env_conf = env_conf
        self.eval_env_conf = eval_env_conf
        pop_size = conf.pop_size
        agent_cls = registered_agents[conf.agent_name]
        if not conf.checkpoint:
            self.agent_pop = [agent_cls(conf) for _ in range(conf.pop_size)]
            self.partner_selectors = [UniformPartnerSelector(pop_size, i, keep_last=conf.uniform_selector_keep_last) for i in range(pop_size)]
            if conf.use_bandit:
                self.partner_selectors = [BanditPartnerSelector(pop_size, i,
                                                                eps=conf.bandit_eps,
                                                                window_size=conf.bandit_window_size)
                                            for i in range(pop_size)]
        else:
            checkpoint = torch.load(f'{conf.checkpoint}/trainer_checkpoint.pt')
            self.agent_pop = checkpoint['agent_pop']
            self.partner_selectors = checkpoint['partner_selectors']
            self.iter = checkpoint['iter']
            self.count = checkpoint['count']

        logger.info('Initializing ray...')
        ray.init(include_dashboard=False, 
                local_mode=conf.debug, log_to_driver=conf.debug)
        self.num_xp_pair_sample = min(conf.num_xp_pair_sample, pop_size-1)
        self.sp_pairs = [(i,i) for i in range(pop_size)]
        self.xp_pairs = [(i,-1) for i,_ in product(range(pop_size), range(self.num_xp_pair_sample))]
        
        self.n_workers = min(self.n_pairs, conf.n_workers, psutil.cpu_count(logical=False))
        self.workers = [RolloutWorker.remote(conf, env_conf, eval_env_conf) for _ in range(self.n_workers)]

    def _setup(self, conf, env_conf, eval_env_conf=None):
        self.iter = 0
        self.count = [0] * conf.pop_size
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

        self.env_stats_list = [env.get_stats() for _ in range(conf.pop_size)] # initial env_stats for each agent
        obs = env.reset()
        p = list(obs.keys())[0]
        self.init_states = obs[p].state

        env_conf['name'] = reg_env_name

        conf['obs_space'] = env.get_observation_space()['obs']
        conf['act_space'] = env.get_action_space()
        conf['n_agents'] = len(env.players)
        assert conf['n_agents'] in [1,2] , f'Only support 2-player game now (no home/away mode implemnetation yet)'
        try:
            conf['state_shape'] = env.get_state_shape()
        except AttributeError:
            conf['state_shape'] = None

    def _log_gif(self, frames, xp=True):
        if xp:
            f = [frames[i][j] for i,j in product(range(len(frames)),range(len(frames[0])))]
            paths = [f'{self.conf.save_dir}/it_{self.iter:05d}/renders/{i}-{j}.gif' for i,j in product(range(len(frames)),range(len(frames)))]
        else:
            f = [frames[i] for i in range(len(frames))]
            paths = [f'{self.conf.save_dir}/it_{self.iter:05d}/renders/0-{i}.gif' for i in range(len(frames))]
        for frame,p in zip(f,paths):
            p_save_gif(frame,p)
        del f
        del frames
        gc.collect()

    @property
    def n_pairs(self):
        return len(self.pairs)

    @property
    def pairs(self):
        return self.sp_pairs + self.xp_pairs

    @property
    def agent_state_list(self):
        return [a.get_state() for a in self.agent_pop]

    def _collect(self, eval_mode):
        """
        Each worker has a separate env stats
        But the stats should be pulled from the agents themselves
        """
        pop_size = self.conf.pop_size
        res = []
        env_stats_list = [defaultdict(lambda:defaultdict(list)) for _ in range(pop_size)] 

        def unpack(res):
            rollouts, infos, frames, metrics = [[None] * len(res) for _ in range(4)]
            for i, r in enumerate(res):
                rollouts[i], infos[i], frames[i], metrics[i] = r['rollouts'], r['infos'], r['frames'], r['metrics']
            return rollouts, infos, frames, metrics

        def get_n_list(pairs, eval_mode):
            if eval_mode:
                return [self.conf.n_eval_ep] * len(pairs)

            n_list = []
            for (i,j) in pairs:
                n = self.conf.n_sp_episodes if i==j else self.conf.n_xp_episodes
                if not eval_mode and (self.conf.runner=='StepsRunner'):
                    n = self.conf.n_sp_ts if i==j else self.conf.n_xp_ts
                n_list.append(n)
            return n_list

        def update_count(count_list, pairs, n_list):
            assert len(pairs)==len(n_list)
            for pair, n in zip(pairs, n_list):
                for i in pair:
                    self.count[i] += n

        for k in range(0, self.n_pairs, self.n_workers):
            # loop over all pairs
            start = k
            end = k + self.n_workers
            pairs = self.pairs[start:end]
            n_list = get_n_list(pairs, eval_mode)
            update_count(self.count, pairs, n_list)
            update_workers(self.workers, self.pairs[start:end], [{'agent_conf':self.conf}]*pop_size,
                            self.agent_state_list, self.env_stats_list)
            res.extend(ray.get([w.rollout.remote(n, self.conf.render, eval_mode, self.conf.render_only_sp)
                               for w,n in zip(self.workers, n_list)]))

            if not eval_mode:
                worker_controllers = ray.get([w.get_controller.remote() for w in self.workers])
                for i, pair in enumerate(self.pairs[start:end]):
                    if isinstance(self.agent_pop[pair[0]], QMIXAgent) or isinstance(self.agent_pop[pair[1]], QMIXAgent):
                        assert pair[0]==pair[1] # currently only handle self-play for QMIX - MAVEN

                        worker_agent = list(worker_controllers[i].agents_dict.values())[0]
                        self.agent_pop[pair[0]].cur_step = worker_agent.cur_step
                        self.agent_pop[pair[0]].eps = worker_agent.eps

                home_env_stats = ray.get([w.get_home_env_stats.remote() for w in self.workers]) # dict[stat: dict[player_name: val]]
                away_env_stats = ray.get([w.get_away_env_stats.remote() for w in self.workers])
                env_stats = list(zip(home_env_stats,away_env_stats))

                for home_away_stats, pair in zip(env_stats,pairs):
                    for side,i in enumerate(pair):
                        stat = home_away_stats[side]
                        for k in stat:
                            for p in stat[k]:
                                env_stats_list[i][k][p].append(stat[k][p])
        if not eval_mode:
            for i in range(len(env_stats_list)):
                for k in env_stats_list[i]:
                    for p in env_stats_list[i][k]:
                        mean = np.mean([rms.mean for rms in env_stats_list[i][k][p]], axis=0)
                        var = np.mean([rms.var for rms in env_stats_list[i][k][p]], axis=0)
                        self.env_stats_list[i][k][p] = RunningMeanStd(shape=mean.shape)
                        self.env_stats_list[i][k][p].mean = mean
                        self.env_stats_list[i][k][p].var = var
                        self.env_stats_list[i][k][p].count = self.count[i]
        logger.debug(f'env stats: {self.env_stats_list}')

        # this code takes diagonal entries in res
        sp_rollouts, sp_infos, sp_frames, sp_metrics = unpack(res[:len(self.agent_pop)])

        # grab xp data from off diagonal workers
        xp_rollouts, xp_infos, xp_frames, xp_metrics = unpack(res[len(self.agent_pop):])
    
        sp_data = Dotdict({'rollouts':sp_rollouts, 'infos':sp_infos, 'frames':sp_frames, 'metrics':sp_metrics})
        xp_data = Dotdict({'rollouts':xp_rollouts, 'infos':xp_infos, 'frames':xp_frames, 'metrics':xp_metrics})
        return sp_data, xp_data

    def _get_home_away_rollouts(self, xp_rollouts, xp_pairs):
        # the input will be a 1d list
        assert len(xp_rollouts)==len(xp_pairs)
        pop_size = self.conf.pop_size
        t = [None] * pop_size**2
        for r,(i,j) in zip(xp_rollouts, xp_pairs):
            t[i*pop_size+j] = r

        home_rollouts = [[None] * pop_size for _ in range(pop_size)] # pop_size x pop_size list
        away_rollouts = [[None] * pop_size for _ in range(pop_size)]
        for k, r in enumerate(t):
            if r is None:
                continue
            i = k // pop_size
            j = k % pop_size
            if self.conf.parent_only:
                if i>j:
                    home_rollouts[i][j] = r[0]
                if j>i:
                    away_rollouts[j][i] = r[1]
            else:
                if i!=j:
                    home_rollouts[i][j] = r[0]
                    away_rollouts[j][i] = r[1]
        return home_rollouts, away_rollouts

    def _max_mask(self, home_rollouts, away_rollouts):
        # remove data from non-max average return pair inplace
        assert len(home_rollouts)==len(away_rollouts)
        max_mask = np.zeros([len(away_rollouts), len(away_rollouts)], dtype=bool)

        for i in range(len(away_rollouts)):
            ret = np.ma.zeros(len(home_rollouts))
            ret.mask = True
            for rollouts in [home_rollouts, away_rollouts]:
                for j, r in enumerate(rollouts[i]):
                    if r is not None:
                        ret.mask[j] = False
                        ret[j] += r.outcome.reward.mean()
                
            idx = np.argmax(ret)
            max_mask[i, idx] = True
        return max_mask

    def _get_mask(self, home_rollouts, away_rollouts):
        value_mask = np.ones([len(home_rollouts), len(home_rollouts)], dtype=np.float32)
        pg_mask = np.ones([len(home_rollouts), len(home_rollouts)], dtype=np.float32) / len(home_rollouts)
        # use only rollouts with highest return
        if self.conf.pg_xp_max_only:
            pg_mask = self._max_mask(home_rollouts, away_rollouts)
        if self.conf.value_xp_max_only:
            value_mask = self._max_mask(home_rollouts, away_rollouts)
        return pg_mask, value_mask

    def _sample_xp_pairs(self, all_pairs=False):
        self.xp_pairs = []
        if all_pairs or (self.num_xp_pair_sample>=(self.conf.pop_size-1)):
            inv_eye = np.ones([self.conf.pop_size, self.conf.pop_size]) - np.eye(self.conf.pop_size)
            row,col = np.where(inv_eye>0)
            self.xp_pairs = list(zip(row,col))
            return 

        pop_size = self.conf.pop_size
        for i in range(pop_size):
            xp_partners = self.partner_selectors[i].select_partners(k=self.num_xp_pair_sample)
            for j in xp_partners:
                self.xp_pairs.append((i,j))

    def train(self):
        self.iter += 1
        self._sample_xp_pairs()

        logger.debug(f'XP pairs: {self.xp_pairs}')
        sp_data, xp_data = self._collect(eval_mode=False)
        # define sp_avg_ret
        avg_metrics = dict()
        for data, name in zip([sp_data, xp_data], ['sp','xp']):
            avg_metrics[name] = get_avg_metrics(data.metrics) # dotdict of lists
            for k,v in avg_metrics[name].items():
                logger.add_scalars(name, {f'{k}_{i}':v[i] for i in range(len(v))}, self.iter)
                logger.add_scalar(f'{name}/{k}', np.mean(v), self.iter)
        if len(avg_metrics['xp']):
            compat_gap = np.mean(avg_metrics['sp']['avg_ret']) - np.mean(avg_metrics['xp']['avg_ret'])
            logger.add_scalar('compatibility_gap', compat_gap, self.iter)

        sp_rollouts, xp_rollouts = sp_data['rollouts'], xp_data['rollouts']
        home_rollouts, away_rollouts = self._get_home_away_rollouts(xp_rollouts, self.xp_pairs)
        pg_mask, value_mask = self._get_mask(home_rollouts, away_rollouts)

        for i in range(self.conf.pop_size):
            # put only view of agent_i to agent_i.train()
            # away game -> traj that played out using param_i for one agent and param_j for the rest
            # home game -> traj that played out using param_i for all agents and param_j for one agent
            # select correct corresponding players for agent_i first before sending them to train()
            if isinstance(self.agent_pop[i], IncompatMAPPOZ):
                self.agent_pop[i].train(sp_rollout=sp_rollouts[i],
                                   away_rollouts=away_rollouts[i],
                                   home_rollouts=home_rollouts[i],
                                   pg_mask=pg_mask[i],
                                   value_mask=value_mask[i]
                                   )
            else:
                self.agent_pop[i].train(sp_rollouts[i])
        logger.debug(f'PG mask: {pg_mask}')
        if self.conf.anneal_lr:
            logger.add_scalar('learning_rate', self.agent_pop[0].cur_lr, self.iter)

        payoff_matrix = np.ma.zeros([self.conf.pop_size, self.conf.pop_size])
        payoff_matrix.mask = True
        xp_metrics = xp_data['metrics']
        for k, (i,j) in enumerate(self.xp_pairs):
           payoff_matrix[i,j] = np.mean(list(xp_metrics[k].avg_ret.values()))
        for i, selector in enumerate(self.partner_selectors):
            payoff = payoff_matrix[i:]
            taken_actions = []
            rets = []
            for payoff in [payoff_matrix[i,:], payoff_matrix[:,i]]:
                taken_actions.extend([j for j,x in enumerate(payoff) if x is not np.ma.masked])
                rets.extend([x for x in payoff[~payoff.mask]])
            selector.update(taken_actions, rets)

        logger.debug(f'Selectors: {self.partner_selectors}')
        return {}

    def evaluate(self):
        self._sample_xp_pairs(all_pairs=self.conf.eval_all_pairs)
        pop_size = self.conf.pop_size
        xp_data = None
        if self.conf.vary_z_eval:
            assert len(self.xp_pairs)==0
            if self.conf.z_discrete:
                z_list = np.eye(self.conf.z_dim)
            sp_data, frames, metrics, avg_ret = [[None] * len(z_list) * self.n_workers for _ in range(4)]
            for i, z in enumerate(z_list):
                ray.get([w.set_z.remote(z, eval_mode=True) for w in self.workers])
                data, _ = self._collect(eval_mode=True)
                for j in range(self.n_workers):
                    sp_data[i*self.n_workers+j] = Dotdict({'rollouts':data.rollouts[j],
                                                             'infos':data.infos[j],
                                                             'frames':data.frames[j],
                                                             'metrics':data.metrics[j]})

                    frames[i*self.n_workers+j] = sp_data[i*self.n_workers+j]['frames']
                    metrics[i*self.n_workers+j] = sp_data[i*self.n_workers+j]['metrics']
                    avg_ret[i*self.n_workers+j] = np.mean(list(metrics[i*self.n_workers+j].avg_ret.values()))
            ray.get([w.unset_z.remote() for w in self.workers])
            logger.info(pblock(avg_ret, 'Average return...'))
            sp_data = merge_dict(sp_data)
        else:
            sp_data, xp_data = self._collect(eval_mode=True)

            frames = [[None] * pop_size for _ in range(pop_size)]
            sp_metrics, xp_metrics = sp_data['metrics'], xp_data['metrics']
            payoff_matrix = np.full([self.conf.pop_size, self.conf.pop_size], None)
            
            for i in range(pop_size):
                payoff_matrix[i,i] = np.mean(list(sp_metrics[i].avg_ret.values()))
                frames[i][i] = sp_data['frames'][i]

            for k, (i,j) in enumerate(self.xp_pairs):
                payoff_matrix[i,j] = np.mean(list(xp_metrics[k].avg_ret.values()))
                frames[i][j] = xp_data['frames'][k]
            logger.info(pblock(payoff_matrix, 'Payoff matrix...'))
        if self.conf.render:
            self._log_gif(frames, xp=not self.conf.vary_z_eval)
        del frames
        # save data from evaluation
        del sp_data['frames']
        if xp_data is not None:
            del xp_data['frames']
        data = Dotdict({'sp_data':sp_data, 'xp_data':xp_data})
        os.makedirs(f'{self.conf.save_dir}/it_{self.iter:05d}', exist_ok=True)
        pickle.dump(data, open(f'{self.conf.save_dir}/it_{self.iter:05d}/data.pkl','wb'))
        return None, None, None, None

    def save(self, save_dir=None):
        if save_dir is None:
            save_dir = self.conf.save_dir
        path = f'{save_dir}/it_{self.iter:05d}'
        os.makedirs(path, exist_ok=True)
        env_tar = dict(env_conf=self.env_conf, eval_env_conf=self.eval_env_conf)
        env_tar['env_stats'] = self.env_stats_list
        
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
        if ('sp_mi' in self.conf.algo_name) or ('maven' in self.conf.algo_name):
            assert self.conf.z_discrete, f'Currently only handle z discrete'
            # repeat z_dim times
            len_repeated = len(self.agent_pop)*self.conf.z_dim
            env_stats = [env_tar['env_stats'][i//self.conf.z_dim] for i in range(len_repeated)]
            p_save = partial(_save, env_conf=self.env_conf, eval_env_conf=self.eval_env_conf)
            for agent, env_stat, i in zip([self.agent_pop[i//self.conf.z_dim] for i in range(len_repeated)],\
                                      env_stats, range(len_repeated)):
                p_save(agent, env_stat, i)

        else:
            env_stats_sp = env_tar['env_stats']
            p_save = partial(_save, env_conf=self.env_conf, eval_env_conf=self.eval_env_conf)
            for (agent, env_stat, i) in zip(self.agent_pop, env_stats_sp, range(len(self.agent_pop))):
                p_save(agent, env_stat, i)

        torch.save({'agent_pop': self.agent_pop,
                    'iter': self.iter,
                    'count': self.count,
                    'partner_selectors': self.partner_selectors,
                    'env_stats_list': self.env_stats_list}, 
                    f'{path}/trainer_checkpoint.pt')
