import gc
import pickle
import os
from functools import partial
from itertools import product

import numpy as np
import torch
import ray

from coop_marl.trainers import Trainer, collect_data
from coop_marl.agents import registered_agents
from coop_marl.envs import registered_envs
from coop_marl.envs.wrappers import wrap
from coop_marl.controllers import MappingController
from coop_marl.runners import  registered_runners, EpisodesRunner, StepsRunner
from coop_marl.utils import get_logger, pblock, Dotdict, Arrdict, arrdict, merge_dict, save_gif
from coop_marl.utils.rl import chop_into_episodes, compute_jsd_metric, get_discount_coef
from coop_marl.utils.metrics import get_avg_metrics

logger = get_logger()
p_save_gif = partial(save_gif, fps=30, size=(200,200))

@ray.remote
class RolloutWorker:
    """
    Two-agent runner
    """
    def __init__(self,conf, env_conf, eval_env_conf, home_id, away_id):
        self.conf = conf
        self.env_conf = env_conf
        self.eval_env_conf = eval_env_conf
        self.home_id = home_id
        self.away_id = away_id
        self.sp = home_id == away_id
        self._setup(conf, env_conf, eval_env_conf)

        agent_cls = registered_agents[conf.agent_name]
        self.agents_dict = {p:agent_cls(conf) for p in self.env.players}
        self.controller = MappingController(action_spaces=self.env.action_spaces,
                                   agents_dict=self.agents_dict,
                                   policy_mapping_fn=lambda name: name,
                                   )

        runner_cls = registered_runners[self.conf.runner]
        self.runner = runner_cls(env=self.env, controller=self.controller)
        self.eval_runner = EpisodesRunner(env=self.eval_env, controller=self.controller)

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

    def get_home_id(self):
        return self.home_id

    def get_away_id(self):
        return self.away_id

    def set_home_param(self, param):
        self.home_param = param

    def set_away_param(self, param):
        self.away_param = param

    def get_env_stats(self):
        return self.env.get_stats()

    def _collect_data(self, n=None, eval=False, render=False, render_mode='rgb_array'):
        runner = self.runner
        if eval:
            runner = self.eval_runner
            self.eval_env.set_stats(**self.env.get_stats())
        if n is None:
            if isinstance(runner, EpisodesRunner):
                n = self.conf.n_ep
            elif isinstance(runner, StepsRunner):
                n = self.conf.n_ts
        return collect_data(runner, n, render, render_mode)

    def _sp_rollout(self, eval_mode):
        setup_sp(self.controller, self.home_param)
        n = self.conf.n_sp_episodes if not eval_mode else self.conf.n_eval_ep
        if not eval_mode and isinstance(self.runner, StepsRunner):
            n = self.conf.n_sp_ts
        sp_rollout, infos, frames, metrics = self._collect_data(n, eval_mode, self.conf.render * eval_mode, self.conf.render_mode)
        return Dotdict({'rollouts':sp_rollout, 'infos':infos, 'frames':frames, 'metrics':metrics})

    def _xp_rollout(self, eval_mode):
        home_rollout, away_rollout = None, None
        infos, frames, metrics = [[None] * self.conf.n_agents for _ in range(3)]
        for k in range(self.conf.n_agents):
            setup_xp(self.controller, self.home_param, self.away_param, self.home_id, self.away_id, k)
            n = self.conf.n_sp_episodes if not eval_mode else self.conf.n_eval_ep
            if not eval_mode and isinstance(self.runner, StepsRunner):
                n = self.conf.n_xp_ts
            xp_rollout, infos[k], frames[k], metrics[k] = self._collect_data(n//self.conf.n_agents,
                                          eval_mode, self.conf.render * eval_mode, self.conf.render_mode)
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


    def rollout(self, eval_mode=False):
        if self.sp:
            out = self._sp_rollout(eval_mode)
        else:
            out = self._xp_rollout(eval_mode)
        return out

def setup_xp(controller, home_param, away_param, home_id, away_id, k):
    for a in controller.agents_dict.values():
        a.set_param(home_param)
    player_k = list(controller.agents_dict.values())[k]
    player_k.set_param(away_param)

def setup_sp(controller, param):
    for a in controller.agents_dict.values():
        a.set_param(param)

class TrajeDiTrainer(Trainer):
    def __init__(self, conf, env_conf, eval_env_conf=None):
        ray.init(include_dashboard=False, 
                 local_mode=conf.debug, log_to_driver=conf.debug)
        self._setup(conf, env_conf, eval_env_conf)
        self.conf = conf
        self.env_conf = env_conf
        self.eval_env_conf = eval_env_conf
        agent_cls = registered_agents[conf.agent_name]
        
        if not conf.checkpoint:
            self.agent_pop = [agent_cls(conf) for _ in range(conf.pop_size)]

        else:
            checkpoint = torch.load(f'{conf.checkpoint}/trainer_checkpoint.pt')
            self.agent_pop = checkpoint['agent_pop']
            self.iter = checkpoint['iter']

        self.p_compute_jsd_metric = partial(compute_jsd_metric, agent_pop=self.agent_pop,
                                            kernel_gamma=conf.kernel_gamma,
                                            use_br=conf.use_br)
        self.workers = [RolloutWorker.remote(conf, env_conf, eval_env_conf, i, i) for i in range(conf.pop_size)]

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

    def _log_gif(self, frames):
        f = [frames[i][j] for i,j in product(range(len(frames)),range(len(frames)))]
        paths = [f'{self.conf.save_dir}/it_{self.iter:05d}/renders/{i}-{j}.gif' for i,j in product(range(len(frames)),range(len(frames)))]
        for frame,p in zip(f,paths):
            p_save_gif(frame,p)
        del f
        del frames
        gc.collect()

    def _update_workers(self):
        home_ids = ray.get([w.get_home_id.remote() for w in self.workers])
        home_params = [self.agent_pop[home_id].get_param() for home_id in home_ids]
        away_ids = ray.get([w.get_away_id.remote() for w in self.workers])
        away_params = [self.agent_pop[away_id].get_param() for away_id in away_ids]
        _ = ray.get([w.set_home_param.remote(home_param) for w, home_param in zip(self.workers, home_params)])
        _ = ray.get([w.set_away_param.remote(away_param) for w, away_param in zip(self.workers, away_params)])
    
    def _start_workers(self, eval_mode=False):
        # assume workers is a 2d list
        ref_list = [w.rollout.remote(eval_mode) for w in self.workers]
        return ref_list

    def _get_home_away_rollouts(self, xp_rollouts):
        # the input will be a 1d list
        pop_size = self.conf.pop_size
        for i in range(pop_size):
            # pad the list to have the size of pop_size**2
            xp_rollouts.insert(i*pop_size+i, None)
        home_rollouts = [[None] * pop_size for _ in range(pop_size)]
        away_rollouts = [[None] * pop_size for _ in range(pop_size)]
        for k, r in enumerate(xp_rollouts):
            i = k // pop_size
            j = k % pop_size
            assert r[0] is not None
            assert r[1] is not None
            home_rollouts[i][j] = r[0]
            away_rollouts[j][i] = r[1]
        return home_rollouts, away_rollouts

    def _collect(self, eval_mode=False):
        pop_size = self.conf.pop_size
        ref_list = []
        self._update_workers()
        ref_list.extend(self._start_workers(eval_mode))
        res = ray.get(ref_list)

        def unpack(res, eval_mode=False):
            # remove the player keys as this method does not need a centralized critic
            rollouts, infos, frames, metrics = [[Arrdict() for _ in range(len(res))] for _ in range(4)]
            for i, r in enumerate(res):
                ro, infos[i], frames[i], metrics[i] = r['rollouts'], r['infos'], r['frames'], r['metrics']
                if eval_mode:
                    rollouts[i] = ro
                else:
                    for p in ro.inp.data:
                        batch = getattr(ro, p) # remove player keys from traj
                        rollouts[i] = arrdict.merge_and_cat([rollouts[i], batch])
            return rollouts, infos, frames, metrics

        def print_metrics(metrics, text=''):
            for i,m in enumerate(metrics):
                print(f'{text} {i}: {list(metrics[i].avg_reward.values())[0]}')

        # grab sp data along the diagonal
        end = pop_size
        if self.conf.use_br:
            end += 1
        sp_rollouts, sp_infos, sp_frames, sp_metrics = unpack(res[:end], eval_mode) # each is a list of length num_sp_workers
        sp_data = Dotdict({'rollouts':sp_rollouts, 'infos':sp_infos, 'frames':sp_frames, 'metrics':sp_metrics})
        xp_data = None
        return sp_data, xp_data

    @torch.no_grad()
    def _compute_jsd_metric(self, sp_rollouts):
        pop_size = self.conf.pop_size
        ep_list = [chop_into_episodes(batch) for batch in sp_rollouts] # [pop_size, n_ep]
        max_n_ep = max([len(episodes) for episodes in ep_list])
        max_len = max([ep.inp.data.obs.shape[0] for episodes in ep_list for ep in episodes])
        ep_pad_mask = np.ones([pop_size, max_n_ep, 1], dtype=bool)
        if not all(np.array([len(episodes) for episodes in ep_list]) == max_n_ep):
            feat_size = ep_list[0][0].inp.data.obs.shape[1]
            dummy_obs = np.zeros([max_len, feat_size])
            action_shape = ep_list[0][0].decision.action.shape
            action_size = action_shape[-1]
            dummy_action = np.zeros([max_len, action_size])
            if len(action_shape)==1:
                action_size = None
                dummy_action = np.zeros([max_len])
            dummy_ep = Arrdict(inp=Arrdict(data=Arrdict(obs=dummy_obs)),
                               decision=Arrdict(action=dummy_action))
            # pad number of episodes
            for i, episodes in enumerate(ep_list):
                n_padded = max_n_ep - len(episodes)
                ep_pad_mask[i, -n_padded:] = False
                episodes.extend([dummy_ep] * n_padded)

        assert all(np.array([len(episodes) for episodes in ep_list]) == max_n_ep)

        time_pad_mask = np.ones([pop_size, max_n_ep, max_len], dtype=bool)
        if not all(np.array([ep.inp.data.obs.shape[0] for episodes in ep_list for ep in episodes])==max_len):
            # pad the observations and actions for batch compute
            for i, episodes in enumerate(ep_list):
                for j, ep in enumerate(episodes):
                    time_pad_len = max_len - ep.inp.data.obs.shape[0]
                    time_pad_mask[i,j,-time_pad_len:] = False
                    ep.inp.data['obs'] = arrdict.postpad(ep.inp.data.obs, max_len=max_len, dim=0)
                    ep.decision['action'] = arrdict.postpad(ep.decision.action, max_len=max_len, dim=0)

        assert all(np.array([ep.inp.data.obs.shape[0] for episodes in ep_list for ep in episodes])==max_len)

        obs_batch = torch.tensor([ep.inp.data.obs for episodes in ep_list for ep in episodes], dtype=torch.float) # [-1, max_len, obs_size]
        obs_batch = obs_batch.view(pop_size*max_n_ep*max_len, *obs_batch.shape[2:])
        action_batch = torch.tensor([ep.decision.action for episodes in ep_list for ep in episodes], dtype=torch.float) # [-1, max_len, obs_size]
        action_batch = action_batch.view(pop_size*max_n_ep*max_len, *action_batch.shape[2:])

        act_logprob = torch.empty([pop_size, max_n_ep, pop_size, max_len])
        pi = torch.empty([pop_size, max_n_ep, pop_size])
        delta = torch.empty([pop_size, max_n_ep, pop_size, max_len])

        for i in range(pop_size):
            agent = self.agent_pop[i]
            act_dist = agent.calc_action_dist(obs_batch)
            act_logprob_i = act_dist.log_prob(action_batch).reshape(pop_size, max_n_ep, max_len) # [pop_size, max_n_ep, max_len]
            act_logprob_i = act_logprob_i * ep_pad_mask
            act_logprob_i = act_logprob_i * time_pad_mask

            act_logprob[:,:,i,:] = act_logprob_i
            pi[:,:,i] = torch.sum(act_logprob_i, axis=-1).exp() # [pop_size, max_n_ep]
            d = get_discount_coef(self.conf.kernel_gamma, max_len, device=act_logprob.device)[None,None,:].repeat(pop_size,max_n_ep,1,1) # [pop_size, max_n_ep, max_len, max_len]
            delta[:,:,i,:] = torch.sum(d * act_logprob_i.unsqueeze(2), axis=3).exp() # [pop_size, max_n_ep, max_len] (delta[t]) is delta_t
        # mean over policies
        delta_hat = torch.mean(delta, axis=2)
        pi_hat = torch.mean(pi, axis=2)

        for i, episodes in enumerate(ep_list):
            for j,ep in enumerate(episodes):
                ep['act_logprob'] = act_logprob[i,j,:,:] # rollout_idx, ep_idx, agent_idx, timestep
                ep['pi'] = pi[i,j,:]
                ep['delta'] = delta[i,j,:,:]
                ep['delta_hat'] = delta_hat[i,j,:]
                ep['pi_hat'] = pi_hat[i,j]
        return ep_list, ep_pad_mask, time_pad_mask

    def train(self):
        self.iter += 1
        pop_size = self.conf.pop_size
        sp_data, xp_data = self._collect()
        
        avg_metrics = dict()
        for data, name in zip([sp_data, xp_data], ['sp','xp']):
            if data is None:
                continue
            avg_metrics[name] = get_avg_metrics(data.metrics) # dotdict of lists
            for k,v in avg_metrics[name].items():
                logger.add_scalars(name, {f'{k}_{i}':v[i] for i in range(len(v))}, self.iter)
                logger.add_scalar(f'{name}/{k}', np.mean(v), self.iter)

        sp_rollouts = sp_data['rollouts']
        if xp_data is not None:
            home_rollouts, away_rollouts = self._get_home_away_rollouts(xp_data['rollouts'])

        ep_list = [None] * pop_size
        if self.conf.diverse_coef != 0:
            r = sp_rollouts
            if self.conf.use_br:
                r = sp_rollouts[:-1]
            ep_list, ep_pad_mask, time_pad_mask = self._compute_jsd_metric(r)
            
        for i in range(pop_size):
            # compute self-play objective with batch sampled from sp_buffer[i]
            # compute xp objective with batch sampled from xp_buffer[i]
            # compute grad for sp and xp objective
            # update theta_i
            batch = sp_rollouts[i]
            if 'trajedi' in self.conf.agent_name.lower():
                self.agent_pop[i].train(batch, ep_list=ep_list[i], agent_idx=i,
                                        ep_pad_mask=ep_pad_mask[i], time_pad_mask=time_pad_mask[i])
            else:
                self.agent_pop[i].train(batch)
        del sp_data, xp_data
        gc.collect()

    def evaluate(self):
        pop_size = self.conf.pop_size
        sp_data, xp_data = self._collect(eval_mode=True)

        frames = [[None] * pop_size for _ in range(pop_size)]
        sp_metrics = sp_data['metrics']
        if xp_data is not None:
            xp_metrics = xp_data['metrics']
        payoff_matrix = np.zeros([self.conf.pop_size, self.conf.pop_size])
        for i,j in product(range(pop_size), range(pop_size)):
            if i==j:
                payoff_matrix[i,j] = list(sp_metrics[i].avg_ret.values())[0]
                frames[i][j] = sp_data['frames'][i]
            else:
                if xp_data is not None:
                    payoff_matrix[i,j] = np.mean(list(xp_metrics[i*pop_size+j - int(j>i) - i].avg_ret.values())[0])
                    frames[i][j] = xp_data['frames'][i*pop_size+j - int(j>i) - i]

        logger.info(pblock(payoff_matrix, 'Payoff matrix...'))
        if self.conf.render:
            self._log_gif(frames)
        del frames
        del sp_data['frames']   
        if xp_data is not None:
            del xp_data['frames']
        data = Dotdict({'sp_data':sp_data, 'xp_data':xp_data})
        os.makedirs(f'{self.conf.save_dir}/it_{self.iter:05d}', exist_ok=True)
        pickle.dump(data, open(f'{self.conf.save_dir}/it_{self.iter:05d}/data.pkl','wb'))
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
            agent_tar['agent_state'] = agent.get_state()
            agent_tar['config'] = self.conf
            agent_tar['agent_no'] = i
            # each worker has its own mean, std stats
            # save only sp env
            agent_tar['env_stats'] = env_stat
            agent_tar['env_conf'] = env_conf
            agent_tar['eval_env_conf'] = eval_env_conf
            torch.save(agent_tar, f'{path}/agent_{i}.tar')

        env_stats_sp = [env_tar['env_stats'][i] for i in range(self.conf.pop_size)]
        p_save = partial(_save, env_conf=self.env_conf, eval_env_conf=self.eval_env_conf)
        for (agent, env_stat, i) in zip(self.agent_pop, env_stats_sp, range(len(self.agent_pop))):
            p_save(agent, env_stat, i)
        torch.save({'agent_pop': self.agent_pop,
                    'iter': self.iter}, 
                    f'{path}/trainer_checkpoint.pt')
        gc.collect()
