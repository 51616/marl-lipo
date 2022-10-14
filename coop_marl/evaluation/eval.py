import os
import warnings
from itertools import product
from glob import glob

import numpy as np
import torch
import ray

from coop_marl.worker import RolloutWorker
from coop_marl.agents import Agent, registered_agents
from coop_marl.utils import Arrdict, arrdict, load_yaml, bcolors

def get_last_iter_dir(path, complete_only=True):
    conf = load_yaml(f'{path}/conf.yaml')
    n_iter = conf['n_iter']
    out = f'{path}/it_{n_iter:05d}'
    if not os.path.isdir(out):
        if complete_only:
            raise FileNotFoundError(f'{bcolors.FAIL}This run haven\'t completed training yet (Expect {n_iter} iterations).{bcolors.ENDC}')
        last_it = sorted(glob(f'{path}/it_*/'))[-1]
        warnings.warn(f'{bcolors.WARNING}This run haven\'t completed training yet (Expect {n_iter} iterations). ' +  
                      f'Using last training iteration instead. ({last_it}){bcolors.ENDC}')
        out = last_it
    return out

def load_agent(agent_conf: dict) -> Agent:
    agent_cls = registered_agents[agent_conf['agent_name']]
    agent = agent_cls(agent_conf)
    return agent

def load_config(path: str) -> dict:
    agent_tar = torch.load(path)
    agent_conf = agent_tar['config']
    env_conf = agent_tar['env_conf']
    eval_env_conf = agent_tar['eval_env_conf']
    return dict(agent_conf=agent_conf, env_conf=env_conf, eval_env_conf=eval_env_conf)

def load_state(path: str) -> dict:
    agent_tar = torch.load(path)
    return agent_tar['agent_state']

def load_env_stats(path: str) -> dict:
    agent_tar = torch.load(path)
    return agent_tar['env_stats']

def unpack(res: list) -> tuple:
    rollouts, infos, frames, metrics = [[None] * len(res) for _ in range(4)]
    for i, r in enumerate(res):
        rollouts[i], infos[i], frames[i], metrics[i] = r['rollouts'], r['infos'], r['frames'], r['metrics']
    return rollouts, infos, frames, metrics

def flatten_rollout(r):
    out = Arrdict()
    for p in r.inp.data:
        batch = getattr(r, p) # remove player keys from traj
        out = arrdict.merge_and_cat([out, batch])
    return out


def update_workers(workers: list, pairs: list, config_list: list,
                   agent_state_list: list, env_stats_list: list) -> None:
    for w, (i,j) in zip(workers, pairs):
        w.set_home_id.remote(i)
        w.set_away_id.remote(j)
        w.set_sp.remote(i==j) # needed to get home, away rollouts

        w.update.remote(home_conf=config_list[i]['agent_conf'],
                             away_conf=config_list[j]['agent_conf'],
                             home_state=agent_state_list[i],
                             away_state=agent_state_list[j],
                             home_env_stats=env_stats_list[i],
                             away_env_stats=env_stats_list[j])

def check_config(config_list):
    for conf in config_list:
        assert conf['agent_conf']['env_wrappers'] == config_list[0]['agent_conf']['env_wrappers'],\
        f'All agents should have the same env wrappers. Found {conf["agent_conf"]["env_wrappers"]} and \
        {config_list[0]["agent_conf"]["env_wrappers"]}'

def load_all(tar_list: list):
    config_list = [load_config(p) for p in tar_list]
    check_config(config_list)
    agent_list = [load_agent(conf['agent_conf']) for conf in config_list]
    agent_state_list = [load_state(p) for p in tar_list]
    env_stats_list = [load_env_stats(p) for p in tar_list]
    return config_list, agent_list, agent_state_list, env_stats_list

def run(n_eval_ep: int,
        render: bool,
        config_list: list,
        agent_state_list:list,
        env_stats_list: list,
        n_workers: int,
        all_pairs: list):

    n_pairs = len(all_pairs)
    # use config of the first agent to create the workers
    # if agents have different view requirement, then uses the one that needs additional view to init worker
    workers = [RolloutWorker.remote(config_list[0]['agent_conf'], 
                                        config_list[0]['env_conf'],
                                    config_list[0]['eval_env_conf']) for _ in range(n_workers)]
    rollouts, infos, frames, metrics = [[] for _ in range(4)]

    for i in range(0, n_pairs, n_workers):
        start = i
        end = i+n_workers
        pairs = all_pairs[start:end]
        print(f'Pairs: {pairs}')
        update_workers(workers, pairs, config_list, agent_state_list, env_stats_list)

        # collect n_workers data from pairs
        res = ray.get([w.rollout.remote(n_eval_ep, render=render, eval_mode=True) for w in workers])
        data = unpack(res)
        for l,d in zip([rollouts, infos, frames, metrics], data):
            l.extend(d)
    return rollouts, infos, frames, metrics

def get_tars(path: str) -> list:
    return glob(f'{path}/agent_*.tar')

def evaluate_meta(meta_tar: str, partner_tars: list, n_eval_ep: int, render: bool, n_workers: int = 1) -> tuple:
    paths = [meta_tar] + partner_tars
    config_list, agent_list, agent_state_list, env_stats_list = load_all(paths)
    
    n_partners = len(partner_tars)
    p = [(0,i) for i in range(1,n_partners+1)] + [(i,0) for i in range(1,n_partners+1)]
    n_workers = min(len(p), n_workers)
    
    return run(n_eval_ep, render, config_list, agent_state_list, env_stats_list, n_workers, all_pairs=p)

def evaluate(tar_list: list, n_eval_ep: int, render: bool, xp: bool, n_workers: int = 1) -> tuple:   
    config_list, agent_list, agent_state_list, env_stats_list = load_all(tar_list)

    n_agents = len(agent_list)
    # each worker correspond to a pair of agents
    p = list(product(range(n_agents), range(n_agents))) if xp else list(zip(range(n_agents), range(n_agents)))
    n_workers = min(len(p), n_workers)

    return run(n_eval_ep, render, config_list, agent_state_list, env_stats_list, n_workers, all_pairs=p)


def evaluate_vary_z(paths: list, n_eval_ep: int, render: bool, n_workers: int = 1, share_z: bool = False):
    # use one policy
    assert len(paths)==1
    path = paths[0]
    conf = load_config(path)
    agent_state = load_state(path)
    env_stats = load_env_stats(path)

    z_dim = conf['agent_conf']['z_dim']
    # assume discrete z

    print(f'z dim: {z_dim}')
    if n_workers > z_dim:
        n_workers = z_dim
    print(f'N workers: {n_workers}')

    workers = [RolloutWorker.remote(conf['agent_conf'], 
                                    conf['env_conf'],
                                    conf['eval_env_conf']) for _ in range(n_workers)]

    for w in workers:
        w.update.remote(home_conf=conf['agent_conf'],
                             away_conf=conf['agent_conf'],
                             home_state=agent_state,
                             away_state=agent_state,
                             home_env_stats=env_stats,
                             away_env_stats=env_stats)
        w.set_sp.remote(True)

    rollouts, infos, frames, metrics = [[] for _ in range(4)]
    # fixed z for each worker
    # get |z| rollouts each with unique z
    eye = np.eye(z_dim)
    if share_z:
        z_pairs = list(zip(eye, eye))
    else:
        z_pairs = list(product(eye,eye))
    for i in range(0, len(z_pairs), n_workers):
        start = i
        end = i+n_workers
        z_list = z_pairs[start:end]
        print(z_list)
        [w.set_z.remote(z, eval_mode=True) for w,z in zip(workers, z_list)]
        res = ray.get([w.rollout.remote(n_eval_ep, render=render, eval_mode=True) for w in workers])
        data = unpack(res)
        for l,d in zip([rollouts, infos, frames, metrics], data):
            l.extend(d)
    return rollouts, infos, frames, metrics
