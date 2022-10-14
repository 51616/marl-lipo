from collections import defaultdict

from coop_marl.utils import Dotdict, chop_into_episodes # , get_logger

def get_avg_metrics(metrics):
    # assume a list of dotdicts of structure {player_name: dotdict}
    # each corresponds to one agent 
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

    return Dotdict(out)

def get_info(episodes):
    out = Dotdict()
    ret = defaultdict(int)
    n_ep = defaultdict(int)
    n_ts = defaultdict(int)
    players = list(episodes[0].inp.data.keys())

    for ep in episodes:
        for p in players:
            dones = getattr(ep.outcome.done, p)
            if dones[-1]:
                rews = ep.outcome.reward[p]
                if 'reward_unnorm' in ep.outcome[p]:
                    rews = ep.outcome[p].reward_unnorm
                ret[p] += sum(rews)
                n_ts[p] += rews.shape[0]
                n_ep[p] += 1
                # overcooked log complete dishes

    for p in players:
        out[p] = Dotdict()
        out[p]['avg_ret'] = ret[p]/n_ep[p]
        out[p]['avg_rew_per_ts'] = ret[p]/n_ts[p]
        out[p]['avg_ep_len'] = n_ts[p]/n_ep[p]
    return out

def get_traj_info(traj):
    episodes = chop_into_episodes(traj)
    infos = get_info(episodes)
    return infos