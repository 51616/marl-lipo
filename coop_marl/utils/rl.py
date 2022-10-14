from functools import lru_cache

import numpy as np
import torch

from coop_marl.utils import Arrdict, dict_to_tensor, arrdict

def get_discount_coef(gamma, ep_len, device=None):
    t_prime = torch.arange(0,ep_len, device=device).unsqueeze(1)
    power = torch.abs(torch.arange(0,ep_len, device=device).unsqueeze(0).repeat(ep_len,1) - t_prime)
    return gamma ** power

def compute_jsd_metric(batch, agent_pop, kernel_gamma, use_br):
        pop_size = len(agent_pop)
        if use_br:
            pop_size = len(agent_pop[:-1])
        ep_list = []
        
        # compute del_(j,t)(tau) for all j,t and tau
        # for j, batch in enumerate(sp_rollouts):
        for ep in chop_into_episodes(batch):
            # create a new field traj.delta[i,t], traj.delta_hat[t]
            # traj.pi[i] and traj.pi_hat
            ep_len = ep.inp.data.obs.shape[0]
            ep['act_logprob'] = torch.empty([pop_size, ep_len])
            ep['delta'] = torch.empty([pop_size, ep_len])
            ep['delta_hat'] = torch.empty([ep_len])
            ep['pi'] = torch.empty([pop_size])
            ep['pi_hat'] = torch.empty(1)

            for i in range(pop_size):
                ep.act_logprob[i] = calc_action_logprob(agent_pop[i], ep).detach() # [ep_len,]
                ep.pi[i] = torch.sum(ep.act_logprob[i], axis=0).exp().detach()
                ep.delta[i,:] = local_action_kernel(ep.act_logprob[i], ep, kernel_gamma).detach()

            ep['delta_hat'] = torch.mean(ep.delta, axis=0)
            ep['pi_hat'] = torch.mean(ep.pi, axis=0)
            ep_list.append(ep)
        return ep_list

def local_action_kernel(a_logprob, batch, gamma):
    '''
    pi: log probablity of actions along the trajectory with shape [ep_len]
    '''
    ep_len = a_logprob.shape[0]
    assert a_logprob.shape == (ep_len,)
    d = get_discount_coef(gamma, ep_len, device=a_logprob.device) # [ep_len, ep_len]
    delta = torch.sum(d * a_logprob.unsqueeze(0), axis=1).exp() # [ep_len] (delta[t]) is delta_t
    return delta

def calc_action_logprob(agent, ep):
    v = list(ep.inp.data.values())[0]
    if isinstance(v, Arrdict):
        # unflatten v (player keys still exist)
        obs_batch = dict_to_tensor(ep.inp.data.obs, device=agent.device) # flatten obs 
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        # flatten ep
        obs_batch = torch.tensor(v, dtype=torch.float, device=agent.device)
    else:
        raise TypeError

    dist = agent.calc_action_dist(obs_batch)
    logprob = dist.log_prob(torch.tensor(ep.decision.action, device=agent.device))
    return logprob

@lru_cache(maxsize=128)
def get_discounted_coef(gamma, length):
    return gamma ** torch.arange(length)

def chop_into_episodes(traj):
    if len(traj.keys())==0:
        # empty Arrdict
        return []
    term_t = []
    num_ts = traj.outcome.done.shape[0]
    if isinstance(num_ts, dict):
        num_ts = max(list(num_ts.values()))
    # unflatten traj
    if isinstance(traj.outcome.done, dict):
        for p,d in traj.outcome.done.items():
            term_t.append(np.nonzero(d)[0])
        assert (np.array(term_t)==term_t[0]).all(), f'All agents must terminates at the same time, got {term_t}'
    # flatten traj
    elif isinstance(traj.outcome.done, torch.Tensor):
        term_t.append(np.nonzero(traj.outcome.done).squeeze())
    elif isinstance(traj.outcome.done, np.ndarray):
        term_t.append(np.nonzero(traj.outcome.done)[0])
    ts = term_t[0]
    i = 0
    eps = []
    for t in ts:
        eps.append(traj[i:t+1])
        i = t+1
        if t==ts[-1] and t>=num_ts:
            eps.append(traj[i:])
    return eps

def flatten_traj(traj):
    batch = Arrdict()
    for p in traj.inp.data:
        p_batch = getattr(traj, p) # remove player keys from traj
        batch = arrdict.merge_and_cat([batch, p_batch])
    return batch


def compute_return(traj, gamma, next_value):
    traj['outcome']['ret'] = np.zeros(traj.outcome.reward.shape[0], dtype=np.float32)
    ret = next_value
    rew = traj.outcome.reward
    for t in reversed(range(traj.outcome.reward.shape[0])):
        if traj.outcome.done[t]:
            ret = 0
        assert ret is not None, f'Please provided the next_value when the trajectory is not terminated'
        ret = rew[t] + gamma * ret
        traj['outcome']['ret'][t] = ret

def compute_gae(traj, gamma, lam, values, next_value):
    adv = np.zeros(traj.outcome.reward.shape[0], dtype=np.float32)
    next_gae = 0
    for t in reversed(range(traj.outcome.reward.shape[0])):
        non_term = not traj.outcome.done[t]
        if not non_term:
            next_gae = 0
            next_value = 0
        assert next_value is not None, f'Please provided the next_value when the trajectory is not terminated'
        delta = traj.outcome.reward[t] + non_term * gamma * next_value - values[t]
        adv[t] = next_gae = delta + gamma * lam * non_term * next_gae
        next_value = values[t]
    traj['outcome']['adv'] = adv
    traj['outcome']['ret'] = adv + values

def test_compute_gae():
    traj = Arrdict(outcome=Arrdict(reward=np.array([1,1,0,1]),done=np.array([0,0,0,1])))
    values = np.array([1,2,3,4])
    next_value = np.array(10)
    compute_gae(traj, 0.99, 0.9, values, next_value)
    print(traj.outcome.adv)
    print(values)
    print(traj.outcome.ret)
    assert abs(traj.outcome.adv-np.array([2.37535185, 0.443717, -1.713, -3])).sum()<0.00001

    traj = Arrdict(outcome=Arrdict(reward=np.array([1,1,0,1]),done=np.array([0,1,0,1])))
    values = np.array([1,2,3,4])
    next_value = np.array(10)
    compute_gae(traj, 0.99, 0.9, values, next_value)
    # print(traj.outcome.adv)
    assert abs(traj.outcome.adv-np.array([1.089,-1, -1.713, -3])).sum()<0.00001

    traj = Arrdict(outcome=Arrdict(reward=np.array([1,1,0,1]),done=np.array([0,1,0,0])))
    values = np.array([1,2,3,4])
    next_value = np.array(10)
    compute_gae(traj, 0.99, 0.9, values, next_value)
    # print(traj.outcome.adv)
    assert abs(traj.outcome.adv-np.array([1.089,-1, 7.1079, 6.9])).sum()<0.00001

    traj = Arrdict(outcome=Arrdict(reward=np.array([0]),done=np.array([1])))
    values = np.array([1])
    next_value = np.array(10)
    compute_gae(traj, 0.99, 0.9, values, next_value)
    # print(traj.outcome.adv)
    assert abs(traj.outcome.adv-np.array([-1])).sum()<0.00001

    traj = Arrdict(outcome=Arrdict(reward=np.array([0]),done=np.array([0])))
    values = np.array([1])
    next_value = np.array(10)
    compute_gae(traj, 0.99, 0.9, values, next_value)
    # print(traj.outcome.adv)
    assert abs(traj.outcome.adv-np.array([8.9])).sum()<0.00001
    print('TEST COMPUTE GAE PASSED!')


def test_compute_return():
    def forward_return(rewards, gamma):
            if rewards.shape[0]==1:
                return rewards[0]
            return rewards[0] + gamma*forward_return(rewards[1:],gamma)
    # test return calc
    traj = Arrdict(outcome=Arrdict(reward=np.array([1,1,0,1]),done=np.array([0,0,0,1])))
    compute_return(traj, 0.99, 0)
    print(traj.outcome.ret)
    assert (traj.outcome.ret-np.array([2.960299, 1.9801, 0.99, 1])).sum()<1e-5

    traj = Arrdict(outcome=Arrdict(reward=np.array([1,1,0,1]),done=np.array([0,0,0,0])))
    compute_return(traj, 0.99, 0)
    assert (traj.outcome.ret-np.array([2.960299, 1.9801, 0.99, 1])).sum()<1e-5
    # print(traj.outcome.ret)

    traj = Arrdict(outcome=Arrdict(reward=np.array([1]),done=np.array([0])))
    compute_return(traj, 0.99, 0)
    assert (traj.outcome.ret-np.array([1])).sum()<1e-5
    # print(traj.outcome.ret)

    traj = Arrdict(outcome=Arrdict(reward=np.array([0]),done=np.array([0])))
    compute_return(traj, 0.99, 0)
    assert (traj.outcome.ret-np.array([0])).sum()<1e-5
    # print(traj.outcome.ret)

    traj = Arrdict(outcome=Arrdict(reward=np.array([1,2,3,1,2,3]),done=np.array([0,0,1,0,0,0])))
    compute_return(traj, 0.5, 0)
    assert (traj.outcome.ret-np.array([2.75,3.5,3,2.75,3.5,3])).sum()<1e-5
    # print(traj.outcome.ret)

    rew_arr = np.array([1,2,3,1,2,3])
    gamma = 0.5
    traj = Arrdict(outcome=Arrdict(reward=rew_arr,done=np.array([0,0,0,0,0,0])))
    compute_return(traj, gamma, 0)
    returns = []
    for i in range(rew_arr.shape[0]):
        returns.append(forward_return(rew_arr[i:],gamma))
    assert (traj.outcome.ret-np.array(returns)).sum()<1e-5, f'{traj.outcome.ret} vs {returns}'
    # print(traj.outcome.ret)
    print('TEST COMPUTE RETURN PASSED!')

if __name__ == '__main__':
    test_compute_return()
    test_compute_gae()

    