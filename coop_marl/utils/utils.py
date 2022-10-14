from collections import defaultdict
from functools import wraps
import inspect
import sys
import select
import os
import random
import yaml
from copy import deepcopy


import torch
import numpy as np
from PIL import Image
from pygifsicle import optimize
from yamlinclude import YamlIncludeConstructor

from coop_marl.utils import Arrdict, arrdict

'''taken from: https://stackoverflow.com/questions/1389180/automatically-initialize-instance-variables'''
def auto_assign(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        if defaults is not None:
            for i in range(len(defaults)):
                index = -(i + 1)
                if not hasattr(self, names[index]):
                    setattr(self, names[index], defaults[index])

        func(self, *args, **kargs)

    return wrapper

def reverse_dict(d):
    '''
    Reverse a dict. Returns a reversed dict with its values as lists
    '''
    out = defaultdict(list)
    for k,v in d.items():
        out[v].append(k)
    return out

def update_existing_keys(target_dict, source_dict):
    unused_param = {}
    for k,v in source_dict.items():
        if k in target_dict:
            if isinstance(target_dict[k],dict) and isinstance(v,dict):
                target_dict[k].update(v)
            else:
                target_dict[k] = v
        else:
            unused_param[k] = v
    return target_dict, unused_param

def merge_dict(d):
    assert len(d)>0
    if len(d)==1:
        return d[0]

    if len(d[0])==0:
        return merge_dict(d[1:])

    out = deepcopy(d[0])
    d1 = d[1]
    for k,v in d1.items():
        if k in out:
            if isinstance(v, dict):
                out[k] = merge_dict([out[k], v])
            elif isinstance(out[k], list):
                if isinstance(v, list):
                    out[k].extend(v)
                else:
                    out[k].append(v)
            elif isinstance(v, list):
                out[k] = [out[k]] + v
            else:
                out[k] = [out[k], v]
        else:
            out[k] = v
    return merge_dict([out] + d[2:])



class TimeoutExpired(Exception):
    pass

def input_with_timeout(prompt, timeout=10):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [],[], timeout)
    if ready:
        return sys.stdin.readline().rstrip('\n') # expect stdin to be line-buffered
    raise TimeoutExpired

def wait():
    try:
        t = 30
        input_with_timeout(f'Press Enter (or wait {t} seconds) to continue...', timeout=t)
    except TimeoutExpired:
        print('Input timed out, executing the next command.')

def save_gif(imgs, path, fps=30, size=None):
    if (imgs is None) or len(imgs)==0:
        return
    folder = path.split('/')[:-1]
    if len(folder) > 0:
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
    if imgs[0].shape[-1]==1:
        imgs = np.array(imgs)
        imgs = np.tile(imgs,(1,1,1,3))
    imgs = [Image.fromarray(img) for img in imgs]
    if size is not None:
        imgs = [i.resize(size) for i in imgs]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=1000/fps, loop=0)
    optimize(path)

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def create_ph_list(*args, **kwargs):
    return

def load_yaml(dir):
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
    with open(dir) as f:
        out = yaml.load(f, Loader=yaml.FullLoader)
    return out

def flatten_traj(traj):
    # remove player keys
    out = Arrdict()
    for p in traj.inp.data:
        print(p)
        batch = getattr(traj, p) # remove player keys from traj
        out = arrdict.merge_and_cat([out, batch])
    return out

def safe_log(val, replace_val=-50):
    log_val = torch.log(val + 1e-8)
    replace_bool = torch.isnan(log_val) + torch.isinf(log_val)
    return torch.where(replace_bool, replace_val * torch.ones_like(log_val), log_val)

if __name__ == '__main__':
    a = {'a':1,'b':2,'c':1}
    d = reverse_dict(a)
    print(d)
