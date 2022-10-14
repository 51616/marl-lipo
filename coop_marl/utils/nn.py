import numpy as np
import torch

FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38

# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
def ortho_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def k_uniform_init(layer, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    torch.nn.init.kaiming_uniform_(layer.weight, a, mode, nonlinearity)
    torch.nn.init.constant_(layer.bias, 0)
    return layer

def dict_to_tensor(obs_dict, device, axis=0, dtype=torch.float):
    # takes a dict of obs (e.g. player->obs) and returns tensor of obs as [N_player, obs_dim]
    return torch.stack([torch.as_tensor(o, dtype=dtype, device=device) for o in obs_dict.values()], axis=axis)

def dict_to_np(obs_dict,*, axis=0, dtype=np.float32):
    # takes a dict of obs (e.g. player->obs) and returns tensor of obs as [N_player, obs_dim]
    return np.stack([np.array(o, dtype=dtype) for o in obs_dict.values()], axis=axis)