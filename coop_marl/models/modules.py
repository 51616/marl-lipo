import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from coop_marl.utils.nn import ortho_layer_init

class FCLayers(nn.Module):
    def __init__(self, input_dim, hidden_size, num_hidden, output_size,
                 activation_fn=nn.ELU, base_std=np.sqrt(2), head_std=0.01,
                last_linear=True, layer_init_fn=ortho_layer_init):
        assert num_hidden > 0
        super().__init__()
        layer_construct = nn.Linear

        pre_hidden = [layer_init_fn(layer_construct(input_dim, hidden_size), base_std),
                                activation_fn()
                    ]
        base = pre_hidden
        if num_hidden>1:
            hiddens = []
            for i in range(num_hidden-1):
                hiddens.append(layer_init_fn(layer_construct(hidden_size, hidden_size), base_std))
                hiddens.append(activation_fn())
            base.extend(hiddens)
        out = [layer_init_fn(layer_construct(hidden_size,output_size), head_std)]
        if not last_linear:
            out.append(activation_fn())
        self.layers = nn.Sequential(*base, *out)

    def forward(self, x):
        return self.layers(x)

# taken from: https://github.com/keynans/HypeRL/blob/main/PEARL/torch/sac/hyper_network.py
class HyperHead(nn.Module):
    def __init__(self, base_inp_dim, meta_hidden_dim, output_size, stddev=0.05):
        super().__init__()
        self.output_size = output_size
        self.base_inp_dim = base_inp_dim
        self.w = nn.Linear(meta_hidden_dim, base_inp_dim * output_size)
        self.b = nn.Linear(meta_hidden_dim, output_size)
        self.init_layers(stddev)

    def forward(self, x):
        w = self.w(x).view(-1, self.output_size, self.base_inp_dim)
        b = self.b(x).view(-1, self.output_size, 1)
        return w, b

    def init_layers(self,stddev):
        ortho_layer_init(self.w, stddev)
        ortho_layer_init(self.b, stddev)

class ResBlock(nn.Module):

    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
                            nn.ELU(),
                            ortho_layer_init(nn.Linear(in_size, out_size)),
                            nn.ELU(),
                            ortho_layer_init(nn.Linear(out_size, out_size)),
                            )

    def forward(self, x):
        h = self.layers(x)
        return x + h

class MetaResNet(nn.Module):
    def __init__(self, meta_dim, hidden_size):
        super(MetaResNet, self).__init__()

        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(meta_dim, hidden_size),
            ResBlock(hidden_size, hidden_size),
            nn.ELU()
        )

        self.init_layers()

    def forward(self, meta_v):
        return self.layers(meta_v)

    def init_layers(self):
        for module in self.layers.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                ortho_layer_init(module)

class HyperNet(nn.Module):
    def __init__(self, base_inp_dim, meta_inp_dim, hidden_size, num_heads=1, head_sizes=[256], head_std=[0.05]):
        super().__init__()
        assert isinstance(head_sizes, list)
        assert isinstance(head_std, list)
        self.encoder = MetaResNet(meta_inp_dim, hidden_size)
        self.heads = nn.ModuleList()
        self.heads.append(HyperHead(base_inp_dim, hidden_size, head_sizes[0], stddev=head_std[0]))
        for i in range(1,num_heads):
            self.heads.append(HyperHead(head_sizes[i-1], hidden_size, head_sizes[i], stddev=head_std[i]))

    def produce_wb(self, z):
        if len(z.shape)==3:
            z = z.view(-1,z.shape[2])
        meta_h = self.encoder(z)
        wb = []
        for head in self.heads:
            w,b = head(meta_h)
            wb.append((w,b))
        return wb

    def forward(self, base_inp, meta_inp):
        wb = self.produce_wb(meta_inp)
        # base_inp for ff net -> [bs, feature] -> [bs, feature ,1]
        # base_inp for rnn -> [bs, ts, feature] -> [bs * ts, feature, 1]
        if len(base_inp.shape)==2:
            # ff net
            h = base_inp.unsqueeze(-1)
        elif len(base_inp.shape)==3:
            # rnn
            h = base_inp.reshape(base_inp.shape[0]*base_inp.shape[1],base_inp.shape[2],1)
        
        for w,b in wb[:-1]:
            h = F.elu(torch.bmm(w, h) + b)
        w,b = wb[-1]
        out = torch.bmm(w,h) + b
        if len(base_inp.shape)==3:
            out = out.view(*base_inp.shape[:2],-1)
            return out, wb
        return out.squeeze(-1), wb
