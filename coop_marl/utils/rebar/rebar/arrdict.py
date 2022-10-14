from multiprocessing import Value
import numpy as np
from copy import deepcopy
from functools import partialmethod
from . import dotdict
try:
    import torch
    TORCH = True
except ModuleNotFoundError:
    TORCH = False

SCREEN_WIDTH = 119
SCREEN_HEIGHT = 200

def _arrdict_factory():
    # This is done with a factory because I am a lazy man and I didn't fancy defining all the binary ops on 
    # the arrdict manually.

    class _arrdict_base(dotdict.dotdict):
        """An arrdict is an :class:`~rebar.dotdict.dotdict` with extra support for array and tensor values.

        arrdicts have a lot of unusual but extremely useful behaviours, which are documented in :ref:`the dotdicts
        and arrdicts concept section <dotdicts>` .
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __str__(self):
            return treestr(self)

        def __getitem__(self, x):
            if isinstance(x, str):
                return super().__getitem__(x)
            else:
                return type(self)({k: v[x] for k, v in self.items()})

        def __setitem__(self, x, y):
            # Valid keys to stick in an arrdict are strings and tuples of strings.
            # Anything else could plausibly be a tensor index.
            if (isinstance(x, str) or 
                    (isinstance(x, tuple) and all(isinstance(xx, str) for xx in x))):
                super().__setitem__(x, y)
            elif isinstance(y, type(self)):
                for k in self:
                    self[k][x] = y[k]
            else:
                raise ValueError('Setting items must be done with a string key or by passing an arrdict')

        def __setattr__(self, key, value):
            raise ValueError('Setting by attribute is not allowed; set by key instead')

        def __binary_op__(self, name, rhs):
            if isinstance(rhs, dict):
                return self.starmap(name, rhs)
            else:
                return super().__getattr__(name)(rhs)

    # Add binary methods
    # https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    binaries = [
        'lt', 'le', 'eq', 'ne', 'ge', 'gt', 
        'add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and', 'or', 'xor',
        'radd', 'rsub', 'rmul', 'rmatmul', 'rtruediv', 'rfloordiv', 'rmod', 'rdivmod', 'rpow', 'rand', 'lshift', 'rshift', 'ror', 'rxor']
    methods = {f'__{name}__': partialmethod(_arrdict_base.__binary_op__, f'__{name}__') for name in binaries}

    methods['__doc__'] = _arrdict_base.__doc__

    return type('arrdict', (_arrdict_base,), methods)

def treestr(t):
    """Stringifies a tree structure. These turn up all over the place in my code, so it's worth factoring out"""
    key_length = max(map(len, map(str, t.keys()))) if t.keys() else 0
    max_spaces = 4 + key_length
    val_length = SCREEN_WIDTH - max_spaces
    
    d = {}
    for k, v in t.items():
        if isinstance(v, dotdict.dotdict):
            d[k] = str(v)
        elif isinstance(v, (list, set, dict)):
            d[k] = f'{type(v).__name__}({len(v)},)'
        elif hasattr(v, 'shape') and hasattr(v, 'dtype'):                    
            d[k] = f'{type(v).__name__}({tuple(v.shape)}, {v.dtype})'
        elif hasattr(v, 'shape'):
            d[k] = f'{type(v).__name__}({tuple(v.shape)})'
        else:
            lines = str(v).splitlines()
            if (len(lines) > 1) or (len(lines[0]) > val_length):
                d[k] = lines[0][:val_length] + ' ...'
            else:
                d[k] = lines[0]

    s = [f'{type(t).__name__}:']
    for k, v in d.items():
        lines = v.splitlines() or ['']
        s.append(str(k) + ' '*(max_spaces - len(str(k))) + lines[0])
        for l in lines[1:]:
            s.append(' '*max_spaces + l)
        if len(s) >= SCREEN_HEIGHT-1:
            s.append('...')
            break

    return '\n'.join(s)

arrdict = _arrdict_factory()

@dotdict.mapping
def torchify(a):
    """Converts an array or a dict of numpy arrays to CPU tensors.

    If you'd like CUDA tensors, follow the tensor-ification ``.cuda()`` ; the attribute delegation
    built into :class:`~rebar.dotdict.dotdict` s will do the rest.
    
    Floats get mapped to 32-bit PyTorch floats; ints get mapped to 32-bit PyTorch ints. This is usually what you want in 
    machine learning work.
    """
    if hasattr(a, 'torchify'):
        return a.torchify()

    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        dtype = torch.float
    elif np.issubdtype(a.dtype, np.integer):
        dtype = torch.int
    elif np.issubdtype(a.dtype, np.bool_):
        dtype = torch.bool
    else:
        raise ValueError(f'Can\'t handle {type(a)}')
    return torch.as_tensor(np.array(a), dtype=dtype)

@dotdict.mapping
def numpyify(tensors):
    """Converts an array or a dict of tensors to numpy arrays.
    """
    if isinstance(tensors, tuple):
        return tuple(numpyify(t) for t in tensors)
    if isinstance(tensors, torch.Tensor):
        return tensors.clone().detach().cpu().numpy()
    if hasattr(tensors, 'numpyify'):
        return tensors.numpyify()
    return tensors

def stack(x, *args, **kwargs):
    """Stacks a sequence of arrays, tensors or dicts thereof.  

    For example, 

    >>> d = arrdict(a=1, b=np.array([1, 2]))
    >>> stack([d, d, d])
    arrdict:
    a    ndarray((3,), int64)
    b    ndarray((3, 2), int64)

    Any ``*args`` or ``**kwargs`` will be forwarded to the ``np.stack`` or ``torch.stack`` call. 

    Python scalars are converted to numpy scalars, so - as in the example above - stacking floats will
    get you a 1D array.
    """
    if isinstance(x[0], dict):
        ks = x[0].keys()
        return x[0].__class__({k: stack([y[k] for y in x], *args, **kwargs) for k in ks})
    if TORCH and isinstance(x[0], torch.Tensor):
        return torch.stack(x, *args, **kwargs)
    if isinstance(x[0], np.ndarray):
        return np.stack(x, *args, **kwargs) 
    if np.isscalar(x[0]):
        return np.array(x, *args, **kwargs)
    raise ValueError(f'Can\'t stack {type(x[0])}')

def cat(x, *args, **kwargs):
    """Concatenates a sequence of arrays, tensors or dicts thereof.  

    For example, 

    >>> d = arrdict(a=1, b=np.array([1, 2]))
    >>> cat([d, d, d])
    arrdict:
    a    ndarray((3,), int64)
    b    ndarray((6,), int64)

    Any ``*args`` or ``**kwargs`` will be forwarded to the ``np.concatenate`` or ``torch.cat`` call. 

    Python scalars are converted to numpy scalars, so - as in the example above - concatenating floats will
    get you a 1D array. 
    """
    if isinstance(x[0], dict):
        ks = x[0].keys()
        return x[0].__class__({k: cat([y[k] for y in x], *args, **kwargs) for k in ks})
    if TORCH and isinstance(x[0], torch.Tensor):
        return torch.cat(x, *args, **kwargs)
    if isinstance(x[0], np.ndarray):
        return np.concatenate(x, *args, **kwargs) 
    if np.isscalar(x[0]):
        return np.array(x)
    raise ValueError(f'Can\'t cat {type(x[0])}')

@dotdict.mapping
def clone(t):
    if hasattr(t, 'clone'):
        return t.clone()
    if hasattr(t, 'copy'):
        return t.copy()
    return t

def create_empty_col(x,lib):    
    # in-place op
    if isinstance(x, dict):
        empty_col = x.map(lambda t: lib.empty([0]+list(t.shape[1:]), dtype=torch.float if lib is torch else np.float32))
    else:
        empty_col = lib.empty([0]+list(x.shape[1:]), dtype=torch.float if lib is torch else np.float32)
    return empty_col

def match_dict(source,target,lib):
    if not isinstance(source, dict):
        return
        
    for k in target.keys():
        if k not in source.keys():
            source[k] = create_empty_col(target[k], lib)
        else:
            match_dict(source[k],target[k],lib)

def merge_and_cat(inp, *args, **kwargs):
    lib = torch if isinstance(dotdict.leaves(inp)[0], torch.Tensor) else np
    # create a copy of x first
    x = [deepcopy(t) for t in inp]
    # put all keys into the first dict
    assert isinstance(x[0], dict)
    for d in x[1:]:
        match_dict(x[0],d,lib)
        # for k in d.keys():
        #     if k not in x[0]:
        #         x[0][k] = create_empty_col(d[k], lib)
        #         print(x[0])
        #     else:

    # copy the keys from the first dict
    for d in x[1:]:
        match_dict(d,x[0],lib)
        # for k in x[0].keys():
        #     if k not in d:
        #         d[k] = create_empty_col(x[0][k], lib)
    # for i,d in enumerate(x):
    #     print(i, d)
    return cat(x,*args, **kwargs)


@dotdict.mapping
def postpad(x,max_len,dim=0):
    if isinstance(x[0], dict):
        ks = x[0].keys()
        return x[0].__class__({k: postpad([y[k] for y in x], max_len, dim) for k in ks})
    if TORCH and isinstance(x, torch.Tensor):
        # https://discuss.pytorch.org/t/how-to-do-padding-based-on-lengths/24442
        out_dims = list(x.shape)
        pad_size = max_len - out_dims[dim]
        out_dims[dim] += pad_size
        out = x.data.new(*out_dims).fill_(0)
        length = x.size(dim)
        out.index_copy_(dim,torch.arange(length, device=x.device),x)
        return out

    if isinstance(x, np.ndarray):
        pad_size = max_len - list(x.shape)[dim]
        pad = (0,pad_size)
        pad_width = [(0,0) for i in range(len(x.shape))] 
        pad_width[dim] = pad
        padded_seq = np.pad(x, pad_width)
        return padded_seq

    raise ValueError(f'Can\'t pad {type(x[0])}')
    