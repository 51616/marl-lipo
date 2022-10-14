from collections import OrderedDict
from functools import wraps
from argparse import Namespace

SCREEN_WIDTH = 119
SCREEN_HEIGHT = 200

class dotdict(OrderedDict):
    """dotdicts are dictionaries with additional support for attribute (dot) access of their elements.
    dotdicts have a lot of unusual but extremely useful behaviours, which are documented in :ref:`the dotdicts
    and arrdicts concept section <dotdicts>` .

    """
    # __getattr__ = dict.get # dict.__getitem__ 
    # def __setattr__(self, k, v): return self.__setitem__(k, v)
    # def __delattr__(self, k): return self.__delitem__(k)

    # def __init__(self, *args, **kwargs):
    #     super().__init__(**kwargs)
    #     assert len(args) in [0,1], f'Only takes 1 positional argument'
    #     if len(args)==1:
    #         # assert isinstance(args[0], dict), f'The input must be a dict'
    #         self._recurse_assign(args[0])
    #     else:
    #         self._recurse_assign(None, **kwargs)
    
    # def _recurse_assign(self, *args, **kwargs):
    #     if isinstance(args[0],dict) or isinstance(args[0], Namespace):
    #         if isinstance(args[0], Namespace):
    #             args[0] = args[0].__dict__
    #         for key,item in args[0].items():
    #             if not isinstance(item,dotdict):
    #                 item = to_dot_dict(item)
    #             self.__setattr__(key,item)
    #     else:
    #         for key,item in kwargs.items():
    #             if not isinstance(item,dotdict):
    #                 item = to_dot_dict(item)
    #             self.__setattr__(key,item)
    
    def __dir__(self):
        return sorted(set(super().__dir__() + list(self.keys())))

    def __setattr__(self, key, value):
        raise ValueError('Setting by attribute is not allowed; set by key instead')

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            try:
                return type(self)([(k, getattr(v, key)) for k, v in self.items()])
            except AttributeError:
                raise AttributeError(f"There is no member called '{key}' and one of the leaves has no attribute '{key}'") from None

    def __call__(self, *args, **kwargs):
        return type(self)([(k, v(*args, **kwargs)) for k, v in self.items()])

    def __str__(self):
        return treestr(self)
    
    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    # def __getstate__(self): return self.__dict__
    # def __setstate__(self, d): self.__dict__.update(d)

    # def update(self, data=None, **kwargs):
    #     self._recurse_assign(data, **kwargs)
    
    def copy(self):
        """Shallow-copy the dotdict"""
        return type(self)(super().copy()) 
    
    def pipe(self, f, *args, **kwargs):
        """Returns ``f(self, *args, **kwargs)`` . 

        >>> d = dotdict(a=1, b=2)
        >>> d.pipe(list)
        ['a', 'b']

        Useful for method-chaining."""
        return f(self, *args, **kwargs)

    def map(self, f, *args, **kwargs):
        """Applies ``f`` to the values of the dotdict, returning a matching dotdict of the results.
        ``*args`` and  ``**kwargs`` are passed as extra arguments to each call.

        >>> d = dotdict(a=1, b=2)
        >>> d.map(int.__add__, 10)
        dotdict:
        a    11
        b    12

        Useful for method-chaining. Works equally well on trees of dotdicts.
        
        See :func:`mapping` for a functional version of this method."""
        return mapping(f)(self, *args, **kwargs)

    def starmap(self, f, *args, **kwargs):
        """Applies ``f`` to the values of the dotdicts one key at a time, returning a matching dotdict of the results.

        >>> d = dotdict(a=1, b=2)
        >>> d.starmap(int.__add__, d)
        dotdict:
        a    2
        b    4

        Useful for method-chaining. Works equally well on trees of dotdicts.
        
        See :func:`starmapping` for a functional version of this method."""
        return starmapping(f)(self, *args, **kwargs)

def to_dot_dict(data, **kwargs):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                data[k] = dotdict(v)
                # to_dot_dict(data[k])
            elif isinstance(v, list):
                data[k] = [to_dot_dict(i) for i in v]
    elif isinstance(data, list):
        return [to_dot_dict(i) for i in data]
    else:
        return data
    return dotdict(data)

def treestr(t):
    """Stringifies a tree structure. These turn up all over the place in my code, so it's worth factoring out"""
    key_length = max(map(len, map(str, t.keys()))) if t.keys() else 0
    max_spaces = 4 + key_length
    val_length = SCREEN_WIDTH - max_spaces
    
    d = {}
    for k, v in t.items():
        if isinstance(v, dotdict):
            d[k] = str(v)
        # elif isinstance(v, (list, set, dict)):
        #     d[k] = f'{type(v).__name__}({len(v)},)'
        # elif hasattr(v, 'shape') and hasattr(v, 'dtype'):                    
        #     d[k] = f'{type(v).__name__}({tuple(v.shape)}, {v.dtype})'
        # elif hasattr(v, 'shape'):
        #     d[k] = f'{type(v).__name__}({tuple(v.shape)})'
        else:
            lines = str(v).splitlines() or ['']
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

def mapping(f):
    """Wraps ``f`` so that when called on a dotdict, ``f`` instead gets called on the dotdict's values
    and a dotdict of the results is returned. Extra ``*args`` and ``**kwargs`` passed to the wrapper are
    passed as extra arguments to ``f`` .

    >>> d = dotdict(a=1, b=2)
    >>> m = mapping(int.__add__)
    >>> m(d, 10)
    dotdict:
    a    11
    b    12
    
    Works equally well on trees of dotdicts, where ``f`` will be applied to the leaves of the tree.

    Can be used as a decorator.

    See :func:`dotdict.map` for an object-oriented version of this function.
    """

    @wraps(f)
    def g(x, *args, **kwargs):
        if isinstance(x, dict):
            return type(x)([(k, g(v, *args, **kwargs)) for k, v in x.items()])
        if isinstance(f, str):
            return getattr(x, f)(*args, **kwargs)
        return f(x, *args, **kwargs)
    return g

def starmapping(f):
    """Wraps ``f`` so that when called on a sequence of dotdicts, ``f`` instead gets called on the dotdict's values
    and a dotdict of the results is returned.

    >>> d = dotdict(a=1, b=2)
    >>> m = starmapping(int.__add__)
    >>> m(d, d)
    dotdict:
    a    2
    b    4
    
    Works equally well on trees of dotdicts, where ``f`` will be applied to the leaves of the trees.

    Can be used as a decorator.

    See :func:`dotdict.starmap` for an object-oriented version of this function.
    """
    @wraps(f)
    def g(x, *args, **kwargs):
        if isinstance(x, dict):
            return type(x)([(k, g(x[k], *(a[k] for a in args))) for k in x])
        if isinstance(f, str):
            return getattr(x, f)(*args)
        else:
            return f(x, *args)
    return g

def leaves(t):
    """Returns the leaves of a tree of dotdicts as a list"""
    if isinstance(t, dict):
        return [l for v in t.values() for l in leaves(v)]
    return [t]