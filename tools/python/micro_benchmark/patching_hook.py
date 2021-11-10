import torch
from torch import nn
import inspect as ins
import pickle


def isfunc(mod, f):
    assert hasattr(mod, f)
    attr = getattr(mod, f)

    if len(f) >= 2:
        if f[0] == "_": return False # Temporarily exclude functions starting with '_'
        # if f[:2] == '__' and f[-2:] == '__': return False

    # Add functions to this list if they cause recursion
    ignore = ['size', 'tolist', 'dim', 'is_storage', 'item']
    if f in ignore:
        return False

    # Temporarily exclude other functions/methods from `torch.Tensor`
    if mod == torch.Tensor and f not in ['repeat', 'view', 'cumsum', 'ne']:
        return False

    return ins.ismethod(attr) or ins.isfunction(attr) or ins.ismethoddescriptor(attr) or ins.isbuiltin(attr)

def add_wrapper(mod, name):
    assert isfunc(mod, name)
    try:
        func = getattr(mod, name)
        def forward(self, *args, **kwargs):
            return func(*args, **kwargs)
        def wrapper_func(*args, **kwargs):
            Wrap = type(func.__name__, (nn.Module,), dict(forward=forward))
            return Wrap()(*args, **kwargs)
        setattr(mod, name, wrapper_func)
    except:
        pass

def patching(mod):
    for f in dir(mod):
        if isfunc(mod, f):
            add_wrapper(mod, f)

def init():
    # patches torch.Tensor/torch.nn.functional functions
    for mod in [torch.Tensor, torch.nn.functional]:
        patching(mod)

def hook(module, inputs, export_file='/tmp/picklefile'):
    module_str = repr(module)
    if not module_str.startswith('_') and module_str.lower() == module_str and module_str != 'wrapper_func()':
        print('module:', module)
        values = []
        for inp in inputs:
            if torch.is_tensor(inp):
                if inp.shape == torch.Size([]):
                    key = 'numeric'
                    value = inp.item()
                else:
                    key = 'tensor'
                    value = inp.shape
            else:
                value = inp
                if isinstance(inp, tuple):
                    key = 'tuple'
                elif isinstance(inp, list):
                    key = 'list'
                else:
                    key = 'numeric'
            values.append([key, value])
            print(key, value)
        module_name = module_str.rstrip('()')
        dict_to_save = {'op': module_name, 'inputs': values}
        with open(export_file, 'ab') as f:
            pickle.dump(dict_to_save, f)
