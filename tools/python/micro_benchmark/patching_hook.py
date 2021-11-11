import torch
from torch import nn
import inspect as ins
import pickle


WRAPPING_PREFIX = 'Wrap_'


def isfunc(mod, f):
    assert hasattr(mod, f)
    attr = getattr(mod, f)

    if len(f) >= 2:
        if f[0] == "_": return False # Temporarily exclude functions starting with '_'
        # if f[:2] == '__' and f[-2:] == '__': return False

    # Ignore functions to this list if they cause recursion
    ignore = ['size', 'tolist', 'dim', 'is_storage', 'item']
    # Also ignore the following functions related to tensor initialization
    ignore += ['zero_', 'uniform_', 'normal_', 'fill_']
    # and more
    ignore += ['copy_', 'numel', 'set_', 'has_names', 'index_select', 'cuda', 'contiguous', 'detach',
               'view_as', 'is_floating_point', 'float', 'half', 'to']
    if f in ignore:
        return False

    ignore_patterns = ['storage', 'stride', 'has_torch_function', 'new', 'as']
    if any([s in f for s in ignore_patterns]):
        return False

    return ins.ismethod(attr) or ins.isfunction(attr) or ins.ismethoddescriptor(attr) or ins.isbuiltin(attr)

def add_wrapper(mod, name):
    assert isfunc(mod, name)
    try:
        func = getattr(mod, name)
        def forward(self, *args, **kwargs):
            return func(*args, **kwargs)
        wrapping_class_name = WRAPPING_PREFIX + func.__name__
        def wrapper_func(*args, **kwargs):
            Wrap = type(wrapping_class_name, (nn.Module,), dict(forward=forward))
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


def hook(module, inputs, export_file='/tmp/picklefile', print_info=True):
    module_str = repr(module).rstrip('()')
    if module_str.startswith(WRAPPING_PREFIX):
        module_name = module_str[len(WRAPPING_PREFIX):]
        if print_info: print('module:', module, sep=' ')
        values = []
        save_dict = True
        for inp in inputs:
            if torch.is_tensor(inp):
                if inp.shape == torch.Size([]): # Treat tensor with no size as a number
                    key = 'numeric'
                    value = inp.item()
                else:
                    key = 'tensor'
                    value = inp.shape
            else:
                value = inp
                inp_type = type(inp)
                if inp_type in (tuple, list, str, bool):
                    key = inp_type.__name__
                elif inp_type in (int, float, complex):
                    key = 'numeric'
                elif inp is None:
                    key = 'none'
                else:
                    # Skip unknown data type case
                    print(f'In module {module_name}, input type {inp_type.__name__} not supported.')
                    save_dict = False
                    break
            values.append([key, value])
            # if print_info: print(key, value)
        if save_dict:
            dict_to_save = {'op': module_name, 'inputs': values}
            with open(export_file, 'ab') as f:
                pickle.dump(dict_to_save, f)

def load_pickle(pickle_file='/tmp/picklefile'):
    '''
    Loads input info.

    Returns: a list of input info dicts.
    '''
    input_info = {}
    with open(pickle_file, 'rb') as f:
        while True:
            try:
                x = pickle.load(f)
                op = x['op']
                info = x['inputs']
                if op in input_info:
                    input_info[op].append(info)
                else:
                    input_info[op] = [info]
            except EOFError:
                break

    print('Operator list:', list(input_info.keys()))
    return input_info
