import torch
from torch import nn
import inspect as ins
import pickle
from typing import Union, List, Tuple
import pprint


WRAPPING_PREFIX = 'Wrap_'
modules_to_wrap = (torch.Tensor, nn.functional)
supported_types = (tuple, list, str, bool, int, float, complex, type(None))


def isfunc(mod, f):
    assert hasattr(mod, f)
    attr = getattr(mod, f)

    if len(f) >= 2:
        if f[0] == "_": return False # Exclude functions starting with '_'
        # if f[:2] == '__' and f[-2:] == '__': return False

    # Ignore functions to this list if they cause recursion
    ignore = ['size', 'tolist', 'dim', 'is_storage', 'item']
    # Also ignore the following functions related to tensor initialization
    ignore += ['zero_', 'uniform_', 'normal_', 'fill_']
    # and more
    ignore += ['copy_', 'numel', 'set_', 'has_names', 'index_select', 'contiguous', 'detach', 'as_strided',
               'view_as', 'cpu', 'cuda', 'bool', 'float', 'half', 'long', 'to', 'type']
    if f in ignore:
        return False

    ignore_patterns = ['storage', 'stride', 'has_torch_function', 'new', 'is_']
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

def init_patching():
    # patches torch.Tensor/torch.nn.functional functions
    for mod in modules_to_wrap:
        patching(mod)


def patching_hook(specify_op: Union[str, List[str], Tuple[str]] = None,
                  export_input_data: bool = False,
                  export_file_path: str ='/tmp/picklefile') -> None:
    '''
    Applies patching to all `modules_to_wrap`, hooks every input of nn modules,
    and exports the info to a pickle file for further analysis.
    '''
    init_patching()

    if specify_op is not None:
        if not isinstance(specify_op, (list, tuple)):
            specify_op = [specify_op]
        for op in specify_op:
            assert any([isfunc(mod, op) for mod in modules_to_wrap]), f'"{op}" is an INVALID operator name.'

    def hook(module, inputs, print_debug_info=False):
        module_str = repr(module).rstrip('()')
        if module_str.startswith(WRAPPING_PREFIX):
            module_name = module_str[len(WRAPPING_PREFIX):]

            # If specified, only hook these interesting operators
            if specify_op is not None and module_name not in specify_op: return

            if print_debug_info: print('module:', module_name)
            values = []
            for inp in inputs:
                if torch.is_tensor(inp):
                    if inp.shape == torch.Size([]): # Treat tensor with no size as a number
                        value = inp.item()
                    else:
                        value = inp if export_input_data else inp.shape
                else:
                    value = inp
                    inp_type = type(inp)
                    if inp_type not in supported_types:
                        # Skip unknown data type case
                        print(f'In module {module_name}, input type {inp_type.__name__} not supported.')
                        return
                values.append(value)
                # if print_debug_info: print(key, value)
            dict_to_save = {'op': module_name, 'inputs': tuple(values)}
            with open(export_file_path, 'ab') as f:
                pickle.dump(dict_to_save, f)

    nn.modules.module.register_module_forward_pre_hook(hook)

def load_pickle(pickle_file: str ='/tmp/picklefile') -> List[dict]:
    '''
    Loads input info, remove duplicates.

    Returns: two dicts [`input_info`, `input_data`].
             `input_data` is empty if there are no tensor data present.
    '''
    input_info = {}
    input_data = {}
    no_data = True
    with open(pickle_file, 'rb') as f:
        while True:
            try:
                x = pickle.load(f)
                op, inputs = x['op'], x['inputs']
                info = tuple(inp.shape if torch.is_tensor(inp) else inp for inp in inputs)
                if no_data and any([torch.is_tensor(inp) for inp in inputs]):
                    no_data = False

                if op in input_info:
                    input_info[op].add(info)
                else:
                    input_info[op] = set(info)

                if op in input_data:
                    input_data[op].append(inputs)
                else:
                    input_data[op] = [inputs]
            except EOFError:
                break

    for k, v in input_info.items():
        input_info[k] = sorted(list(v), key=pprint._safe_key)

    print('Operator list:', list(input_info.keys()))
    if no_data: return input_info, {}
    return input_info, input_data
