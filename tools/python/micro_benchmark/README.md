# Micro-benchmarking

Some introductions to micro-benchmarking...

## Extract input info of operators in a training model

We want to obtain the input scales and parameter settings of the interesting operators for benchmarking.

Operators subclassing `torch.nn.Module` can be hooked with native PyTorch hooks. The following setting "registers a forward pre-hook common to all modules".

```py
from torch import nn

def hook(module, input):
    # Do something in the hook
    pass

nn.modules.module.register_module_forward_pre_hook(hook)
```

However, other functions like that from `torch.nn.functional` (noted as `F`) and `torch.Tensor` cannot be captured. To overcome this problem, we use a monkey-patching method to rewrite those functions or methods from `F` and `torch.Tensor` at runtime.

The main idea is to rewrite functions with a wrapper subclassing `torch.nn.Module`:

```py
def add_wrapper(mod, name):
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
```

For example, for function `F.relu`, `add_wrapper` wraps it with a new class `relu`. Now `F.relu` points to an instance of the new class `relu`, but other settings are kept.

### usage

Add the following lines to use's Python training script:

```py
    import sys
    sys.path.append('<PATH/TO/patching_hook.py>')
    from patching_hook import init, hook
    from torch import nn
    nn.modules.module.register_module_forward_pre_hook(hook)
    init()
```
The `init()` method initializes the patching.

Output: The input information will be collected and exported to `/tmp/picklefile` by default.

### Known issues/Todos

- Not all functions/methods from `torch.Tensor` are included (`repeat`, `view`, `ne`, `cumsum` are included for now).


## Performance measurement of operators


The simple prototype...

### Usage

User needs to provide some necessary information:

- `model`: mini-model for test
- `model_init_params`: list of param names for initializing `model`
- `input_params`: list of list of input dim names, needed for generating inputs: `[[dim names for first input], ...]`
- `input_scales`: dict of input shapes: `{dim_name: dim_variation}`
- `job_configs`: using fp16 or fp32, pt or ort

Output: a `.csv` file with kernel performance statistics under different input scales and configs

### Known issues/Todos
