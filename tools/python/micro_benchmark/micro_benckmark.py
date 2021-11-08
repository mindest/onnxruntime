import torch
from torch import nn, randn
from onnxruntime.training.ortmodule import ORTModule
import itertools
import time
import numpy as np
import csv


class BatchNorm1dModel(nn.Module):
    def __init__(self, C=100):
        super().__init__()
        self.bn1d = nn.BatchNorm1d(C)
    def forward(self, x):
        return self.bn1d(x)

class LinearModel(nn.Module):
    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.ln = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.ln(x)


def cartesian_product_configs(**configs):
    '''
    return an iterator of all possible config combinations
    '''
    configs_attrs_list = []
    for key, values in configs.items():
        tmp_results = [{key : value} for value in values]
        configs_attrs_list.append(tmp_results)
    generated_configs = itertools.product(*configs_attrs_list)
    return generated_configs

def run_step(model, *inputs, run_backward=True):
    prediction = model(*inputs)
    if run_backward:
        loss = prediction.sum()
        loss.backward()

def generate_init_and_input(config_dict, init_params, input_params):
    init_dict = {key: config_dict[key] for key in init_params}
    inputs = []
    for input_param in input_params:
        input_shape = [config_dict[key] for key in input_param]
        inputs.append(randn(*input_shape).cuda())
    return init_dict, inputs

def write_to_csv(keys, data, csv_path='perf_stat.csv'):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = keys + ['mean (us)', 'std (us)']
        writer.writerow(header)
        writer.writerows(data)

def run_over_configs(configs, orig_model, model_init_params, input_params, keys, warmup=100, steps=500, repeat=20):
    stat = []
    for config in configs:
        config = {key: val for d in config for key, val in d.items()}

        use_fp16 = config['use_fp16']
        cuda_sync = True # tmp
        use_ort = config['use_ort']

        init_dict, inputs = generate_init_and_input(config, model_init_params, input_params)

        avgs = []
        for _ in range(repeat):
            model = orig_model(**init_dict).cuda()

            if use_fp16:
                model = model.half()
                inputs = [inp.half() for inp in inputs]

            if use_ort:
                model = ORTModule(model)

            # warmup steps are not included in statistics
            for _ in range(warmup):
                run_step(model, *inputs)

            if cuda_sync:
                torch.cuda.synchronize(torch.cuda.current_device())
            start = time.time()

            for _ in range(steps):
                run_step(model, *inputs)

            if cuda_sync:
                torch.cuda.synchronize(torch.cuda.current_device())
            end = time.time()

            avg = (end - start) / steps * 1e6
            avgs.append(avg)
        mean = np.mean(avgs)
        std = np.std(avgs)
        stat.append([config[key] for key in keys] + [mean, std])
    write_to_csv(keys, stat)

def main(model, model_init_params, input_params, input_scales, job_configs):
    config_iter = cartesian_product_configs(**input_scales, **job_configs)
    keys = list(input_scales.keys()) + list(job_configs.keys())
    run_over_configs(config_iter, model, model_init_params, input_params, keys)


if __name__ == '__main__':
    model = LinearModel
    model_init_params = ['in_features', 'out_features']
    input_params = [
        ['N', 'in_features'] # first input
    ]

    # specify values for each dim of interest
    input_scales = {
        'N': [2 ** i for i in range(9, 11)],
        'in_features': [2 ** i for i in range(9, 11)],
        'out_features': [2 ** i for i in range(9, 11)]
    }

    job_configs = {
        'use_fp16': [True, False],
        'use_ort': [True, False]
    }

    main(model, model_init_params, input_params, input_scales, job_configs)
