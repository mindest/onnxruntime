import copy
import numpy as np
import os
import torch
import itertools
from _visual import write_to_csv

def op_benchmarks(benchmarks):
    """A function decorator for benchmarking. The benchmark can then be executed by `.run` method on the return value.

    Args:
        benchmarks (list of BenchmarkDef): Benchmarking configurations.
    """
    wrapper = lambda fn: BenchmarkRunner(fn, benchmarks)
    return wrapper

class BenchmarkDef:
    """
    This class is used by the `op_benchmarks` function.

    """
    def __init__(
        self,
        input_generator,
        variable_arg_names,
        variable_arg_vals,
    ):
        """
        Args:
            input_generator (lamda to return a list of input_values): A function to generate all input data values.
            variable_arg_names (list string): The argument name of variables in this run.
            variable_arg_vals (list of any type): All possible values of the variables in this run.
        """
        self.input_generator = input_generator

        self.variable_arg_names = variable_arg_names
        self.variable_arg_vals = variable_arg_vals


class BenchmarkRunner:
    """
    The class manages the benchmark running, result analysis.

    """
    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    def _generate_variable_combinations(self, bench):
        names = bench.variable_arg_names
        vals = bench.variable_arg_vals
        settings = []
        for name, val in zip(names, vals):
            setting = []
            for v in val:
                setting.append((name, v))
            settings.append(setting)
        cartesian_prod = itertools.product(*settings)
        return [dict(prod) for prod in cartesian_prod]

    def _run(self, bench, save_path):
        combination_list = self._generate_variable_combinations(bench)
        print(f"all combinations listed as below: ", combination_list)

        table_body = []
        stat_names = None
        for input_name_value_pair in bench.input_generator:
            input_args = {}
            row_data_cells = []
            for input_name, input_value in input_name_value_pair:
                row_data_cells.append(list(input_value.shape))
                input_args[input_name] = input_value

            for one_variables_combination in combination_list:
                row_body = list(one_variables_combination.values())
                row_body += copy.deepcopy(row_data_cells)
                rets = self.fn(**input_args, **one_variables_combination)
                row_body.extend(list(rets.values()))
                if not stat_names:
                    stat_names = list(rets.keys())
                table_body.append(row_body)

        table_header = list(combination_list[0].keys()) + bench.input_generator.input_names + stat_names

        if save_path:
            write_to_csv(table_header, table_body, csv_path=os.path.join(save_path, f"perf_stat.csv"))

    def run(self, save_path=''):
        has_single_bench = isinstance(self.benchmarks, BenchmarkDef)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        for bench in benchmarks:
            self._run(bench, save_path)

def run_op_benchmark(fn, extract_kernel_info=False, warmup_step=25, repeat_step=100,
                     grad_to_none=None, percentiles=[0.5, 0.2, 0.8]):
    """Run operator computation representation fn `repeat_step` times, and generate kernel latency statistics.

    To minimize the cold start impact, we allow ignoring the initial `warmup_step` steps in out statistics.

    Args:
        fn (lamda function): Callable function that run the computation.
        warmup_step (int): How many initial steps are NOT included in final statistics.
        repeat_step (int): How many steps are used for statistics.
        percentiles: (list of float): A list indicating the performance percentile matrix.
            For example [0.2, 0.8] means, return 20-th and 80-th performance percentile.
        grad_to_none: (optional, list of gradients) List of gradients that are not intented to be accumulated
            in case of the overhead affecting kernel time measurement.

    Returns:
        Returns a dictionary (stat name, stat value):
            The first item in the dict is always the median run time.
            The other items return corresponding performance percentiles,
                aligned with the input 'percentiles'.
    """

    # block all previous CUDA operations, avoid any impact for benchmark.
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear before each kernel call to make sure that the L2
    # doesn't contain any input data before the run. (we don't want the potentially L2 cached input
    # affect the overall latency)
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(repeat_step)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(repeat_step)]
    cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')

    # PyTorch Profiler profiling steps
    if extract_kernel_info:
        print("PyTorch Profiler profiling...")
        from torch.profiler import profile, ProfilerActivity, schedule
        import json
        from tempfile import NamedTemporaryFile
        with profile(activities=[ProfilerActivity.CUDA], schedule=schedule(wait=1, warmup=1, active=1)) as prof:
            for _ in range(3):
                fn()
                prof.step()
        with NamedTemporaryFile('w+t') as f:
            # Export tracing info to a temp JSON file and parse kernel info.
            # Temp file is auto-deleted afterwards.
            prof.export_chrome_trace(f.name)
            tracing_events = json.load(open(f.name))['traceEvents']
        kernel_events = [evt for evt in tracing_events if 'cat' in evt and evt['cat'] == 'Kernel']

    # warm up
    print(f"warming up now....")
    for _ in range(warmup_step):
        fn()

    # start benchmark
    print(f"starting benchmark now....")
    for i in range(repeat_step):
        # we don't want `fn` to accumulate gradient values if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])

    ret = {}
    ret["mean"] = torch.mean(times).item()
    if extract_kernel_info and len(kernel_events) > 0:
        ret['kernel'] = [
            f'{evt["name"]}, grid {evt["args"]["grid"]}, block {evt["args"]["block"]}, dur {evt["dur"]}us' \
                for evt in kernel_events]
    if percentiles:
        percentiles_rets = torch.quantile(times, torch.tensor(percentiles)).tolist()
        for index, r in enumerate(percentiles_rets):
            ret["p{}".format(percentiles[index])] = r

    return ret
