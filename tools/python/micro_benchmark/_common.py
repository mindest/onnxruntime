import os
import torch
import itertools
import numpy as np
from _data import persistent_stats

def op_benchmarks(benchmark_configs, visual_config=None):
    """A function decorator for benchmarking. The benchmark can then be executed by `.run` method on the return value.

    Args:
        benchmarks (list of BenchmarkConfig): Benchmarking configurations.
    """
    wrapper = lambda fn: BenchmarkRunner(fn, benchmark_configs, visual_config)
    return wrapper

class BenchmarkConfig:
    """
    This class is used by the `op_benchmarks` function.

    """
    def __init__(
        self,
        input_generator,
        variable_names,
        variable_values_pool
    ):
        """
        Args:
            input_generator (lamda to return a list of input_values): A function to generate all input data values.
            variable_names (list string): The argument name of variables in this run.
            variable_values_pool (list of any type): All possible values of the variables in this run.
        """
        self.input_generator = input_generator

        self.variable_names = variable_names
        self.variable_values_pool = variable_values_pool


class VisualConfig:
    """
    This class is used by the `op_benchmarks` function.

    """
    def __init__(
        self,
        pivot_variable_name = None,
        pivot_varible_control_value = None
    ):
        """
        Args:
            pivot_variable_name (string): The variable we planned to use as comparasion pivot.
            pivot_varible_control_value (string): The value we take as baseline to do the compare.
        """
        self.pivot_variable_name = pivot_variable_name
        self.pivot_varible_control_value = pivot_varible_control_value

    @property
    def is_valid(self):
        return self.visual_config.pivot_variable_name and self.visual_config.pivot_varible_control_value

class StatItem:
    """
    This class is used to represent the statistics for a fine-grained run at the minimum unit.
        (e.g. one fixed dataset + one of the `variable combinatations`).

    """
    def __init__(
        self,
        variable_combination,
        input_combination,
        statistic_dict
    ):
        """
        Args:
            variable_combination (dict <str, any type>): Contains the concrete values for each variable.
            input_combination (dict <str, numpy value>): Contains the concrete inputs for each input.
            statistic_dict (dict <str, float>): The statistics returned by benchmark run.
        """
        self.variable_combination = variable_combination
        self.input_combination = {}
        for input_name, input_value in input_combination.items():
            # TODO: refine this when we need show initial N elements to distinguish given concrete input values.
            self.input_combination[input_name] = str(list(input_value.shape)) + "-" + str(input_value.dtype) 

        self.statistic_dict = statistic_dict

    @property
    def input_names(self):
        return list(self.input_combination.keys())

    @property
    def input_values(self):
        return list(self.input_combination.values())

    @property
    def variable_names(self):
        return list(self.variable_combination.keys())

    @property
    def variable_values(self):
        return list(self.variable_combination.values())

    @property
    def statistic_names(self):
        return list(self.statistic_dict.keys())

    @property
    def statistic_values(self):
        return list(self.statistic_dict.values())

class BenchmarkRunner:
    """
    The class manages the benchmark running, result analysis.

    """
    def __init__(self, fn, benchmark_configs, visual_config):
        self.fn = fn
        self.benchmark_configs = benchmark_configs
        self.visual_config = visual_config

    def _generate_variable_combinations(self, bench):
        names = bench.variable_names
        vals = bench.variable_values_pool
        settings = []
        for name, val in zip(names, vals):
            setting = []
            for v in val:
                setting.append((name, v))
            settings.append(setting)
        cartesian_prod = itertools.product(*settings)
        return [dict(prod) for prod in cartesian_prod]

    def _run(self, bench):
        combination_list = self._generate_variable_combinations(bench)
        print(f"all combinations listed as below: ", combination_list)

        stats = []
        for input_name_value_pair in bench.input_generator:
            input_args = {}
            for input_name, input_value in input_name_value_pair:
                input_args[input_name] = input_value

            for one_variables_combination in combination_list:
                rets = self.fn(**input_args, **one_variables_combination)
                stats.append(StatItem(one_variables_combination, input_args, rets))

        return stats

    def _save_stats(self, save_path, stats):
        header = None
        statistic_value_count = None
        body = []
        postfix = f'raw_perf_stat.pkl'
        for stat in stats:
            if header is None:
                header = stat.input_names + stat.variable_names + stat.statistic_names
                statistic_value_count = len(stat.statistic_names)
            body.append({
                "input_values" : stat.input_values, "variable_values": stat.variable_values,
                "statistic_values": stat.statistic_values})
        persistent_stats(header, body, self.visual_config, statistic_value_count, file_path=os.path.join(save_path, postfix))

    def run(self, save_path=''):
        has_single_bench = isinstance(self.benchmark_configs, BenchmarkConfig)
        benchmarks = [self.benchmark_configs] if has_single_bench else self.benchmark_configs

        aggregated_stats = []
        for bench in benchmarks:
            stats = self._run(bench)
            aggregated_stats.extend(stats)

        self._save_stats(save_path, aggregated_stats)



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

    # maintain a buffer of 256 MB that we clear before each kernel call to make sure that the L2
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
        # clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()

        # TODO: add backward support

    # record clocks
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
