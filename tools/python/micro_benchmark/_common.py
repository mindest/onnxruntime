import os
import torch
import itertools
import numpy as np
from _data import persistent_stats
import copy

def op_benchmarks(benchmark_configs, visual_config=None):
    """
    A function decorator for benchmarking. The benchmark can then be executed by `.run` 
    method on the return value.

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
            input_generator (lamda to return a list of input_values):
                A function to generate all input data values.
            variable_names (list string):
                The argument name of variables in this run.
            variable_values_pool (list of any type):
                All possible values of the variables in this run.
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
            pivot_variable_name (string):
                The variable we planned to use as comparasion pivot.
            pivot_varible_control_value (string):
                The value we take as baseline to do the compare.
        """
        self.pivot_variable_name = pivot_variable_name
        self.pivot_varible_control_value = pivot_varible_control_value

    @property
    def is_valid(self):
        return self.pivot_variable_name and self.pivot_varible_control_value

class StatisticItem:
    """
    This class is used to represent one single entry of statistic.

    """
    def __init__(
        self,
        value,
        is_diffable = True
    ):
        """
        Args:
            value (any type): statistic value (usually is a float).
            is_diffable (bool): whether this statistic value can be diff-able in comparision mode.
        """
        self._value = value
        self._is_diffable = is_diffable

class RetRecordName:
    """
    This class is used to represent one single column properties.

    """
    def __init__(self, name, is_input=False, is_variable=False, is_statistic=False, is_diffable=False):
        self._name = name
        self._is_input = is_input
        self._is_variable = is_variable
        self._is_statistic = is_statistic
        self._is_diffable = is_diffable

    def __str__(self):
        return 'name: {}, _is_input: {}, _is_variable: {}, _is_statistic: {}, _is_diffable: {}'.format(
            self._name, self._is_input, self._is_variable, self._is_statistic, self._is_diffable
        )

class RetRecordNameSet:
    """
    This class is used to be containers of multiple `RetRecordName`s 

    """
    def __init__(self):
        self._name_set = []

    def extend(self, record_names):
        self._name_set.extend(record_names)

    def __iter__(self):
        for n in self._name_set:
            yield n

class RetRecordValue:
    """
    This class is used to represent one single column value.

    """
    def __init__(
        self,
        value,
    ):
        self._value = value

    def __str__(self):
        return '_value: {}'.format(self._value)


class RetRecordValueSet:
    """
    This class is used to be containers of multiple `RetRecordValue`s 

    """
    def __init__(self):
        self._value_set = []

    def extend(self, record_values):
        self._value_set.extend(record_values)

    def append(self, record_value):
        self._value_set.append(record_value)

    def __iter__(self):
        for _, v in enumerate(self._value_set):
            yield v

class RunRets:
    """
    This class is used to manage column names and all comulmn values for one benchmark run.

    """
    def __init__(
        self
    ):
        # record keys
        self._record_names = None

        # record values
        self._record_values = []

        self._is_initialized = False

    def append(self, input_combination, variable_combination, statistic_items_in_dict):
        if not self._is_initialized:
            self._record_names = RetRecordNameSet()
            self._record_names.extend(
                list(RetRecordName(input_name, is_input=True) for input_name in input_combination.keys())
            )

            self._record_names.extend(
                list(RetRecordName(variable_name, is_variable=True) for variable_name in variable_combination.keys())
            )

            self._record_names.extend(
                list(RetRecordName(statistic_name, is_statistic=True, is_diffable=item._is_diffable)
                    for statistic_name, item in statistic_items_in_dict.items())
            )

        new_record_values = RetRecordValueSet()
        for record_name in self._record_names:
            if record_name._name in input_combination:
                assert record_name._is_input == True
                new_record_values.append(RetRecordValue(input_combination[record_name._name]))
            elif record_name._name in variable_combination:
                assert record_name._is_variable == True
                new_record_values.append(RetRecordValue(variable_combination[record_name._name]))
            elif record_name._name in statistic_items_in_dict.keys():
                assert record_name._is_statistic == True
                new_record_values.append(RetRecordValue(statistic_items_in_dict[record_name._name]._value))
            else:
                raise ValueError('find input name mismatch')

        self._record_values.append(new_record_values)

        if not self._is_initialized:
            self._is_initialized = True

    @property
    def record_names(self):
        return copy.deepcopy(self._record_names)

    def iterator(self):
        for values in self._record_values:
            yield {n : v for n, v in zip(self._record_names, values)}.items()


class BenchmarkRunner:
    """
    The class manages the benchmark running, result saving.

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

        run_rets = RunRets()
        for input_name_value_pair in bench.input_generator:
            input_args = {}
            for input_name, input_value in input_name_value_pair:
                input_args[input_name] = input_value

            for one_variables_combination in combination_list:
                ret_in_dict = self.fn(**input_args, **one_variables_combination)
                run_rets.append(input_args, one_variables_combination, ret_in_dict)

        return run_rets

    def _save_stats(self, save_path, stats):
        postfix = f'raw_perf_stat.pkl'
        persistent_stats(stats, file_path=os.path.join(save_path, postfix))

    def run(self, save_path=''):
        has_single_bench = isinstance(self.benchmark_configs, BenchmarkConfig)
        benchmarks = [self.benchmark_configs] if has_single_bench else self.benchmark_configs

        aggregated_stats = []
        for bench in benchmarks:
            run_rets = self._run(bench)
            aggregated_stats.append(run_rets)

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

    ret = {'mean' : StatisticItem(torch.mean(times).item())}
    if extract_kernel_info and len(kernel_events) > 0:
        kernel_sv = StatisticItem( 
            [f'{evt["name"]}, grid {evt["args"]["grid"]}, block {evt["args"]["block"]}, dur {evt["dur"]}us'
                for evt in kernel_events
            ], is_diffable=False)
        ret['kernel'] = kernel_sv
    if percentiles:
        percentiles_rets = torch.quantile(times, torch.tensor(percentiles)).tolist()
        for index, r in enumerate(percentiles_rets):
            ret['p{}'.format(percentiles[index])] = StatisticItem(r)

    return ret
