import argparse
import sys
import os
import inspect
from _common import BenchmarkRunner
from _visual import show_benchmark_reports

def run_benchmarks(save_dir, name_filter):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bench_dir = os.path.dirname(os.path.realpath(__file__))
    bench_report_save_dirs = []
    for filename in os.listdir(bench_dir):
        if not filename.endswith('.py') or not filename.startswith('bench_'):
            continue

        # filter out the cases to run based on passed in 'name_filter' argument.
        if name_filter and name_filter not in filename:
            continue

        print(f'running {filename}...')
        mod = __import__(os.path.splitext(filename)[0])
        benchmarks = inspect.getmembers(mod, lambda x: isinstance(x, BenchmarkRunner))

        for name, bench in benchmarks:
            save_dir_for_this_bench = os.path.join(save_dir, mod.__name__)
            save_dir_for_this_bench = os.path.join(save_dir_for_this_bench, name)
            if not os.path.exists(save_dir_for_this_bench):
                os.makedirs(save_dir_for_this_bench)
            bench.run(save_path=save_dir_for_this_bench)
            bench_report_save_dirs.append(save_dir_for_this_bench)

    return bench_report_save_dirs


def main(args):
    parser = argparse.ArgumentParser(description="Run the benchmark suite.")
    parser.add_argument("-d", "--save_dir", type=str, default='./', required=False)
    parser.add_argument("-f", "--filter", type=str, default='', required=False)

    args = parser.parse_args(args)
    bench_report_save_dirs = run_benchmarks(args.save_dir, args.filter)
    show_benchmark_reports(bench_report_save_dirs)

if __name__ == '__main__':
    main(sys.argv[1:])
