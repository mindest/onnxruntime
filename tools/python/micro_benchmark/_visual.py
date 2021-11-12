import csv
import os
import pandas as pd
import numpy as np
from _data import load_stats

def show_benchmark_reports(dirs):
    for d in dirs:
        for filename in os.listdir(d):
            if not filename.endswith('.pkl'):
                continue

            full_filename = os.path.join(d, filename)
            table = load_stats(full_filename)
            if 'control_prefix' in table:
                index = [
                    np.array(
                        table['header'][:-3 * table['statistic_value_count']]
                         + [table['control_prefix']] * table['statistic_value_count']
                         + [table['treatment_prefix']] * table['statistic_value_count']
                         + [table['diff_prefix']] * table['statistic_value_count']
                    ),
                    np.array(table['header'])
                ]

                df = pd.DataFrame(data=table['data'], columns=index)
            else:
                df = pd.DataFrame(data=table['data'], columns=table['header'])
            print('=' * 100, "\nstats from {}: \n".format(full_filename), df.to_string(index=False, justify='left'))
