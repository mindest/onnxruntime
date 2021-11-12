import csv
import os
import pandas as pd
from _data import load_stats

def show_benchmark_reports(dirs):
    for d in dirs:
        for filename in os.listdir(d):
            if not filename.endswith('.pkl'):
                continue

            full_filename = os.path.join(d, filename)
            table = load_stats(full_filename)
            df = pd.DataFrame(data=table['data'], columns=table['header'])
            print('=' * 100, "\nstats from {}: \n".format(full_filename), df)
