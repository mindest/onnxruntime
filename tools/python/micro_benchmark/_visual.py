import csv
import os
import pandas as pd

def write_to_csv(header, data, csv_path='perf_stat.csv'):
    print("benchmark stat file written to {} ...".format(csv_path))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def show_benchmark_reports(dirs):
    for d in dirs:
        for filename in os.listdir(d):
            if not filename.endswith('.csv'):
                continue

            full_filename = os.path.join(d, filename)
            df = pd.read_csv(full_filename)

            new_df = pd.concat([df['mean'], df['p0.5'], df['p0.2'], df['p0.8']]).reset_index(drop=True)
            percentile_0_8 = float(new_df.describe(percentiles=[0.8])['80%'])

            unit = 'ms'
            if percentile_0_8 > 1.0:
                # use the default ms as unit
                pass
            else:
                unit = 'us'
                df['mean'] = 1000 * df['mean']
                df['p0.5'] = 1000 * df['p0.5']
                df['p0.2'] = 1000 * df['p0.2']
                df['p0.8'] = 1000 * df['p0.8']

            df = df.rename({'mean': 'mean ({})'.format(unit), 'p0.5': 'p0.5 ({})'.format(unit), 'p0.2': 'p0.2 ({})'.format(unit), 'p0.8': 'p0.8 ({})'.format(unit) }, axis=1)  # new method
            print('=' * 100, "\nstats from {}: \n".format(full_filename), df)