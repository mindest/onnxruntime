import csv
import os
import pickle

from pandas.core.reshape.pivot import pivot_table
import numpy as np

class ConcreteInputDesc:
    """
    This class is used to depict an fixed input.

    """
    def __init__(self, input_values):
        """
        Args:
            input_values (list of torch tensor or numpy value or python list):
                The inputs that can be directly used by training without any post-processing.
        """
        self.values = input_values

    def __len__(self):
        return len(self.values)

    def value_at(self, index):
        assert index < len(self.values)
        return self.values[index]

class LazyInputDesc:
    """
    This class is used to depict an input that can be allocated/randomized before the benchmark started.

    """
    def __init__(self, input_shapes, dtypes):
        self.input_shapes = input_shapes
        self.dtypes = dtypes
        assert len(self.input_shapes) == len(self.dtypes)

    def __len__(self):
        return len(self.input_shapes)

    def value_at(self, index):
        assert index < len(self.input_shapes)
        data = np.random.rand(*self.input_shapes[index])
        return data

class InputGenerator:
    """
    This class is used by to generate inputs.

    """
    def __init__(self, input_desc_dict):
        """
        Args:
            input_names (list of string): The input data names.
            input_generator (lamda to return a list of input_values): A function to generate all input data values.
            variable_names (list string): The argument name of variables in this run.
            variable_values_pool (list of any type): All possible values of the variables in this run.
        """
        self._input_desc_dict = input_desc_dict
        data_group_size = None
        for name, desc in self._input_desc_dict.items():
            if data_group_size is not None and data_group_size != len(desc):
                raise ValueError("Given inputs' count mismatch between different inputs")
            data_group_size = len(desc)

        self._data_group_size = data_group_size

    def __iter__(self):
        for i in range(self._data_group_size):
            input_name_value_pairs = {}
            for name, desc in self._input_desc_dict.items():
                input_name_value_pairs[name] = desc.value_at(i)
            yield input_name_value_pairs.items()

    @property
    def input_names(self):
        return list(self._input_desc_dict.keys())

def persistent_stats(stats, visual_config, file_path='perf_stat.csv'):
    print("benchmark stat file written to {} ...".format(file_path))
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, 'ab') as f:
        header = None
        body = []

        for stat in stats:
            print(stat.input_values)
            if header is None:
                header = {
                    'input_names': stat.input_names,
                    'variable_names': stat.variable_names,
                    'statistic_names': stat.statistic_names
                }
            body.append({
                "input_values" : stat.input_values,
                "variable_values": stat.variable_values,
                "statistic_values": stat.statistic_values
            })
        tbl = {"header": header, "data": body, "visual_config": visual_config}
        pickle.dump(tbl, f)

def load_stats(filename):
    if not filename.endswith('.pkl'):
        raise ValueError("invalid file type: {}".format(filename))

    with open(filename, 'rb') as f:
        tbl = pickle.load(f)

        all_statistic_values = []
        for row in tbl['data']:
            all_statistic_values.extend(row["statistic_values"])
        percentile_0_8 = float(np.percentile(all_statistic_values, 80))
        unit = 'ms' if percentile_0_8 > 1.0 else 'us'

        visual_config = tbl['visual_config']
        if visual_config and visual_config.is_valid:
            pivot_idx = None
            for idx, name in enumerate(tbl['header']['variable_names']):
                if name == visual_config.pivot_variable_name:
                    pivot_idx = idx
                    break

            if pivot_idx is None:
                raise RuntimeError("pivot variable name not found")

            group_by_pivot = {}
            for row in tbl['data']:
                flattened_row = row['input_values'] + row['variable_values']
                pivot_value = row['variable_values'][pivot_idx]
                if pivot_value not in group_by_pivot:
                    group_by_pivot[pivot_value] = {}

                flattened_row.pop(len(row['input_values']) + pivot_idx)
                sub_key = '_'.join(flattened_row)
                group_by_pivot[pivot_value][sub_key] = row

            control = visual_config.pivot_varible_control_value
            if control in group_by_pivot and len(group_by_pivot.keys()) != 2:
                raise RuntimeError("cannot use more than 2-values variable as pivot")

            povot_value_pool = list(group_by_pivot.keys())
            povot_value_pool.remove(control)
            treatment = povot_value_pool[0]
            updated_header = []
            header = tbl['header']
            updated_header += header['input_names'] + header['variable_names']
            updated_header.pop(len(header['input_names']) + pivot_idx)
            updated_header += header['statistic_names']
            updated_header += header['statistic_names']
            updated_header += header['statistic_names']

            updated_tbl = {}
            updated_tbl['header'] = updated_header

            updated_data = []
            statistic_value_count = None
            for sub_key, row in group_by_pivot[control].items():
                updated_row = row['input_values'] + row['variable_values']
                updated_row.pop(len(header['input_names']) + pivot_idx)

                updated_row += row['statistic_values']
                updated_row += group_by_pivot[treatment][sub_key]['statistic_values']
                from operator import truediv
                sub_ret = list(map(truediv, group_by_pivot[treatment][sub_key]['statistic_values'], row['statistic_values']))
                updated_row += ["{0:.3%}".format(s - 1.0) for s in sub_ret]
                updated_data.append(updated_row)

                statistic_value_count = len(row['statistic_values'])

            updated_tbl['data'] = updated_data
            updated_tbl['control_prefix'] = "{} ({})".format(control, unit)
            updated_tbl['treatment_prefix'] = "{} ({})".format(treatment, unit)
            updated_tbl['diff_prefix'] = "{}".format('diff')
            updated_tbl['statistic_value_count'] = statistic_value_count
        else:
            updated_tbl = {'data': []}

            updated_header = []
            header = tbl['header']
            updated_header += header['input_names'] + header['variable_names']
            updated_header += ["{} ({})".format(name, unit) for name in header['statistic_names']]

            updated_tbl['header'] = updated_header

            statistic_value_count = None
            for row in tbl['data']:
                updated_statistic_values = [v * 1000 for v in row["statistic_values"]]
                updated_tbl['data'].append(row['input_values'] + row['variable_values'] + updated_statistic_values)
                statistic_value_count = statistic_value_count

            updated_tbl['statistic_value_count'] = statistic_value_count

    return updated_tbl


def write_to_csv(header, data, csv_path='perf_stat.csv'):
    print("benchmark stat file written to {} ...".format(csv_path))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)