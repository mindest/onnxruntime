import csv
import os
import pickle
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

def persistent_stats(header, data, visual_config, statistic_value_count, file_path='perf_stat.csv'):
    print("benchmark stat file written to {} ...".format(file_path))
    with open(file_path, 'ab') as f:
        tbl = {"header": header, "data": data, "statistic_value_count": statistic_value_count, "visual_config": visual_config}
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

        updated_tbl = {'data': []}
        updated_tbl['header'] = [ "{} ({})".format(name, unit) for name in tbl['header']]

        for row in tbl['data']:
            updated_statistic_values = [v * 1000 for v in row["statistic_values"]]
            updated_tbl['data'].append(row['input_values'] + row['variable_values'] + updated_statistic_values)

    return updated_tbl


def write_to_csv(header, data, csv_path='perf_stat.csv'):
    print("benchmark stat file written to {} ...".format(csv_path))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)