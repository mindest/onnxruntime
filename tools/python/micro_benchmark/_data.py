import torch
import numpy as np
from collections import OrderedDict

class ConcreteInputDesc:
    def __init__(self, numpy_input_values):
        self.values = numpy_input_values

    def __len__(self):
        return len(self.values)

    def value_at(self, index):
        assert index < len(self.values)
        return self.values[index]

class LazyInputDesc:
    def __init__(self, input_shapes, dtypes):
        self.input_shapes = input_shapes
        self.dtypes = dtypes
        assert len(self.input_shapes) == len(self.dtypes)

    def __len__(self):
        return len(self.input_shapes)

    def value_at(self, index):
        assert index < len(self.input_shapes)
        with torch.no_grad():
            data = torch.rand(*self.input_shapes[index])
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
            variable_arg_names (list string): The argument name of variables in this run.
            variable_arg_vals (list of any type): All possible values of the variables in this run.
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