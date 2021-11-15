import csv
import copy
import os
import pandas as pd
import numpy as np
from _data import load_stats
from _common import RunRets

def npyarray2string(npy_arr):
    return str(npy_arr.shape) + str(npy_arr.dtype)

def show_benchmark_report(cur_benchmark_stat, visual_config, full_filename):
    all_diffable_statistic_values = []
    for record_name_value_pairs in cur_benchmark_stat.iterator():
        for record_name, record_value in record_name_value_pairs:
            if record_name._is_statistic and record_name._is_diffable:
                all_diffable_statistic_values.append(record_value._value)
    percentile_0_8 = float(np.percentile(all_diffable_statistic_values, 80))
    unit, scalor = tuple(['ms', 1.0]) if percentile_0_8 > 1.0 else tuple(['us', 1000.0])

    # update the diffable statistic values according to scalor.
    for record_name_value_pairs in cur_benchmark_stat.iterator():
        for record_name, record_value in record_name_value_pairs:
            if record_name._is_statistic and record_name._is_diffable:
                record_value._value *= scalor

    if visual_config and visual_config.is_valid:
        group_by_pivot = {}
        # iterate each run result, group by two layers:
        #   layer 1: pivot viariable value
        #   layer 2: input values + variable values (excluding pivot) as key.
        #       This is needed to map between control and treatment compare.
        for record_name_value_pairs in cur_benchmark_stat.iterator():
            input_values_and_excluding_pivot_variable_values = []
            pivot_value = None
            for record_name, record_value in record_name_value_pairs:
                if (record_name._is_variable or record_name._is_input) and record_name._name != visual_config.pivot_variable_name:
                    value_str = record_value._value
                    if isinstance(record_value._value, np.ndarray):
                        value_str = npyarray2string(record_value._value)
                    input_values_and_excluding_pivot_variable_values.append(value_str)

                if record_name._name == visual_config.pivot_variable_name:
                    pivot_value = record_value._value

            if pivot_value is None:
                raise RuntimeError('pivot_value is none: {}'.format(record_name_value_pairs))

            if pivot_value not in group_by_pivot:
                group_by_pivot[pivot_value] = {}

            sub_key = '_'.join(input_values_and_excluding_pivot_variable_values)

            group_by_pivot[pivot_value][sub_key] = record_name_value_pairs

        control = visual_config.pivot_varible_control_value
        if control in group_by_pivot and len(group_by_pivot.keys()) != 2:
            raise RuntimeError("cannot use more than 2-values variable as pivot")

        povot_value_pool = list(group_by_pivot.keys())
        povot_value_pool.remove(control)
        treatment = povot_value_pool[0]

        # prepare the result headers, including top header (category for columns) + sub header (detailed columns)
        sub_header = []
        statistic_names = []
        diffable_statistic_names = []
        for record_name in cur_benchmark_stat.record_names:
            if (record_name._is_variable or record_name._is_input) and record_name._name != visual_config.pivot_variable_name:
                sub_header.append(record_name._name)

            if record_name._is_statistic:
                if record_name._is_diffable:
                    diffable_statistic_names.append(record_name._name)
                statistic_names.append(record_name._name)


        header = len(sub_header) * ['input & run config']
        sub_header += statistic_names # for control group
        header += len(statistic_names) * ['control group - {} ({})'.format(control, unit)]
        sub_header += statistic_names # for treatment group
        header += len(statistic_names) * ['treatment group - {} ({})'.format(treatment, unit)]
        sub_header += diffable_statistic_names # for diff group
        header += len(diffable_statistic_names) * ['diff']

        # prepare the diff table body
        updated_data = []
        for sub_key, record_name_value_pairs in group_by_pivot[control].items():
            updated_row = []
            diff_control_colums = []
            for record_name, record_value in record_name_value_pairs:
                if (record_name._is_variable or record_name._is_input) \
                    and record_name._name != visual_config.pivot_variable_name:
                        value_str = record_value._value
                        if isinstance(record_value._value, np.ndarray):
                            value_str = npyarray2string(record_value._value)
                        updated_row.append(value_str)

            for record_name, record_value in record_name_value_pairs:
                if record_name._is_statistic:
                    if record_name._is_diffable:
                        diff_control_colums.append(record_value._value)
                    updated_row.append(record_value._value)

            diff_treatment_colums = []
            for record_name, record_value in group_by_pivot[treatment][sub_key]:
                if record_name._is_statistic:
                    if record_name._is_diffable:
                        diff_treatment_colums.append(record_value._value)
                    updated_row.append(record_value._value)


            from operator import truediv
            sub_ret = list(map(truediv, diff_treatment_colums, diff_control_colums))
            updated_row += ["{0:.3%}".format(s - 1.0) for s in sub_ret]
            updated_data.append(updated_row)

        index = [
            np.array(header),
            np.array(sub_header)
        ]

        df = pd.DataFrame(data=updated_data, columns=index)

    else:
        header = []
        for record_name in cur_benchmark_stat.record_names:
            new_record_name = record_name._name + ' ({})'.format(unit) if record_name._is_statistic and record_name._is_diffable else record_name._name
            header.append(new_record_name)

        updated_data = []
        for record_name_value_pairs in cur_benchmark_stat.iterator():
            updated_row = []
            for record_name, record_value in record_name_value_pairs:
                value_str = record_value._value
                if isinstance(record_value._value, np.ndarray):
                    value_str = npyarray2string(record_value._value)
                updated_row.append(value_str)
            updated_data.append(updated_row)

        index = [
            np.array(header)
        ]

        df = pd.DataFrame(data=updated_data, columns=index)

    print('=' * 100, "\nstats from {}: \n".format(full_filename), df.to_string(index=False, justify='left'))

def show_benchmark_reports(bench_reports):
    for directory, visual_config in bench_reports:
        for filename in os.listdir(directory):
            if not filename.endswith('.pkl'):
                continue

            full_filename = os.path.join(directory, filename)
            run_stats = load_stats(full_filename)

            for cur_benchmark_stat in run_stats:
                show_benchmark_report(cur_benchmark_stat, visual_config, full_filename)
