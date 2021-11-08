from collections import OrderedDict
import os

def get_functionals():
    from torch.nn import functional
    from inspect import getmembers, isfunction
    functional_list = [item[0] for item in getmembers(functional, isfunction)]
    return functional_list

functionals = get_functionals()

replacing_map = OrderedDict([
    (full_name, key) for key in functionals for full_name in [
        f'torch.nn.functional.{key}',
        f'F.{key}',
        f'nn.functional.{key}',
        f'functional.{key}'
    ]])
print(replacing_map)

def rename_functional(name):
    prefix = 'UWrapped' if name.startswith('_') else 'Wrapped'
    new_name = ''.join([n.capitalize() for n in name.split('_')])
    if name.endswith('_'): new_name += 'I'
    return prefix + new_name

def wrap_functional(func):
    return f'''class {rename_functional(func)}(nn.Module):
    def forward(self, *args, **kwargs):
        return F.{func}(*args, **kwargs)
'''

def create_wrapped_functions_file(file_path):
    with open(file_path, 'w') as f:
        f.write('from torch import nn\n')
        f.write('import torch.nn.functional as F\n\n')
        for func in functionals:
            f.write(wrap_functional(func))
            f.write('\n')

def replacing(original):
    for old, new in replacing_map.items():
        new = rename_functional(new) + '()'
        original = original.replace(old, new)
    return original

def replace_files(files, path):
    for file_path in files:
        with open(file_path) as f:
            contents = f.readlines()
        with open(file_path, 'w') as g:
            g.write('import sys\n')
            g.write(f'sys.path.append("{path}")\n')
            g.write('from wrapped_functional import *\n\n')
            for line in contents:
                new_line = replacing(line)
                g.write(new_line)

def list_py_files(path):
    def list_dir(p):
        if os.path.isfile(p):
            return [p] if p.endswith('.py') else []
        else:
            files = []
            for d in os.listdir(p):
                full_path = os.path.join(p, d)
                files.extend(list_dir(full_path))
            return files
    return list_dir(path)

def main(orig_path, wrapped_path):
    files = list_py_files(orig_path)
    path = orig_path if os.path.isdir(orig_path) else os.path.split(orig_path)[0]
    replace_files(files, path)
    create_wrapped_functions_file(os.path.join(path, wrapped_path))


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--file_path', type=str, required=True, help='original file/folder')
    arg_parser.add_argument('--wrapped', type=str, default='wrapped_functional.py',
                            help='generated wrapped functional file name')

    file_path = arg_parser.file_path
    wrapped_file_name = arg_parser.wrapped
    # file_path = '/home/linmin/result/workspace/generation/prophetnet'
    # wrapped_file_name = 'wrapped_functional.py'

    main(file_path, wrapped_file_name)
