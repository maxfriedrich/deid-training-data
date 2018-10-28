import argparse
import itertools
import os
import random

import yaml

from ..env import env

config_dir = os.path.join(env.resources_dir, 'config')
generated_dir = os.path.join(config_dir, 'generated')
if not os.path.isdir(generated_dir):
    os.mkdir(generated_dir)


def generate_config(config):
    result = {}
    for key, value in config.items():
        if key == 'choice':
            if isinstance(value, list):
                return random.choice(value)
            else:
                raise ValueError('does not support other inputs than lists at the moment')
        elif isinstance(value, dict):
            result[key] = generate_config(value)
        else:
            result[key] = value
    return result


def generate_random_configs(config, name, n, start, output_path):
    for i in range(n):
        config_num = i + start
        result = generate_config(config)
        with open(os.path.join(output_path, name) + f'_{config_num:03d}.yaml', 'w') as f:
            f.write(yaml.dump(result))
    print(f'Generated {n} configs.')


def flatten_config(config, sep='--', prefix=None):
    """
    >>> flatten_config({'a': 1, 'b': {'c': {'d': 4, 'e': [5, 6]}}})
    {'a': 1, 'b--c--d': 4, 'b--c--e': [5, 6]}
    """
    result = {}
    for key, value in config.items():
        key = key if prefix is None else f'{prefix}{sep}{key}'
        if isinstance(value, dict):
            result.update(flatten_config(value, sep, prefix=key))
        else:
            result[key] = value
    return result


def unflatten_config(config, sep='--'):
    """
    >>> unflatten_config({'a': 1, 'b--c--d': 4, 'b--c--e': [5, 6]})
    {'a': 1, 'b': {'c': {'d': 4, 'e': [5, 6]}}}
    """
    result = {}
    for key, value in config.items():
        parts = key.split(sep)
        parent = result
        for child in parts[:-1]:
            if child not in parent.keys():
                parent[child] = {}
            parent = parent[child]
        parent[parts[-1]] = value
    return result


def remove_choices(config, sep='--'):
    """
    >>> remove_choices({'a': 1, 'b--c--choice': 2})
    {'a': 1, 'b--c': 2}
    """
    result = {}
    for key, value in config.items():
        if key.endswith(f'{sep}choice'):
            result[key[:-len(f'{sep}choice')]] = value
        else:
            result[key] = value
    return result


def generate_grid_configs(config, name, output_path):
    flattened = flatten_config(config)
    choice_keys = [key for key in flattened.keys() if key.endswith('choice')]

    for i, choices in enumerate(itertools.product(*[flattened[key] for key in choice_keys]), start=1):
        for choice_key, choice in zip(choice_keys, choices):
            flattened[choice_key] = choice

        result = unflatten_config(remove_choices(flattened))
        with open(os.path.join(output_path, name) + f'_grid_{i:03d}.yaml', 'w') as f:
            f.write(yaml.dump(result))
    print(f'Generated {i} configs.')


def find_config(name):
    filename = name
    if os.path.isfile(filename):
        return filename

    filename = os.path.join(config_dir, filename)
    if os.path.isfile(filename):
        return filename

    filename = filename + '.yaml'
    if os.path.isfile(filename):
        return filename

    raise argparse.ArgumentTypeError(f'{name} is not a valid config name or path')


def main():
    def ensure_dir(arg) -> str:
        if type(arg) == str and os.path.isdir(arg):
            return arg
        raise argparse.ArgumentTypeError(f'{arg} is not a directory')

    parser = argparse.ArgumentParser()
    parser.description = 'Create experiment configs from a config template.'
    parser.add_argument('input_config', help='the input config template')
    parser.add_argument('-o', '--output_path', help='the path to store the results', type=ensure_dir,
                        default=generated_dir)
    parser.add_argument('-n', '--n', help='the number of configs to generate', type=int, default=10)
    parser.add_argument('-a', '--all', help='generate all configs (grid), overrides --n', action='store_true')
    parser.add_argument('-s', '--start', help='the starting number for config filenames', type=int, default=0)

    args = parser.parse_args()

    filename = find_config(args.input_config)
    config = yaml.load(open(filename))
    name = '.'.join(os.path.basename(filename).split('.')[:-1])
    name = name.replace('_template', '')
    if args.all:
        generate_grid_configs(config, name, args.output_path)
    else:
        generate_random_configs(config, name, args.n, args.start, args.output_path)


if __name__ == '__main__':
    main()
