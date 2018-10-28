import os
import sys

import yaml

from ..env import env

config_dir = os.path.join(env.resources_dir, 'config')


class Config(dict):
    """ A dict that returns None for missing items instead of raising an exception, including for child dicts """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if k == 'choice':
                raise ValueError('This is a config template, not an experiment config. Please generate configs from '
                                 'it with python -m deid.tools.config')
            # please don't put a dict into itself (can't happen when importing from yaml anyway)
            if isinstance(v, dict):
                self[k] = Config(v)

    def __getitem__(self, key):
        if key.endswith('_args'):
            return self.get(key, {})
        return self.get(key)


def get_config(name):
    if os.path.isfile(name):
        return load_config_yaml(name)

    for parent in [config_dir, os.path.join(config_dir, 'generated')]:
        filename = os.path.join(parent, name)
        if os.path.isfile(filename):
            return load_config_yaml(filename)

        filename = filename + '.yaml'
        if os.path.isfile(filename):
            return load_config_yaml(filename)

    raise ValueError(f'Could not locate config "{name}" in config dir')


def load_config_yaml(path):
    config = Config(yaml.load(open(path)))
    config['name'] = '.'.join(os.path.basename(path).split('.')[:-1])
    config['path'] = path
    sys.stderr.write(f"Using {config['name']} config.\n")
    return config
