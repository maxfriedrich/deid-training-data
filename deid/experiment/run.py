from . import get_config, get as get_experiment


def run_experiment(config_name_or_path):
    config = get_config(config_name_or_path)
    experiment = get_experiment(config['experiment']['type'])
    experiment(config)
