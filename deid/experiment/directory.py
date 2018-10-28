import os
import shutil
import socket
from datetime import datetime

from ..env import env


def experiment_directory(name, config_path=None, work_dir=env.work_dir):
    """ Creates a directory for the experiment

    :param name:
    :param config_path:
    :param work_dir:
    :return:
    """
    date_str = datetime.now().strftime('%Y%m%d-%H%M%S')
    directory = os.path.join(work_dir, name + '_' + socket.gethostname() + '_' + date_str)
    if env.experiment_dir_postfix is not None:
        directory += '_' + env.experiment_dir_postfix
    os.mkdir(directory)
    if config_path is not None:
        shutil.copy2(config_path, directory)

    return directory
