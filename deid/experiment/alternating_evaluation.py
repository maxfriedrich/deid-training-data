import math
import os
import pickle
import sys

import numpy as np
from keras.callbacks import EarlyStopping, LambdaCallback
from keras.utils.generic_utils import Progbar

from .alternating import alternating_experiment
from ..env import env


def make_progress_bar(target):
    return Progbar(target=target, verbose=env.keras_verbose)


def alternating_evaluation_experiment(config):
    weights = config['test']['test_weights']
    model, tr, train_gen, val, valid_gen, experiment_dir = alternating_experiment(config, run_experiment=False)

    model.complete_model.load_weights(weights)

    batch_size = config['training']['batch_size']
    test_batch_size = config['training']['test_batch_size']
    if test_batch_size is None:
        test_batch_size = batch_size

    early_stopping = EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience'])
    flush = LambdaCallback(on_epoch_end=lambda epoch, logs: sys.stdout.flush())

    before_fine_tuning_weights = model.train_representer.get_weights()

    def assert_fixed_weights():
        after_fine_tuning_weights = model.train_representer.get_weights()
        for i in range(len(before_fine_tuning_weights)):
            assert np.all(before_fine_tuning_weights[i] == after_fine_tuning_weights[i])

    assert_fixed_representer = LambdaCallback(on_epoch_end=lambda epoch, logs: assert_fixed_weights())
    callbacks = [early_stopping, flush, assert_fixed_representer]

    print('Training adversary')
    history = model.pretrain_adversary.fit_generator(train_gen,
                                                     epochs=config['training']['train_epochs'],
                                                     steps_per_epoch=int(math.ceil(len(tr.X) / batch_size)),
                                                     validation_data=valid_gen,
                                                     validation_steps=int(math.ceil(len(val.X) / test_batch_size)),
                                                     callbacks=callbacks,
                                                     verbose=env.keras_verbose)

    history_pickle_path = os.path.join(experiment_dir, 'history.pickle')
    print('Saving history to', history_pickle_path)
    with open(history_pickle_path, 'wb') as f:
        pickle.dump(history.history, f)
