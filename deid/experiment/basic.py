import math
import os
import pickle
import sys

import numpy as np
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint

from . import DeidentificationEvaluationCallback, evaluate_deid_performance, experiment_directory
from ..data import TrainingSet, ValidationSet, BatchGeneratorWithExtraFeatures, StratifiedSamplingWithExtraFeatures, \
    is_phi_sentence
from ..data.augment import Augment, get as get_strategy
from ..data.class_weight import get as get_class_weight
from ..data.feature import get as get_feature
from ..data.util import compounding
from ..embeddings import PrecomputedEmbeddings, Matrix, get as get_embeddings
from ..env import env
from ..model import get as get_model
from ..model.optimizer import get as get_optimizer


def basic_experiment(config):
    name = config['name']
    batch_size = config['training']['batch_size']
    test_batch_size = config['training']['test_batch_size']
    if test_batch_size is None:
        test_batch_size = batch_size
    test_weights = config['test']['test_weights']

    experiment_dir = experiment_directory(name, config['path'])

    print('Loading embeddings...')
    embeddings = get_embeddings(config['experiment']['embeddings'])
    print('Done.')

    if config['augment'] is not None and test_weights is None:
        if isinstance(embeddings, PrecomputedEmbeddings):
            matrix = Matrix(embeddings, precomputed_word2ind=embeddings.precomputed_word2ind,
                            precomputed_matrix=embeddings.precomputed_matrix)
            strategy_kwargs = {'matrix': matrix}
        else:
            strategy_kwargs = {}

        strategy = get_strategy(config['augment']['strategy'], **strategy_kwargs)
        digit_strategy = get_strategy(config['augment']['digit_strategy'], **strategy_kwargs)
        augment = Augment(embeddings=embeddings, strategy=strategy, digit_strategy=digit_strategy,
                          **config['augment']['augment_args'])
    else:
        augment = None

    if config['experiment']['extra_features'] is None or len(config['experiment']['extra_features']) == 0:
        extra_features = []
    else:
        extra_features = [get_feature(identifier) for identifier in config['experiment']['extra_features']]

    tr = TrainingSet(train_set=config['experiment']['train_set'],
                     embeddings=embeddings,
                     use_short_sentences=env.use_short_sentences,
                     limit_documents=env.limit_training_documents,
                     binary_classification=config['experiment']['binary_classification'],
                     hipaa_only=config['experiment']['hipaa_only'],
                     augment=augment,
                     extra_features=extra_features)

    model = get_model(config['experiment']['model'])(name=name,
                                                     input_size=embeddings.size,
                                                     extra_input_size=tr.X_extra_size,
                                                     output_size=tr.output_size,
                                                     optimizer=get_optimizer(config['training']['optimizer'])(
                                                         **config['training']['optimizer_args']),
                                                     **config['model_args'])

    if test_weights is None:
        train_and_validate(model, config, tr, embeddings, extra_features, batch_size, test_batch_size, experiment_dir)
    else:
        model.load_weights(test_weights)

    if config['test']['run_test']:
        test_set = config['test']['test_set']
        if test_set is None:
            test_set = 'test'
        evaluate_deid_performance(model=model, batch_size=test_batch_size, embeddings=embeddings,
                                  label2ind=tr.label2ind, ind2label=tr.ind2label,
                                  test_set=test_set, experiment_dir=experiment_dir,
                                  binary_classification=config['experiment']['binary_classification'],
                                  hipaa_only=config['experiment']['hipaa_only'],
                                  extra_features=extra_features, epoch=99)


def train_and_validate(model, config, tr, embeddings, extra_features, batch_size, test_batch_size, experiment_dir):
    val = ValidationSet(validation_set=config['experiment']['validation_set'],
                        embeddings=embeddings,
                        label2ind=tr.label2ind,
                        use_short_sentences=env.use_short_sentences,
                        limit_documents=env.limit_validation_documents,
                        binary_classification=config['experiment']['binary_classification'],
                        hipaa_only=config['experiment']['hipaa_only'],
                        extra_features=extra_features)

    if config['augment'] is not None and config['augment']['include_original']:
        tr_X, tr_y, tr_X_extra = tr.data_with_augmented
        augment_training_generator = None
    else:
        tr_X, tr_y, tr_X_extra = tr.X, tr.y, tr.X_extra
        augment_training_generator = tr.augmented

    print('Size of the training set:', len(tr_X), 'with maxlen:', tr.maxlen)
    compound = config['training']['batch_size_compound']
    if compound is not None and compound != 0 and compound < batch_size:
        training_batch_size = compounding(1, batch_size, compound)
    else:
        training_batch_size = batch_size

    if config['training']['batch_mode'] == 'stratified':
        train_gen_class, train_gen_args = StratifiedSamplingWithExtraFeatures, {'split_condition': is_phi_sentence}
    else:
        train_gen_class, train_gen_args = BatchGeneratorWithExtraFeatures, {}

    training_generator = train_gen_class(tr_X, tr_y, tr_X_extra,
                                         batch_size=training_batch_size,
                                         augment=augment_training_generator, **train_gen_args)

    validation_generator = BatchGeneratorWithExtraFeatures(val.X, val.y, val.X_extra, test_batch_size,
                                                           shuffle=False)

    if config['experiment']['class_weight'] is not None:
        class_weight = get_class_weight(config['experiment']['class_weight'])(tr.output_size, tr_y)
    else:
        class_weight = None

    early_stopping = EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience'])
    flush = LambdaCallback(on_epoch_end=lambda epoch, logs: sys.stdout.flush())
    evaluation = DeidentificationEvaluationCallback(model, batch_size=test_batch_size, embeddings=embeddings,
                                                    label2ind=tr.label2ind, ind2label=tr.ind2label,
                                                    test_set=config['experiment']['validation_set'],
                                                    experiment_dir=experiment_dir,
                                                    evaluate_every=config['training']['i2b2_evaluate_every'],
                                                    binary_classification=config['experiment'][
                                                        'binary_classification'],
                                                    hipaa_only=config['experiment']['hipaa_only'],
                                                    extra_features=extra_features)

    callbacks = [early_stopping, evaluation, flush]
    if env.save_model:
        checkpoint = ModelCheckpoint(os.path.join(experiment_dir, 'model.hdf5'), save_best_only=True)
        callbacks.append(checkpoint)

    history = model.fit_generator(training_generator,
                                  epochs=config['training']['train_epochs'],
                                  steps_per_epoch=int(math.ceil(len(tr_X) / batch_size)),
                                  validation_data=validation_generator,
                                  validation_steps=int(math.ceil(len(val.X) / test_batch_size)),
                                  class_weight=class_weight,
                                  callbacks=callbacks,
                                  verbose=env.keras_verbose,
                                  use_multiprocessing=True)
    if env.save_model:
        best_epoch = np.argmin(history.history['val_loss']) + 1  # epoch numbering is 1-based
        print(f'Resetting to weights from epoch {best_epoch:02d}')
        model.load_weights(os.path.join(experiment_dir, 'model.hdf5'))

    history_pickle_path = os.path.join(experiment_dir, 'history.pickle')
    print('Saving history to', history_pickle_path)
    with open(history_pickle_path, 'wb') as f:
        pickle.dump(history.history, f)
