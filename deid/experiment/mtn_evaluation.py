import math
import os
import pickle
import random
import sys

import numpy as np
from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from keras.layers import Input

from . import experiment_directory
from ..data import TrainingSet, ValidationSet, StratifiedSampling, is_phi_sentence
from ..data.augment import Augment, get as get_strategy
from ..data.batch import IteratorWithEpochLength
from ..data.util import pad_2d_sequences
from ..embeddings import Matrix, get as get_embeddings
from ..env import env
from ..model.adversary import TwoRepresentationsAreSameOriginalDiscriminator


def fake_augmented_sentences_batch(X, y, indices, augm_alternatives, fake_alternatives, split_condition):
    indices = [i for i in indices if split_condition(X[i], y[i])]
    real_sentences = [X[i] for i in indices]
    augmented_sentences = [augm_alternatives[ind][0] for ind in indices]
    fake_sentences = [random.choice(fake_alternatives[ind]) for ind in indices]

    X_1 = []
    X_2 = []
    y = []
    for real, augm, fake in zip(real_sentences, augmented_sentences, fake_sentences):
        X_1 += [augm, augm]
        X_2 += [real, fake]
        y += [1, 0]

    return pad_2d_sequences(X_1), pad_2d_sequences(X_2), np.array(y)


class MTNGenerator(IteratorWithEpochLength):
    def __init__(self, generator: IteratorWithEpochLength, dataset, dataset2):
        self.generator = generator
        self.dataset = dataset
        self.dataset2 = dataset2

    def __next__(self):
        _, _, indices = next(self.generator)
        X_1, X_2, adv_y = fake_augmented_sentences_batch(self.dataset.X, self.dataset.y, indices,
                                                         self.dataset.augmented, self.dataset2.augmented,
                                                         split_condition=is_phi_sentence)
        return [X_1, X_2], adv_y

    @property
    def epoch_length(self) -> int:
        return self.generator.epoch_length


def mtn_evaluation_experiment(config):
    print('Loading embeddings...')
    embeddings = get_embeddings(config['experiment']['embeddings'])

    name = config['name']
    experiment_dir = experiment_directory(name, config['path'])

    print('Loading matrix...')
    matrix = Matrix(embeddings, precomputed_word2ind=embeddings.precomputed_word2ind,
                    precomputed_matrix=embeddings.precomputed_matrix)

    strategy = get_strategy(config['augment']['strategy'], matrix)
    digit_strategy = get_strategy(config['augment']['digit_strategy'], matrix)
    adv_strategy = get_strategy('move_to_neighbor-5', matrix)

    augment = Augment(embeddings, strategy=strategy, digit_strategy=digit_strategy, n_augmentations=1)

    augment2 = Augment(embeddings, strategy=adv_strategy, digit_strategy=digit_strategy,
                       n_augmentations=config['augment']['n_augmentations'], augment_max=1)

    print('Augmenting training set...', flush=True)
    tr = TrainingSet(train_set=config['experiment']['train_set'],
                     embeddings=embeddings,
                     use_short_sentences=env.use_short_sentences,
                     limit_documents=env.limit_training_documents,
                     augment=augment)

    tr2 = TrainingSet(train_set=config['experiment']['train_set'],
                      embeddings=embeddings,
                      use_short_sentences=env.use_short_sentences,
                      limit_documents=env.limit_training_documents,
                      augment=augment2)

    assert np.all(tr.X[100] == tr2.X[100])  # making sure that the training sets have the same order

    print('Augmenting validation set...', flush=True)
    val = ValidationSet(validation_set=config['experiment']['validation_set'],
                        embeddings=embeddings,
                        label2ind=tr.label2ind,
                        use_short_sentences=env.use_short_sentences,
                        limit_documents=env.limit_validation_documents,
                        augment=augment)

    val2 = ValidationSet(validation_set=config['experiment']['validation_set'],
                         embeddings=embeddings,
                         label2ind=tr.label2ind,
                         use_short_sentences=env.use_short_sentences,
                         limit_documents=env.limit_validation_documents,
                         augment=augment2)

    inputs = {'train_representation': Input(shape=(None, embeddings.size)),
              'fake_representation': Input(shape=(None, embeddings.size))}
    adversary = TwoRepresentationsAreSameOriginalDiscriminator(inputs, representation_size=embeddings.size,
                                                               lstm_size=embeddings.size)
    adversary.model.compile(loss=adversary.loss, optimizer='nadam', metrics=['accuracy'])

    batch_size = test_batch_size = 32
    train_gen = MTNGenerator(StratifiedSampling(tr.X, tr.y, split_condition=is_phi_sentence,
                                                batch_size=batch_size, yield_indices=True, shuffle=True), tr, tr2)
    valid_gen = MTNGenerator(StratifiedSampling(val.X, val.y, split_condition=is_phi_sentence,
                                                batch_size=test_batch_size, yield_indices=True, shuffle=False), val,
                             val2)

    early_stopping = EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience'])
    flush = LambdaCallback(on_epoch_end=lambda epoch, logs: sys.stdout.flush())
    callbacks = [early_stopping, flush]
    if env.save_model:
        checkpoint = ModelCheckpoint(os.path.join(experiment_dir, 'model.hdf5'), save_best_only=True)
        callbacks.append(checkpoint)

    history = adversary.model.fit_generator(train_gen,
                                            epochs=config['training']['train_epochs'],
                                            steps_per_epoch=int(math.ceil(len(tr.X) / batch_size)),
                                            validation_data=valid_gen,
                                            validation_steps=int(math.ceil(len(val.X) / test_batch_size)),
                                            callbacks=callbacks,
                                            verbose=env.keras_verbose)

    if config['test']['run_test']:
        label2ind = tr.label2ind
        del tr, tr2, val, val2, train_gen, valid_gen

        if env.save_model:
            print('Restoring best weights')
            adversary.model.load_weights(os.path.join(experiment_dir, 'model.hdf5'))

        print('Augmenting test set...', flush=True)

        test = ValidationSet(validation_set='test',
                             embeddings=embeddings,
                             label2ind=label2ind,
                             use_short_sentences=env.use_short_sentences,
                             limit_documents=env.limit_validation_documents,
                             augment=augment)

        test2 = ValidationSet(validation_set='test',
                              embeddings=embeddings,
                              label2ind=label2ind,
                              use_short_sentences=env.use_short_sentences,
                              limit_documents=env.limit_validation_documents,
                              augment=augment2)
        test_gen = MTNGenerator(StratifiedSampling(test.X, test.y, split_condition=is_phi_sentence,
                                                   batch_size=test_batch_size, yield_indices=True, shuffle=False), test,
                                test2)

        loss, acc = adversary.model.evaluate_generator(test_gen, int(math.ceil(len(test.X) / test_batch_size)))
        print(f'Test loss: {loss}, test acc: {acc}')
        history.history['test_loss'] = loss
        history.history['test_acc'] = acc

    history_pickle_path = os.path.join(experiment_dir, 'history.pickle')
    print('Saving history to', history_pickle_path)
    with open(history_pickle_path, 'wb') as f:
        pickle.dump(history.history, f)
