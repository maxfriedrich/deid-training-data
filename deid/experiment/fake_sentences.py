import math
import os
import pickle
import random

import numpy as np
from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense

from . import experiment_directory
from ..data import TrainingSet, ValidationSet, StratifiedSampling, is_phi_sentence
from ..data.augment import Augment, get as get_strategy
from ..data.batch import IteratorWithEpochLength
from ..data.util import pad_2d_sequences
from ..embeddings import Matrix, get as get_embeddings
from ..env import env


def real_and_fake_sentences(X, y, indices, alternatives, split_condition):
    indices = [i for i in indices if split_condition(X[i], y[i])]
    real_sentences = [X[i] for i in indices]
    fake_sentences = [random.choice(alternatives[ind]) for ind in indices]

    X = []
    y = []
    for real, fake in zip(real_sentences, fake_sentences):
        X += [real, fake]
        y += [1, 0]

    return pad_2d_sequences(X), np.array(y)


class FakeSentencesGenerator(IteratorWithEpochLength):
    def __init__(self, generator: IteratorWithEpochLength, dataset):
        self.generator = generator
        self.dataset = dataset

    def __next__(self):
        _, _, indices = next(self.generator)
        X, y = real_and_fake_sentences(self.dataset.X, self.dataset.y, indices, self.dataset.augmented,
                                       split_condition=is_phi_sentence)
        return X, y

    @property
    def epoch_length(self) -> int:
        return self.generator.epoch_length


def fake_sentences_experiment(config):
    print('Loading embeddings...')
    embeddings = get_embeddings(config['experiment']['embeddings'])

    name = config['name']
    experiment_dir = experiment_directory(name, config['path'])

    print('Loading matrix...')
    matrix = Matrix(embeddings, precomputed_word2ind=embeddings.precomputed_word2ind,
                    precomputed_matrix=embeddings.precomputed_matrix)

    strategy = get_strategy(config['augment']['strategy'], matrix)
    digit_strategy = get_strategy(config['augment']['digit_strategy'], matrix)
    augment = Augment(embeddings, strategy=strategy, digit_strategy=digit_strategy,
                      **config['augment']['augment_args'])

    print('Augmenting training set...', flush=True)
    tr = TrainingSet(embeddings=embeddings,
                     train_set=config['experiment']['train_set'],
                     use_short_sentences=env.use_short_sentences,
                     limit_documents=env.limit_training_documents,
                     augment=augment)

    print('Augmenting validation set...', flush=True)
    val = ValidationSet(embeddings=embeddings,
                        validation_set=config['experiment']['validation_set'],
                        label2ind=tr.label2ind,
                        use_short_sentences=env.use_short_sentences,
                        limit_documents=env.limit_validation_documents,
                        augment=augment)

    model = Sequential()
    model.add(Bidirectional(LSTM(embeddings.size), input_shape=(None, embeddings.size)))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

    batch_size = test_batch_size = 32
    train_gen = FakeSentencesGenerator(StratifiedSampling(tr.X, tr.y, split_condition=is_phi_sentence,
                                                          batch_size=batch_size, yield_indices=True, shuffle=True), tr)
    valid_gen = FakeSentencesGenerator(StratifiedSampling(val.X, val.y, split_condition=is_phi_sentence,
                                                          batch_size=batch_size, yield_indices=True, shuffle=False),
                                       val)

    history = model.fit_generator(train_gen,
                                  epochs=config['training']['train_epochs'],
                                  steps_per_epoch=int(math.ceil(len(tr.X) / batch_size)),
                                  validation_data=valid_gen,
                                  validation_steps=int(math.ceil(len(val.X) / test_batch_size)),
                                  verbose=env.keras_verbose)

    history_pickle_path = os.path.join(experiment_dir, 'history.pickle')
    print('Saving history to', history_pickle_path)
    with open(history_pickle_path, 'wb') as f:
        pickle.dump(history.history, f)
