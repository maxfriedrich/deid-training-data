import os
import pickle
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils.generic_utils import Progbar

from . import evaluate_deid_performance, experiment_directory
from ..data import TrainingSet, ValidationSet, StratifiedSampling, is_phi_sentence, fake_sentences_batch
from ..data.augment import Augment, get as get_strategy
from ..data.batch import BatchGeneratorWithExtraFeatures, IteratorWithEpochLength
from ..data.class_weight import get as get_class_weight
from ..data.feature import get as get_feature
from ..data.util import compounding, pad_2d_sequences
from ..embeddings import Matrix, PrecomputedEmbeddings, FastTextEmbeddings, get as get_embeddings
from ..env import env
from ..model import AdversarialModel


def make_progress_bar(target):
    return Progbar(target=target, verbose=env.keras_verbose)


class AdversaryGenerator(IteratorWithEpochLength):
    def __init__(self, generator: IteratorWithEpochLength, dataset):
        self.generator = generator
        self.dataset = dataset

    def __next__(self):
        _, _, indices = next(self.generator)
        X_1, X_2, adv_y = fake_sentences_batch(self.dataset.X, self.dataset.y, indices, self.dataset.augmented,
                                               split_condition=is_phi_sentence)
        return [X_1, X_2], adv_y

    @property
    def epoch_length(self) -> int:
        return self.generator.epoch_length


class CombinedGenerator(IteratorWithEpochLength):
    def __init__(self, generator, dataset):
        self.generator = generator
        self.dataset = dataset

    def __next__(self):
        X, y, indices = next(self.generator)
        X_extra = pad_2d_sequences([self.dataset.X_extra[i] for i in indices])
        X_1, X_2, adv_y = fake_sentences_batch(self.dataset.X, self.dataset.y, indices, self.dataset.augmented,
                                               split_condition=is_phi_sentence)
        return [X, X_extra, X_1, X_2], [y, adv_y]

    @property
    def epoch_length(self) -> int:
        return self.generator.epoch_length


class MainModelCheckpoint(ModelCheckpoint):
    def __init__(self, main_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_model = main_model

    @property
    def model(self):
        return self.main_model

    @model.setter
    def model(self, value):
        pass


class Flush(Callback):
    def on_epoch_end(self, epoch, logs=None):
        sys.stdout.flush()


class StopAfterEveryEpoch(Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.stop_training = True


def save_history(history, experiment_dir):
    history_pickle_path = os.path.join(experiment_dir, 'history.pickle')
    with open(history_pickle_path, 'wb') as f:
        pickle.dump(history, f)


def alternating_experiment(config, run_experiment=True):
    model_args = config['model_args']
    if model_args['adversaries'] is None or not model_args['adversaries'][0].startswith('discriminate'):
        raise ValueError(f'config does not have an adversary mode that starts with "discriminate"')

    print('Loading embeddings...')
    lookup_embeddings = get_embeddings(config['experiment']['embeddings'])
    if isinstance(lookup_embeddings, PrecomputedEmbeddings):
        matrix_embeddings = lookup_embeddings
    else:
        matrix_embeddings = FastTextEmbeddings()

    name = config['name']
    experiment_dir = experiment_directory(name, config['path'])

    print('Loading matrix...')
    matrix = Matrix(matrix_embeddings, precomputed_word2ind=matrix_embeddings.precomputed_word2ind,
                    precomputed_matrix=matrix_embeddings.precomputed_matrix)

    strategy = get_strategy(config['augment']['strategy'], matrix)
    digit_strategy = get_strategy(config['augment']['digit_strategy'], matrix)
    augment = Augment(lookup_embeddings, strategy=strategy, digit_strategy=digit_strategy,
                      **config['augment']['augment_args'])

    if config['experiment']['extra_features'] is None or len(config['experiment']['extra_features']) == 0:
        extra_features = []
    else:
        extra_features = [get_feature(identifier) for identifier in config['experiment']['extra_features']]

    print('Augmenting training set...', flush=True)
    tr = TrainingSet(train_set=config['experiment']['train_set'],
                     embeddings=lookup_embeddings,
                     use_short_sentences=env.use_short_sentences,
                     limit_documents=env.limit_training_documents,
                     augment=augment,
                     binary_classification=config['experiment']['binary_classification'],
                     hipaa_only=config['experiment']['hipaa_only'],
                     extra_features=extra_features)

    model = AdversarialModel(embedding_size=lookup_embeddings.size,
                             output_size=tr.output_size,
                             extra_input_size=tr.X_extra_size,
                             optimizer=config['training']['optimizer'],
                             optimizer_args=config['training']['optimizer_args'],
                             **config['model_args'])

    print('Augmenting validation set...', flush=True)
    val = ValidationSet(validation_set=config['experiment']['validation_set'],
                        embeddings=lookup_embeddings,
                        label2ind=tr.label2ind,
                        use_short_sentences=env.use_short_sentences,
                        limit_documents=env.limit_validation_documents,
                        augment=augment,
                        binary_classification=config['experiment']['binary_classification'],
                        hipaa_only=config['experiment']['hipaa_only'],
                        extra_features=extra_features)

    del matrix

    print('Size of the training set:', len(tr.X), 'with maxlen:', tr.maxlen)

    batch_size = config['training']['batch_size']
    test_batch_size = config['training']['test_batch_size']
    if test_batch_size is None:
        test_batch_size = batch_size

    compound = config['training']['batch_size_compound']
    if compound is not None and compound != 0 and compound < batch_size:
        training_batch_size = compounding(1, batch_size, compound)
    else:
        training_batch_size = batch_size

    if config['experiment']['class_weight'] is not None:
        class_weight = get_class_weight(config['experiment']['class_weight'])(tr.output_size, tr.y)
    else:
        class_weight = None

    history = {}

    pretrain_weights = config['training']['pretrain_weights']
    if pretrain_weights is not None:
        print('Loading pretrain weights')
        model.complete_model.load_weights(pretrain_weights)

    # (1) Train the representation model and de-identifier jointly
    print('(1) Pre-training de-identifier', flush=True)
    history['deid_pretrain'] = pretrain_deidentifier(config=config,
                                                     experiment_dir=experiment_dir,
                                                     model=model,
                                                     training_set=tr,
                                                     validation_set=val,
                                                     training_batch_size=training_batch_size,
                                                     validation_batch_size=test_batch_size,
                                                     class_weight=class_weight)
    save_history(history, experiment_dir)

    # (2) Freeze the representation model and train the adversaries
    print('(2) Pre-training adversary', flush=True)
    history[f'adv_pretrain'] = pretrain_adversary(config=config,
                                                  experiment_dir=experiment_dir,
                                                  model=model,
                                                  train_set=tr,
                                                  training_batch_size=training_batch_size,
                                                  validation_set=val,
                                                  validation_batch_size=test_batch_size)
    save_history(history, experiment_dir)

    # (3) Alternate training between branches and representation
    print('(3) Alternating training', flush=True)
    train_gen = CombinedGenerator(
        StratifiedSampling(tr.X, tr.y, batch_size=training_batch_size, split_condition=is_phi_sentence,
                           yield_indices=True, shuffle=True), tr)
    valid_gen = CombinedGenerator(
        StratifiedSampling(tr.X, tr.y, batch_size=training_batch_size, split_condition=is_phi_sentence,
                           yield_indices=True, shuffle=False), tr)

    if not run_experiment:
        train_gen = AdversaryGenerator(StratifiedSampling(tr.X, tr.y, split_condition=is_phi_sentence,
                                                          batch_size=training_batch_size, yield_indices=True), tr)
        valid_gen = AdversaryGenerator(
            StratifiedSampling(val.X, val.y, split_condition=is_phi_sentence,
                               batch_size=test_batch_size, shuffle=False, yield_indices=True), val)
        return model, tr, train_gen, val, valid_gen, experiment_dir

    history['branches'] = []
    history['representer'] = []

    fine_tune_epochs = config['training']['train_epochs']
    early_stopping_counter = 0
    early_stopping_best = 100
    for epoch in range(fine_tune_epochs):
        if early_stopping_counter == config['training']['early_stopping_patience']:
            print('Early stopping')
            break

        flush = Flush()
        stop_after_every_epoch = StopAfterEveryEpoch()
        callbacks = [flush, stop_after_every_epoch]

        # Train representer
        print('Training representer')
        epoch_history = model.fine_tune_representer.fit_generator(train_gen,
                                                                  steps_per_epoch=train_gen.epoch_length,
                                                                  validation_data=valid_gen,
                                                                  validation_steps=valid_gen.epoch_length,
                                                                  verbose=env.keras_verbose,
                                                                  callbacks=callbacks,
                                                                  epochs=fine_tune_epochs,
                                                                  initial_epoch=epoch,
                                                                  class_weight=[class_weight, None]).history
        history['representer'].append(epoch_history)
        save_history(history, experiment_dir)
        if epoch_history['val_loss'][-1] < early_stopping_best:
            early_stopping_best = epoch_history['val_loss'][-1]
            early_stopping_counter = 0
            if env.save_model:
                model.complete_model.save_weights(os.path.join(experiment_dir, 'model-fine-tuning.hdf5'),
                                                  overwrite=True)
        else:
            early_stopping_counter += 1

        # Train branches
        print('Training branches')
        epoch_history = model.fine_tune_branches.fit_generator(train_gen,
                                                               steps_per_epoch=train_gen.epoch_length,
                                                               validation_data=valid_gen,
                                                               validation_steps=valid_gen.epoch_length,
                                                               verbose=env.keras_verbose,
                                                               callbacks=callbacks,
                                                               epochs=fine_tune_epochs,
                                                               initial_epoch=epoch,
                                                               class_weight=[class_weight, None]).history
        history['representer'].append(epoch_history)
        save_history(history, experiment_dir)

    if config['test']['run_test']:
        print('Restoring best weights')
        if env.save_model and fine_tune_epochs > 0:
            model.complete_model.load_weights(os.path.join(experiment_dir, 'model-fine-tuning.hdf5'))
        elif config['test']['test_weights'] is not None:
            model.complete_model.load_weights(config['test']['test_weights'])

        deid_result = evaluate_deid_performance(model=model.pretrain_deidentifier, batch_size=test_batch_size,
                                                embeddings=lookup_embeddings, label2ind=tr.label2ind,
                                                ind2label=tr.ind2label, test_set='test', experiment_dir=experiment_dir,
                                                binary_classification=config['experiment']['binary_classification'],
                                                hipaa_only=config['experiment']['hipaa_only'],
                                                extra_features=extra_features, epoch=99)
        history['deid_result'] = deid_result
        save_history(history, experiment_dir)

        label2ind = tr.label2ind
        del tr, val, train_gen, valid_gen

        test_augment = Augment(lookup_embeddings, strategy=strategy, digit_strategy=digit_strategy,
                               **{**config['augment']['augment_args'], 'n_augmentations': 1})

        print('Augmenting test set...')
        test = ValidationSet(validation_set='test',
                             embeddings=lookup_embeddings,
                             label2ind=label2ind,
                             use_short_sentences=env.use_short_sentences,
                             limit_documents=env.limit_validation_documents,
                             augment=test_augment,
                             binary_classification=config['experiment']['binary_classification'],
                             hipaa_only=config['experiment']['hipaa_only'],
                             extra_features=extra_features)

        test_gen = AdversaryGenerator(
            StratifiedSampling(test.X, test.y, batch_size=training_batch_size, split_condition=is_phi_sentence,
                               yield_indices=True, shuffle=False), test)

        test_loss, test_acc = model.pretrain_adversary.evaluate_generator(test_gen,
                                                                          steps=test_gen.epoch_length,
                                                                          verbose=env.keras_verbose)
        print(f'test loss: {test_loss}, test_acc: {test_acc}')
        history['test'] = {'loss': test_loss, 'acc': test_acc}
        save_history(history, experiment_dir)


def pretrain_deidentifier(config, experiment_dir, model, training_set, validation_set, training_batch_size,
                          validation_batch_size,
                          class_weight):
    epochs = config['training']['pretrain_deidentifier_epochs']
    if epochs == 0:
        print('Skipping deidentifier pretraining.')
        return {}

    train_gen = BatchGeneratorWithExtraFeatures(training_set.X, training_set.y, training_set.X_extra,
                                                batch_size=training_batch_size)
    valid_gen = BatchGeneratorWithExtraFeatures(validation_set.X, validation_set.y, validation_set.X_extra,
                                                batch_size=validation_batch_size, shuffle=False)

    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])
    flush = Flush()
    callbacks = [early_stopping, flush]

    weights_path = os.path.join(experiment_dir, 'model-deid-pretrain.hdf5')
    if env.save_model:
        checkpoint = MainModelCheckpoint(model.complete_model, weights_path, save_weights_only=True,
                                         save_best_only=True)
        callbacks.append(checkpoint)

    history = model.pretrain_deidentifier.fit_generator(train_gen,
                                                        steps_per_epoch=train_gen.epoch_length,
                                                        epochs=epochs,
                                                        validation_data=valid_gen,
                                                        validation_steps=valid_gen.epoch_length,
                                                        verbose=env.keras_verbose,
                                                        callbacks=callbacks,
                                                        class_weight=class_weight)
    if env.save_model:
        print('Restoring best weights...', flush=True)
        model.complete_model.load_weights(weights_path)
    return history.history


def pretrain_adversary(config, experiment_dir, model, train_set, validation_set, training_batch_size,
                       validation_batch_size):
    epochs = config['training']['pretrain_adversary_epochs']
    if epochs == 0:
        print('Skipping adversary pretraining.')
        return {}

    train_gen = AdversaryGenerator(StratifiedSampling(train_set.X, train_set.y, split_condition=is_phi_sentence,
                                                      batch_size=training_batch_size, yield_indices=True), train_set)
    valid_gen = AdversaryGenerator(
        StratifiedSampling(validation_set.X, validation_set.y, split_condition=is_phi_sentence,
                           batch_size=validation_batch_size, shuffle=False, yield_indices=True), validation_set)

    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'])
    flush = Flush()
    callbacks = [early_stopping, flush]

    weights_path = os.path.join(experiment_dir, 'model-adversary-pretrain.hdf5')
    if env.save_model:
        checkpoint = MainModelCheckpoint(model.complete_model, weights_path, save_weights_only=True,
                                         save_best_only=True)
        callbacks.append(checkpoint)

    history = model.pretrain_adversary.fit_generator(train_gen,
                                                     steps_per_epoch=train_gen.epoch_length,
                                                     epochs=epochs,
                                                     validation_data=valid_gen,
                                                     validation_steps=valid_gen.epoch_length,
                                                     verbose=env.keras_verbose,
                                                     callbacks=callbacks)
    if env.save_model:
        print('Restoring best weights...', flush=True)
        model.complete_model.load_weights(weights_path)
    return history.history
