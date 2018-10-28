from keras import backend as K
from keras.layers import Dense, Lambda, LSTM, Bidirectional, TimeDistributed, Masking
from keras.models import Sequential

from .layers import Noise


def get(identifier):
    if identifier == 'noisy':
        return build_noise_representer
    elif identifier == 'dense':
        return build_dense_representer
    elif identifier == 'lstm':
        return build_lstm_representer
    else:
        raise ValueError(f'Unknown representation type: "{identifier}"')


def build_noise_representer(embedding_size, representation_size, noises, single_stddev, apply_noise,
                            l2_normalize=False, **_):
    """ Build a representer that applies a series of noise steps.

    :param embedding_size: the embedding (input) size
    :param representation_size: the representation (output) size
    :param noises: the types of noise to add if using the 'noisy' representation. Must be a single
    identifier or sequence of identifiers, allowed identifiers are '+'/'add' or '*'/'mult'
    :param single_stddev: whether to use a single stddev for all embedding dimensions
    :param apply_noise: whether to apply noise or the mean in this model
    :param l2_normalize: whether to L2 normalize the inputs (outputs are always L2 normalized)
    :return: a noisy representer model
    """
    if type(noises) == str:
        noises = [noises]

    model = Sequential(name='representer')
    model.add(Masking(input_shape=(None, embedding_size)))
    if l2_normalize:
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
    for i, noise_operation in enumerate(noises):
        model.add(Noise(noise_operation, apply_noise=apply_noise, single_stddev=single_stddev,
                        input_shape=(None, embedding_size)))

    model.add(TimeDistributed(Dense(representation_size)))
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
    return model


def build_dense_representer(embedding_size, representation_size, apply_noise, num_hidden=2, hidden_size=None,
                            l2_normalize=False, noise_before=True, noise_after=True, single_stddev=False, **_):
    """ Build a dense representer that applies the same dense weights to each element in the input sequence.

    :param embedding_size: the embedding (input) size
    :param representation_size: the representation (output) size
    :param apply_noise: whether to apply noise or the mean in this model
    :param num_hidden: the number of hidden layers in the dense model
    :param hidden_size: the number of units per hidden layer in the dense model
    :param l2_normalize: whether to L2 normalize the inputs (outputs are always L2 normalized)
    :param noise_before: whether to add noise with trainable stddev to the inputs
    :param noise_after: whether to add noise with trainable stddev to the outputs
    :param single_stddev: whether to use a single stddev for all embedding dimensions
    :param _: ignored kwargs
    :return: a dense representer model
    """
    if hidden_size is None:
        hidden_size = embedding_size

    model = Sequential(name='representer')
    model.add(Masking(input_shape=(None, embedding_size)))
    if l2_normalize:
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
    if noise_before:
        model.add(Noise('add', single_stddev=single_stddev, apply_noise=apply_noise))

    for _ in range(num_hidden):
        model.add(TimeDistributed(Dense(hidden_size, activation='relu')))
    model.add(TimeDistributed(Dense(representation_size)))

    if noise_after:
        model.add(Noise('add', single_stddev=single_stddev, apply_noise=apply_noise))
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
    return model


def build_lstm_representer(embedding_size, representation_size, apply_noise, num_hidden=1, lstm_size=128,
                           l2_normalize=False, noise_before=True, noise_after=True, single_stddev=False, **_):
    """ Build an LSTM representer.

    :param embedding_size: the embedding (input) size
    :param representation_size: the representation (output) size
    :param apply_noise: whether to apply noise or the mean in this model
    :param num_hidden: the number of LSTM layers
    :param lstm_size: the number of LSTM units per direction and layer
    :param l2_normalize: whether to L2 normalize the inputs (outputs are always L2 normalized)
    :param noise_before: whether to add noise with trainable stddev to the inputs
    :param noise_after: whether to add noise with trainable stddev to the outputs
    :param single_stddev: whether to use a single stddev for all embedding dimensions
    :param _: ignored kwargs
    :return: an LSTM representer model
    """
    model = Sequential(name='representer')
    model.add(Masking(input_shape=(None, embedding_size)))
    if l2_normalize:
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
    if noise_before:
        model.add(Noise('add', single_stddev=single_stddev, apply_noise=apply_noise))

    for _ in range(num_hidden):
        model.add(Bidirectional(LSTM(lstm_size, return_sequences=True)))
    model.add(TimeDistributed(Dense(representation_size)))

    if noise_after:
        model.add(Noise('add', single_stddev=single_stddev, apply_noise=apply_noise))
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
    return model
