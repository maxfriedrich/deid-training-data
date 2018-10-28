from keras import backend as K
from keras.layers import Input, Dense, LSTM, Bidirectional, TimeDistributed, Masking, Dropout, concatenate, Lambda
from keras.models import Model
from keras_contrib.layers import CRF


def make_lstm_crf(input_size, hidden_size, output_size, name='deidentifier', extra_input_size=0, num_hidden=1,
                  input_dropout=0., recurrent_dropout=0., after_hidden_dropout=0., use_crf=False, optimizer=None,
                  l2_normalize=False):
    """ Make a BiLSTM(-CRF) model that can be used for de-identification.

    :param input_size: the embedding/representation input size
    :param hidden_size: the number of LSTM units per direction
    :param output_size: the number of output labels
    :param name: a name for the model
    :param extra_input_size: size for an additional input, if it is 0, this returns a single-input model
    :param num_hidden: the number of LSTM layers
    :param input_dropout: dropout probability for the input layer
    :param recurrent_dropout: recurrent (variational) dropout probability
    :param after_hidden_dropout: dropout probability for the LSTM outputs
    :param use_crf: whether to use a CRF to optimize the output sequences
    :param optimizer: a Keras optimizer, or None if the model should not be compiled
    :param l2_normalize: whether to L2 normalize the embedding/representation input
    :return: a tuple (model, loss), or a compiled Keras model if an optimizer was specified
    """
    embedding_input = Input(shape=(None, input_size))
    x = Masking()(embedding_input)
    if l2_normalize:
        x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)
    x = Dropout(input_dropout)(x)

    extra_input = Input(shape=(None, extra_input_size))
    if extra_input_size > 0:
        x2 = Masking()(extra_input)
        x = concatenate([x, x2])

    for _ in range(num_hidden):
        x = Bidirectional(LSTM(hidden_size, return_sequences=True, dropout=after_hidden_dropout,
                               recurrent_dropout=recurrent_dropout))(x)
    if use_crf:
        # CRF learn mode 'join' does not work at the moment, this GitHub issue contains a minimal example showing
        # the problem: https://github.com/keras-team/keras-contrib/issues/271
        x = TimeDistributed(Dense(output_size, activation=None))(x)
        crf = CRF(output_size, sparse_target=True, learn_mode='marginal', name='deid_output')
        x = crf(x)
        loss = crf.loss_function
    else:
        x = TimeDistributed(Dense(output_size, activation='softmax'), name='deid_output')(x)
        loss = 'sparse_categorical_crossentropy'

    if extra_input_size > 0:
        model = Model([embedding_input, extra_input], x, name=name)
    else:
        model = Model(embedding_input, x, name=name)

    if optimizer is not None:
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model
    return model, loss
