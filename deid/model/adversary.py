from keras import backend as K
from keras.layers import Input, Dense, Lambda, LSTM, Bidirectional, TimeDistributed, Dropout, concatenate
from keras.models import Model, Sequential

from .layers import GradientReversal

discriminator_loss = 'binary_crossentropy'


def get(identifier):
    if identifier == 'reconstruct':
        return Reidentifier
    elif identifier == 'discriminate-representations':
        return TwoRepresentationsAreSameOriginalDiscriminator
    elif identifier == 'discriminate-representation-embedding-pair':
        return OriginalAndRepresentationAreMatchingDiscriminator
    else:
        raise ValueError(f'Unknown adversary: "{identifier}"')


class Adversary:
    """ An adversary is a model with a gradient reversal layer. It can chose its inputs from a dictionary that contains
    entries for 'train_representation', 'fake_representation', and 'original_embeddings'.
    """

    def __init__(self, model, loss, inputs, **compile_kwargs):
        self.model = model
        self.loss = loss
        self.inputs = inputs
        self.compile_kwargs = compile_kwargs


class Reidentifier(Adversary):
    def __init__(self, inputs, representation_size, embedding_size, lstm_size, input_dropout=0., recurrent_dropout=0.,
                 reverse_gradient=True, **_):
        model = Sequential(name='reidentifier')
        model.add(Dropout(input_dropout, input_shape=(None, representation_size)))
        if reverse_gradient:
            model.add(GradientReversal())
        model.add(Bidirectional(LSTM(lstm_size, return_sequences=True, recurrent_dropout=recurrent_dropout)))
        model.add(TimeDistributed(Dense(embedding_size)))
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))
        super().__init__(model, inputs=[inputs['train_representation']], loss='mse', sample_weight_mode='temporal',
                         metrics=['cosine_proximity'])


class TwoRepresentationsAreSameOriginalDiscriminator(Adversary):
    def __init__(self, inputs, representation_size, lstm_size, input_dropout=0., recurrent_dropout=0.,
                 reverse_gradient=True, **_):
        """ LSTM size should be at least the representation size for this to converge quickly. """
        representation_input1 = Input(shape=(None, representation_size))
        representation_input2 = Input(shape=(None, representation_size))

        # (batch_size, maxlen, repr_size) -> (batch_size, maxlen, 1) -- the dot layer doesn't do this
        normalized_1 = Lambda(lambda x: K.l2_normalize(x, axis=-1))(representation_input1)
        normalized_2 = Lambda(lambda x: K.l2_normalize(x, axis=-1))(representation_input2)
        dot_product = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True))([normalized_1, normalized_2])

        both_inputs = concatenate([representation_input1, representation_input2], axis=-1)
        both_inputs = Dropout(input_dropout)(both_inputs)

        inputs_and_dot_product = concatenate([both_inputs, dot_product], axis=-1)
        if reverse_gradient:
            inputs_and_dot_product = GradientReversal()(inputs_and_dot_product)

        summary = Bidirectional(LSTM(lstm_size, recurrent_dropout=recurrent_dropout))(inputs_and_dot_product)
        output = Dense(1, activation='sigmoid')(summary)

        model = Model([representation_input1, representation_input2], output, name='rr-adv')
        super().__init__(model, inputs=[inputs['train_representation'], inputs['fake_representation']],
                         loss=discriminator_loss, metrics=['accuracy'])


class OriginalAndRepresentationAreMatchingDiscriminator(Adversary):
    def __init__(self, inputs, representation_size, embedding_size, lstm_size, input_dropout=0., recurrent_dropout=0.,
                 reverse_gradient=True, **_):
        embedding_input = Input(shape=(None, embedding_size))
        representation_input = Input(shape=(None, representation_size))

        both_inputs = concatenate([embedding_input, representation_input], axis=-1)
        if reverse_gradient:
            both_inputs = GradientReversal()(both_inputs)
        both_inputs = Dropout(input_dropout)(both_inputs)
        summary = Bidirectional(LSTM(lstm_size, recurrent_dropout=recurrent_dropout))(both_inputs)

        output = Dense(1, activation='sigmoid')(summary)

        model = Model([embedding_input, representation_input], output, name='er-adv')
        super().__init__(model, inputs=[inputs['original_embeddings'], inputs['fake_representation']],
                         loss=discriminator_loss, metrics=['accuracy'])
