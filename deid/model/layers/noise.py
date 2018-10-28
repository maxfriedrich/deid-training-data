from keras import backend as K
from keras.engine.topology import Layer


class Noise(Layer):
    """ Abstract Gaussian Noise layer with trainable mean and standard deviation """

    def __init__(self, operation, single_stddev: bool, apply_noise: bool = True, **kwargs) -> None:
        """ Initializes the Noise layer.

        :param operation: the operation to apply to the inputs and noise, may be '+'/'add' or '*'/'mult'. The mean of
        the noise will be set according to this operator.
        :param single_stddev: whether to learn a matrix of noise stddev values instead of only one stddev value that is
        applied to all dimensions of the data
        :param apply_noise: set this to False to only apply the mean instead of noise
        :param kwargs: other Layer arguments
        """
        super().__init__(**kwargs)
        if operation == '+' or operation == 'add':
            self.operation = lambda x, y: x + y
            self.mean = 0.
        elif operation == '*' or operation == 'mult':
            self.operation = lambda x, y: x * y
            self.mean = 1.
        else:
            raise ValueError(f'unknown operation: {operation}')

        self.apply_noise = K.constant(value=apply_noise)
        self.single_stddev = single_stddev
        self.k = self.stddev = None  # will be initialized in the build method

        self.supports_masking = True

    def build(self, input_shape):
        self.k = self.add_weight(name='k',
                                 shape=(1,),
                                 initializer='ones',
                                 trainable=True)
        self.stddev = self.add_weight(name='stddev',
                                      shape=(1,) if self.single_stddev else (input_shape[-1],),
                                      initializer='normal',
                                      trainable=True)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        def noise():
            noise_matrix = K.random_normal(shape=K.shape(inputs), mean=self.mean, stddev=self.stddev)
            return self.operation(inputs, self.k * noise_matrix)

        return K.switch(self.apply_noise, noise, inputs)

    def get_config(self):
        config = {'apply_noise': self.apply_noise,
                  'mean': self.mean,
                  'single_stddev': self.single_stddev,
                  'k': self.k}
        base_config = super().get_config()
        return {**base_config, **config}


class AdditiveNoise(Noise):
    def __init__(self, **kwargs):
        super().__init__('+', **kwargs)


class MultiplicativeNoise(Noise):
    def __init__(self, **kwargs):
        super().__init__('*', **kwargs)
