import numpy as np

from . import Embeddings


class Noise:
    def noise(self, size: int) -> None:
        raise NotImplementedError


class GaussianNoise(Noise):
    def __init__(self, scale: float, loc=0., clip=None) -> None:
        self.loc = loc
        self.scale = scale
        self.clip = clip

    def noise(self, size):
        result = np.random.normal(self.loc, self.scale, size)
        if self.clip is not None:
            result = np.clip(result, self.clip[0], self.clip[1])
        return result


class DropoutNoise(Noise):
    def __init__(self, dropout_prob) -> None:
        self.dropout_prob = dropout_prob

    def noise(self, size):
        return np.random.choice(2, size, p=[self.dropout_prob, 1 - self.dropout_prob])


class NoiseWrapper(Embeddings):
    def __init__(self, embeddings: Embeddings, op, noise: Noise) -> None:
        self.wrapped_embeddings = embeddings
        self.noise = noise

        if type(op) == str:
            if op == 'add' or op == '+':
                self.op = lambda x, y: x + y
            elif op == 'mul' or op == '*':
                self.op = lambda x, y: x * y
            else:
                raise ValueError(f'Unrecognized op: {op}')
        else:
            self.op = op

    @property
    def size(self):
        return self.wrapped_embeddings.size

    def lookup(self, word):
        return self.op(self.wrapped_embeddings.lookup(word), self.noise.noise(self.size))

    def __str__(self):
        return f'<{self.__class__.__name__} wrapper of {self.wrapped_embeddings} {vars(self)}>'
