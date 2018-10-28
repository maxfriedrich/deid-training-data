import numpy as np

from .token import Token
from .util import one_hot


def get(identifier):
    if identifier == 'case':
        return CaseFeature()
    if identifier == 'one':
        return UselessOneFeature()
    raise ValueError(f'unknown feature identifier: {identifier}')


class Feature:
    def apply(self, token) -> np.ndarray:
        raise NotImplementedError

    @property
    def dimension(self):
        return NotImplementedError


class CaseFeature(Feature):
    """ Casing feature from Reimers and Gurevych (2017) https://arxiv.org/abs/1707.06799 """
    OTHER = 0
    NUMERIC = 1
    MAINLY_NUMERIC = 2
    ALL_LOWER = 3
    ALL_UPPER = 4
    INITIAL_UPPER = 5
    CONTAINS_DIGIT = 6

    def apply(self, token: Token) -> np.ndarray:
        token = token.text

        num_digits = len([char for char in token if char.isdigit()])
        digit_fraction = num_digits / len(token)

        if token.isdigit():
            casing = self.NUMERIC
        elif digit_fraction > 0.5:
            casing = self.MAINLY_NUMERIC
        elif token.islower():
            casing = self.ALL_LOWER
        elif token.isupper():
            casing = self.ALL_UPPER
        elif token[0].isupper():
            casing = self.INITIAL_UPPER
        elif num_digits > 0:
            casing = self.CONTAINS_DIGIT
        else:
            casing = self.OTHER

        return one_hot(casing, 7)

    @property
    def dimension(self):
        return 7


class UselessOneFeature(Feature):
    def apply(self, token) -> np.ndarray:
        return np.array([1])

    @property
    def dimension(self):
        return 1


def apply_features(features, sent):
    if len(features) == 0:
        return np.array([np.array([]) for _ in sent])
    return np.array([np.concatenate([feature.apply(word) for feature in features]) for word in sent])
