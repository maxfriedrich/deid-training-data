from typing import Sequence, Dict

import numpy as np


class Embeddings:
    """ Flexible base class for embeddings that doesn't necessarily use a matrix """

    @property
    def size(self) -> int:
        raise NotImplementedError

    @property
    def mean(self) -> float:
        return 0.

    @property
    def std(self) -> float:
        raise NotImplementedError

    def is_unknown(self, word: str) -> bool:
        raise NotImplementedError

    def lookup(self, word: str) -> np.ndarray:
        """ Looks up the vector representation of one word.

        :param word: an input string
        :return: a vector representation of size `size`
        """
        raise NotImplementedError

    def lookup_sentence(self, words: Sequence[str]) -> Sequence[np.ndarray]:
        """ Looks up the vector representation of multiple words. Override this if there is a more efficient way to get
        a batch of embeddings than looking them up one by one.

        :param words: a sequence of input strings
        :return: a vector representation of size `(len(words), size)`
        """
        return np.array([self.lookup(word) for word in words])

    def lookup_sentences(self, sentences: Sequence[Sequence[str]]) -> Sequence[Sequence[np.ndarray]]:
        """ Looks up the vector representation of an entire sentence. Override this if there is a more efficient way to
        get a batch of embeddings sequences than looking them up one by one.

        :param sentences: a sequence of sequences of input strings
        :return: a sequence of arrays that have size `(len(sentence), size)` for the corresponding sentence
        """

        return [self.lookup_sentence(sentence) for sentence in sentences]


class PrecomputedEmbeddings(Embeddings):
    """ Base class for embeddings that provide a precomputed matrix in addition to the lookup """

    @property
    def size(self) -> int:
        raise NotImplementedError

    @property
    def std(self) -> float:
        raise NotImplementedError

    def is_unknown(self, word: str) -> bool:
        raise NotImplementedError

    def lookup(self, word: str) -> np.ndarray:
        raise NotImplementedError

    @property
    def precomputed_word2ind(self) -> Dict[str, int]:
        raise NotImplementedError

    @property
    def precomputed_matrix(self) -> np.ndarray:
        raise NotImplementedError
