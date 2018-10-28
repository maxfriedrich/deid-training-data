import os
import re
from typing import Optional, Dict

import numpy as np

from . import PrecomputedEmbeddings
from ..env import env

glove_dir = os.path.join(env.resources_dir, 'glove.6B')


class GloveEmbeddings(PrecomputedEmbeddings):
    """ Pre-trained GloVe embeddings, see https://nlp.stanford.edu/projects/glove/ """

    def __init__(self, dims: int = 300, vocab_size: Optional[int] = None) -> None:
        """ Initialize a GloveEmbeddings object.

        :param dims: the GloVe variant to use (50, 100, 200, or 300 dimensions)
        :param vocab_size: limits the size of the embedding matrix
        """
        self._dims = dims
        filename = os.path.join(glove_dir, f'glove.6B.{dims}d.txt')
        if not os.path.isfile(filename):
            raise ValueError(f"Can't find GloVe embeddings with {dims} dims in {glove_dir}.")

        embeddings = [np.zeros(dims), np.random.normal(0., scale=1e-6, size=dims)]  # Padding and UNK
        self._word2ind = {env.unk_token: 1}
        self._ind2word = {1: env.unk_token}

        with open(filename) as f:
            for i, line in enumerate(f, start=2):
                values = line.split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                self._word2ind[word] = i
                self._ind2word[i] = word
                embeddings.append(embedding / np.linalg.norm(embedding))
                if i == vocab_size:
                    break

        self._embeddings = np.array(embeddings)

    @property
    def precomputed_word2ind(self) -> Dict[str, int]:
        return self._word2ind

    @property
    def precomputed_matrix(self) -> np.ndarray:
        return self._embeddings

    @property
    def size(self) -> int:
        return self._dims

    @property
    def std(self):
        return 0.37

    def word2ind(self, word: str) -> int:
        result = self._word2ind.get(word)
        if result is not None:
            return result

        word = word.lower()
        result = self._word2ind.get(word)
        if result is not None:
            return result

        word = re.sub(r'\W', '', word)
        result = self._word2ind.get(word)
        if result is not None:
            return result

        # replace every digit with a 0
        result = self._word2ind.get(re.sub(r'\d', '0', word))
        if result is not None:
            return result

        # replace all connected digits with a single 0
        result = self._word2ind.get(re.sub(r'\d*', '0', word))
        if result is not None:
            return result

        return self._word2ind[env.unk_token]

    def lookup(self, word: str) -> np.ndarray:
        return self._embeddings[self.word2ind(word)]

    def is_unknown(self, word: str):
        return np.all(self.word2ind(word) == self._word2ind[env.unk_token])

    def __str__(self) -> str:
        return '<GloVeEmbeddings>'
