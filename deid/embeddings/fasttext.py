import os
import pickle
from typing import Dict

import fastText
import numpy as np
from tqdm import tqdm

from . import PrecomputedEmbeddings
from ..env import env

fasttext_dir = os.path.join(env.resources_dir, 'fastText')
fasttext_embeddings_name = 'wiki-news-300d-1M-subword'


class FastTextEmbeddings(PrecomputedEmbeddings):
    def __new__(cls, *args, **kwargs):
        if env.embeddings_cache:
            return CachedFastTextEmbeddings()
        return PreloadFastTextEmbeddings()

    def __init__(self, *_, **__):
        raise NotImplementedError('this should not happen')

    @property
    def size(self) -> int:
        raise NotImplementedError

    @property
    def std(self):
        raise NotImplementedError

    def lookup(self, word: str) -> np.ndarray:
        raise NotImplementedError

    def is_unknown(self, word: str):
        return NotImplementedError

    @property
    def precomputed_word2ind(self) -> Dict[str, int]:
        raise NotImplementedError

    @property
    def precomputed_matrix(self) -> np.ndarray:
        raise NotImplementedError


class FastTextEmbeddingsImpl(PrecomputedEmbeddings):
    def __init__(self, size, *_, **__):
        self._size = size
        self._precomputed_word2ind = None
        self._precomputed_matrix = None

    @property
    def precomputed_word2ind(self) -> Dict[str, int]:
        if self._precomputed_word2ind is None:
            vocab_filename = os.path.join(fasttext_dir, fasttext_embeddings_name + '.vec.vocab.pickle')
            self._precomputed_word2ind = pickle.load(open(vocab_filename, 'rb'))
        return self._precomputed_word2ind

    @property
    def precomputed_matrix(self) -> np.ndarray:
        if self._precomputed_matrix is None:
            matrix_filename = os.path.join(fasttext_dir, fasttext_embeddings_name + '.vec.matrix.npy')
            self._precomputed_matrix = np.load(matrix_filename)
        return self._precomputed_matrix

    @staticmethod
    def l2_normalize_if_needed(vec: np.ndarray, l2_normalize: bool) -> np.ndarray:
        if l2_normalize:
            vec /= np.linalg.norm(vec)  # all-zero embeddings shouldn't exist
        return vec

    @property
    def size(self) -> int:
        return self._size

    @property
    def std(self) -> float:
        return 0.05

    def lookup(self, word: str) -> np.ndarray:
        raise NotImplementedError

    def is_unknown(self, word: str) -> bool:
        return False


class PreloadFastTextEmbeddings(FastTextEmbeddingsImpl):
    def __init__(self) -> None:
        self.model = fastText.load_model(os.path.join(fasttext_dir, fasttext_embeddings_name + '.bin'))
        super().__init__(self.model.get_dimension())

    def lookup(self, word: str, l2_normalize: bool = True) -> np.ndarray:
        vec = self.model.get_word_vector(word)
        if np.count_nonzero(vec) == 0:
            # add small amount of noise to all-zero embeddings to make them work with masking / CRF
            vec += np.random.normal(0., scale=1e-6, size=len(vec))

        return self.l2_normalize_if_needed(vec, l2_normalize)

    def __str__(self) -> str:
        return '<PreloadFastTextEmbeddings>'


class CachedFastTextEmbeddings(FastTextEmbeddingsImpl):  # always L2 normalized!
    def __init__(self, vocab=None):
        cache_path = os.path.join(fasttext_dir, fasttext_embeddings_name + '.pickle')
        if vocab is None:
            self.word2ind, self.matrix = pickle.load(open(cache_path, 'rb'))
        else:
            vocab = set(vocab)
            embeddings = PreloadFastTextEmbeddings()
            self.word2ind = {word: i + 1 for i, word in enumerate(vocab)}
            self.matrix = np.zeros((len(vocab) + 1, embeddings.size))
            for i, word in tqdm(enumerate(vocab, start=1), desc='Looking up words', total=len(vocab)):
                self.matrix[i] = embeddings.lookup(word, l2_normalize=True)

            pickle.dump((self.word2ind, self.matrix), open(cache_path, 'wb'))
        super().__init__(self.matrix.shape[1])

    def lookup(self, word: str, include_precomputed: bool = True) -> np.ndarray:
        index = self.word2ind.get(word)
        if index is not None:
            return self.matrix[index]

        index = self.precomputed_word2ind.get(word)
        if index is not None:
            return self.precomputed_matrix[index]

        raise RuntimeError(f'Cache/precomputed lookup failed for "{word}". Please rebuild the embedding cache.')
