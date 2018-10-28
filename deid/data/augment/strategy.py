import logging
import random
import re
from typing import Any, Optional, Sequence, Dict

import numpy as np

from ...embeddings import Matrix, EmbeddingSimilarity

logger = logging.getLogger()
digit_pattern = '^[0-9]*$'


class AugmentStrategy:
    augments_words: bool

    @property
    def description(self) -> Optional[str]:
        return None

    def augment(self, word_or_embedding: Any) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        options = '' if self.description is None else ' ' + self.description
        return f'<{self.__class__.__name__}{options}>'


class AugmentWord(AugmentStrategy):
    augments_words = True

    def augment(self, word: str) -> str:
        raise NotImplementedError


class AugmentEmbedding(AugmentStrategy):
    augments_words = False

    def augment(self, word_embedding: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Zeros(AugmentEmbedding):
    """ Not actually zeros to distinguish from masking. """

    def augment(self, word_embedding: np.ndarray) -> np.ndarray:
        return np.random.normal(0., scale=1e-6, size=len(word_embedding))


class RandomEmbedding(AugmentEmbedding):
    """ A random normal embedding, optionally L2 normalized """

    def __init__(self, scale=None, l2_normalize=True):
        self.scale = 1. if scale is None else scale
        self.l2_normalize = l2_normalize

    @property
    def description(self) -> Optional[str]:
        return f'scale={self.scale}, l2_normalize={self.l2_normalize}'

    def augment(self, word_embedding: np.ndarray) -> np.ndarray:
        embedding = np.random.normal(0., scale=self.scale, size=len(word_embedding))
        if self.l2_normalize:
            embedding = embedding / np.linalg.norm(embedding)
        return embedding


class RandomDigits(AugmentWord):
    def __init__(self, matrix: Matrix) -> None:
        self.matrix = matrix
        logger.info('getting digit indices')
        self.digit_ind = [ind for word, ind in matrix.word2ind.items() if re.match(digit_pattern, str(word))]
        logger.info('found %d indices', len(self.digit_ind))

    def augment(self, word: str) -> str:
        ind = random.choice(self.digit_ind)
        return self.matrix.ind2word[ind]


class AdditiveNoise(AugmentEmbedding):
    def __init__(self, scale: float) -> None:
        self.scale = scale

    @property
    def description(self) -> Optional[str]:
        return f'scale={self.scale}'

    def augment(self, word_embedding: np.ndarray) -> np.ndarray:
        noisy = word_embedding + np.random.normal(0, scale=self.scale, size=len(word_embedding))
        return noisy / np.linalg.norm(noisy)


class MoveToNeighbor(AugmentWord):
    """ Only makes sense for embeddings like GloVE and fastText that have a fixed word->embedding lookup """

    def __init__(self, matrix: Matrix, n_neighbors: int, cache_mode: str = 'neighbors') -> None:
        self.matrix = matrix
        self.n_neighbors = n_neighbors
        self.cache = NeighborsCache(cache_mode)

    @property
    def description(self) -> Optional[str]:
        return f'n_neighbors={self.n_neighbors}'

    def augment(self, word: str) -> str:
        cache_result = self.cache.lookup(word)
        if cache_result is None:
            neighbors = self.matrix.most_similar_cosine(word, n=self.n_neighbors)
            selected = random.choice(neighbors)
            self.cache.store(word, neighbors, selected)
        else:
            selected = cache_result
        return selected.word


class NeighborsCache:
    def __init__(self, mode: Optional[str]) -> None:
        if mode not in [None, 'neighbors', 'selected']:
            raise ValueError("Cache mode must be either None, 'neighbors' or 'selected'")
        self.mode = mode
        self.cache: Dict[str, Sequence[EmbeddingSimilarity]] = {}

    def lookup(self, word: str) -> Optional[EmbeddingSimilarity]:
        if self.mode is None:
            return None

        result = self.cache.get(word)
        return result if result is None else random.choice(result)

    def store(self, word: str, neighbors: Sequence[EmbeddingSimilarity], selected: EmbeddingSimilarity) -> None:
        if self.mode == 'neighbors':
            self.cache[word] = neighbors
        if self.mode == 'selected':
            self.cache[word] = [selected]
