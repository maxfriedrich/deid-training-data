from typing import Iterable, List, Tuple, NamedTuple, Union, Optional, Dict

import numpy as np
from tqdm import tqdm

from . import Embeddings


class EmbeddingSimilarity(NamedTuple):
    rank: int
    word: str
    similarity: float
    vec: np.ndarray


MostSimilarResult = List[EmbeddingSimilarity]
WordOrVec = Union[str, np.ndarray]


class Matrix:
    """ Transforms a lookup-based Embeddings object into a classical embedding matrix by looking up a fixed vocabulary
    and storing the results. The matrix can then be used for distance measuring.
    """

    def __init__(self, lookup_embeddings: Embeddings, vocab: Optional[Iterable[str]] = None,
                 precomputed_word2ind: Optional[Dict[str, int]] = None, precomputed_matrix: Optional[np.ndarray] = None,
                 verbose: bool = False) -> None:
        """ Initialize the Matrix object.

        :param lookup_embeddings: the embeddings object used for lookup
        :param vocab: an iterable containing the words that should be stored in the matrix
        :param precomputed_word2ind: a precomputed word2ind dict, e.g. from the fastText .vec file
        :param precomputed_matrix: a precomputed embedding matrix, e.g. from the fastText .vec file
        :param verbose: setting this to True will show a progress bar when first looking up embeddings as well as output
        means when computing distances
        """
        self.verbose = verbose
        self.lookup_embeddings = lookup_embeddings

        if vocab is not None:
            self._init_from_vocab(lookup_embeddings, vocab=vocab)
        elif precomputed_word2ind is not None and precomputed_matrix is not None:
            self._init_from_word2ind_and_matrix(precomputed_word2ind, precomputed_matrix)
        else:
            raise ValueError('The Matrix needs to be initialized either with vocab or word2ind+matrix')

    def _init_from_vocab(self, lookup_embeddings, vocab):
        vocab = set(vocab)
        self.vocab_size = len(vocab)
        self.word2ind = {word: i for i, word in enumerate(vocab)}
        self.ind2word = {i: word for i, word in enumerate(vocab)}
        self.embedding_matrix = np.zeros((self.vocab_size, lookup_embeddings.size))
        self.is_norm = False

        items: Iterable[Tuple[str, int]] = self.word2ind.items()
        if self.verbose:
            items = tqdm(items, desc='Looking up embeddings')
        for word, ind in items:
            looked_up = lookup_embeddings.lookup(word)
            if np.count_nonzero(looked_up) > 0:
                self.embedding_matrix[ind] = looked_up
            else:
                # this shouldn't happen anymore
                raise RuntimeError(f'Embedding vector for {word} is all zeros')

    def _init_from_word2ind_and_matrix(self, word2ind, matrix):
        self.vocab_size = len(word2ind)
        self.word2ind = word2ind
        self.ind2word = {i: word for word, i in self.word2ind.items()}
        self.embedding_matrix = matrix
        self.is_norm = True

    def init_norms(self, force: bool = False) -> None:
        """ Initializes self.norms with pre-computed L2 normalized vectors for cosine distance computation.

        :param force: setting this to True will update the norms even if they were already computed
        :return: None
        """
        if not self.is_norm or force:
            # noinspection PyAttributeOutsideInit
            self.embedding_matrix = self.embedding_matrix / np.sqrt((self.embedding_matrix ** 2).sum(-1))[
                ..., np.newaxis]
            self.is_norm = True

    def _most_similar_cosine_measurement(self, vec):
        self.init_norms()
        normalized_vec = vec / np.linalg.norm(vec)
        return np.dot(self.embedding_matrix, normalized_vec)

    def most_similar_cosine(self, word_or_vec: WordOrVec, n: int = 20) -> MostSimilarResult:
        """ Calculate the cosine distance of the input vector to all vectors in the embedding matrix and return the
        most similar ones.

        :param word_or_vec: the input word or vector
        :param n: the number of results to return, or None if all should be returned
        :return: a list of MostSimilarResult objects
        """
        return self._generic_most_similar(word_or_vec, self._most_similar_cosine_measurement,
                                          higher_is_more_similar=True, n=n)

    def cosine_distance_rank(self, word_or_vec: WordOrVec, word):
        return self._generic_rank(word_or_vec, word, self._most_similar_cosine_measurement, higher_is_more_similar=True)

    def cosine_distance(self, vec: np.ndarray, word: str) -> float:
        """ Returns the cosine distance between an input word and vector.

        :param vec: the input vector
        :param word: the input word
        :return: a float between -1 and 1
        """
        self.init_norms()
        normalized_vec = vec / np.linalg.norm(vec)
        return float(np.dot(self.embedding_matrix[self.word2ind[word]], normalized_vec))

    def most_similar_l2(self, word_or_vec: WordOrVec, n: int = 20) -> MostSimilarResult:
        """ Calculate the L2 norm distance of the input vector to all vectors in the embedding matrix and return the
        most similar ones.

        :param word_or_vec: the input word or vector
        :param n: the number of results to return, or None if all should be returned
        :return: a list of (word, distance) pairs, with lower distance meaning more similar
        """

        def measurement(vec):
            distances = np.zeros(self.vocab_size)
            for i, emb in enumerate(self.embedding_matrix):
                distances[i] = np.linalg.norm(vec - emb)
            return distances

        return self._generic_most_similar(word_or_vec, measurement, higher_is_more_similar=False, n=n)

    def _lookup_if_needed(self, word_or_vec: WordOrVec) -> np.ndarray:
        if type(word_or_vec) == str:
            return self.lookup_embeddings.lookup(word_or_vec)
        else:
            return word_or_vec

    def _generic_most_similar(self, word_or_vec: WordOrVec, measurement, higher_is_more_similar, n: int = 20):
        self.init_norms()
        vec = self._lookup_if_needed(word_or_vec)
        distances = measurement(vec)
        assert len(distances) == len(self.embedding_matrix)
        if self.verbose:
            print('mean distance', np.mean(distances))

        distances_for_sorting = -distances if higher_is_more_similar else distances

        if n is None or n == len(self.embedding_matrix):
            sorted_most_similar_ind = np.argsort(distances_for_sorting)
        else:
            most_similar_ind = np.argpartition(distances_for_sorting, n)[:n]
            sorted_most_similar_ind = most_similar_ind[np.argsort(distances_for_sorting[most_similar_ind])]

        return [EmbeddingSimilarity(rank=rank,
                                    word=self.ind2word[ind],
                                    similarity=distances[ind],
                                    vec=self.embedding_matrix[ind])
                for rank, ind in enumerate(sorted_most_similar_ind, start=1)]

    def _generic_rank(self, word_or_vec: WordOrVec, word, measurement, higher_is_more_similar):
        self.init_norms()
        vec = self._lookup_if_needed(word_or_vec)
        distances = measurement(vec)
        distances = -distances if higher_is_more_similar else distances

        word_distance = distances[self.word2ind[word]]
        return np.count_nonzero(distances[distances < word_distance]) + 1
