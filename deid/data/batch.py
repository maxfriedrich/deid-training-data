import itertools
import math
import random
from typing import Generic, TypeVar, Optional
from typing import Sequence, Union, Tuple, Callable, Dict, Iterator, List

import numpy as np

from .util import pad_2d_sequences, peek

X_type = TypeVar('X_type')
y_type = TypeVar('y_type')

TwoArrays = Tuple[np.ndarray, np.ndarray]
ThreeArrays = Tuple[np.ndarray, np.ndarray, np.ndarray]


class IteratorWithEpochLength(Iterator):
    def __next__(self) -> Union[TwoArrays, ThreeArrays]:
        raise NotImplementedError

    @property
    def epoch_length(self) -> int:
        raise NotImplementedError


class IteratorWithEpochLengthImpl(Generic[X_type, y_type], IteratorWithEpochLength):
    def __init__(self,
                 X: Sequence[X_type],
                 y: Sequence[y_type],
                 total_size: int,
                 batch_size_iter: Iterator[int],
                 yield_incomplete_batches: bool = True,
                 yield_indices: bool = False,
                 augment: Optional[Dict[int, Sequence[X_type]]] = None,
                 augment_include_original: bool = True) -> None:
        assert len(X) == len(y)
        self.X, self.y = X, y
        self.augment = augment
        self.augment_include_original = augment_include_original

        self.total_size = total_size
        self.batch_size_iter = batch_size_iter
        self.yield_indices = yield_indices
        self.yield_incomplete_batches = yield_incomplete_batches
        self.init_epoch()

    def __next__(self) -> Union[TwoArrays, ThreeArrays]:
        if self.batch_number == self.epoch_length:
            self.init_epoch()

        current_batch_size = self.epoch_batch_sizes[self.batch_number]
        end = min(self.cursor + current_batch_size, self.total_size)
        batch_ind = self.select_batch_ind(self.cursor, end)

        if self.augment is not None:
            if self.augment_include_original:
                batch_X = [random.choice(self.augment[i] + [self.X[i]]) for i in batch_ind]
            else:
                batch_X = [random.choice(self.augment[i]) if len(self.augment[i]) > 0 else self.X[i] for i in batch_ind]
        else:
            batch_X = [self.X[i] for i in batch_ind]
        batch_y = [self.y[i] for i in batch_ind]
        self.cursor += current_batch_size
        self.batch_number += 1

        batch_X, batch_y = pad_2d_sequences(batch_X), pad_2d_sequences(batch_y)
        if self.yield_indices:
            return batch_X, batch_y, batch_ind
        else:
            return batch_X, batch_y

    def select_batch_ind(self, cursor, end) -> np.ndarray:
        raise NotImplementedError

    def __iter__(self):
        return self

    @property
    def epoch_length(self) -> int:
        return len(self.epoch_batch_sizes)

    # noinspection PyAttributeOutsideInit
    def init_epoch(self):
        self.batch_number = self.cursor = 0
        self.epoch_batch_sizes = self._make_epoch_batch_sizes(self.total_size)

    def _make_epoch_batch_sizes(self, total_size):
        """ Take items from the batch size iter until they make an epoch."""
        result = []
        seen = 0
        while seen < total_size:
            if self.yield_incomplete_batches:
                size = min(next(self.batch_size_iter), total_size - seen)
                seen += size
                result.append(size)
            else:
                size, self.batch_size_iter = peek(self.batch_size_iter)
                if seen + size > total_size:
                    break
                size = next(self.batch_size_iter)
                seen += size
                result.append(size)

        assert seen == total_size if self.yield_incomplete_batches else seen <= total_size
        return result


class BatchGenerator(IteratorWithEpochLengthImpl):
    def __init__(self,
                 X: Sequence[X_type],
                 y: Sequence[y_type],
                 batch_size: Union[int, Iterator[int]],
                 shuffle: bool = True,
                 **kwargs) -> None:

        self.shuffle = shuffle

        if isinstance(batch_size, int):
            batch_size_iter = itertools.repeat(batch_size)
        else:
            batch_size_iter = batch_size
        super().__init__(X, y, total_size=len(X), batch_size_iter=batch_size_iter, **kwargs)

    # noinspection PyAttributeOutsideInit
    def init_epoch(self):
        super().init_epoch()
        if self.shuffle:
            self.shuffled_ind = np.random.permutation(np.arange(len(self.X)))
        else:
            self.shuffled_ind = np.arange(len(self.X))

    def select_batch_ind(self, cursor, end):
        return self.shuffled_ind[cursor:end]


class BatchGeneratorWithExtraFeatures(BatchGenerator):
    def __init__(self,
                 X: Sequence[X_type],
                 y: Sequence[y_type],
                 X_extra,
                 batch_size: Union[int, Iterator[int]],
                 **kwargs) -> None:
        self.X_extra = X_extra
        super().__init__(X, y, batch_size=batch_size, yield_indices=True, **kwargs)

    def __next__(self):
        X, y, ind = super().__next__()
        return [X, pad_2d_sequences([self.X_extra[i] for i in ind])], y


class StratifiedSampling(IteratorWithEpochLengthImpl):
    def __init__(self,
                 X: Sequence[X_type],
                 y: Sequence[y_type],
                 batch_size: Union[int, Iterator[int]],
                 split_condition: Callable[[X_type, y_type], bool],
                 shuffle: bool = False,
                 **kwargs) -> None:
        self.X_pos_ind, self.X_neg_ind = self.split_indices(X, y, split_condition)
        self.shorter_partition_size = min(len(self.X_pos_ind), len(self.X_neg_ind))

        self.shuffle = shuffle

        if isinstance(batch_size, int):
            batch_size_iter = itertools.repeat(math.ceil(batch_size / 2))
        else:
            double_batch_size_iter: Iterator[int] = batch_size
            batch_size_iter = (math.ceil(size / 2) for size in double_batch_size_iter)

        super().__init__(X, y, total_size=self.shorter_partition_size, batch_size_iter=batch_size_iter, **kwargs)

    # noinspection PyAttributeOutsideInit
    def init_epoch(self):
        super().init_epoch()
        if self.shuffle:
            self.shuffled_pos = np.random.permutation(self.X_pos_ind)
            self.shuffled_neg = np.random.permutation(self.X_neg_ind)
        else:
            self.shuffled_pos, self.shuffled_neg = self.X_pos_ind, self.X_neg_ind

    def select_batch_ind(self, cursor, end):
        return np.concatenate((self.shuffled_pos[cursor:end], self.shuffled_neg[cursor:end]), axis=0)

    @staticmethod
    def split_indices(X: Sequence[X_type],
                      y: Sequence[y_type],
                      split_condition: Callable[[X_type, y_type], bool]) -> Tuple[Sequence[int], Sequence[int]]:
        pos: List[int] = []
        neg: List[int] = []
        for i in range(len(X)):
            (pos if split_condition(X[i], y[i]) else neg).append(i)
        return pos, neg


class StratifiedSamplingWithExtraFeatures(StratifiedSampling):
    def __init__(self,
                 X: Sequence[X_type],
                 y: Sequence[y_type],
                 X_extra,
                 batch_size: Union[int, Iterator[int]],
                 **kwargs) -> None:
        self.X_extra = X_extra
        super().__init__(X, y, batch_size=batch_size, yield_indices=True, **kwargs)

    def __next__(self):
        X, y, ind = super().__next__()
        return [X, pad_2d_sequences([self.X_extra[i] for i in ind])], y


def fake_sentences_batch(X: np.ndarray,
                         y: np.ndarray,
                         indices: np.ndarray,
                         alternatives: Dict[int, Sequence[np.ndarray]],
                         split_condition: Callable[[np.ndarray, np.ndarray], bool]) -> ThreeArrays:
    """ Generate a batch of real and fake/augmented sentence pairs.

    :param X: the complete X array
    :param y: the complete y array
    :param indices: the indices of this batch
    :param alternatives: a dictionary (index -> sequence of alternatives) providing fake alternatives for each index
    :param split_condition: a condition determining if the sentence should be used
    :return: A batch `X_1, X_2, y`
    """

    indices = [i for i in indices if split_condition(X[i], y[i])]
    real_sentences = [X[i] for i in indices]
    fake_sentences = [random.choice(alternatives[ind]) for ind in indices]

    X_1: List[np.ndarray] = []
    X_2: List[np.ndarray] = []
    y = []
    for real, fake in zip(real_sentences, fake_sentences):
        X_1 += [real, real]
        X_2 += [real, fake]
        y += [1, 0]

    return pad_2d_sequences(X_1), pad_2d_sequences(X_2), np.array(y)
