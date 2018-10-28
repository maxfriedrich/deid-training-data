import itertools
from typing import Sequence, Optional, Any
from spacy.util import compounding as spacy_compounding
import numpy as np


def one_hot(x: int, n: int) -> np.ndarray:
    result = np.zeros(n)
    result[x] = 1
    return result


def compounding(start, stop, compound):
    """ Wraps spaCy's compounding utility to always return ints.

      >>> sizes = compounding(1., 10., 1.5)
      >>> assert next(sizes) == 1.
      >>> assert next(sizes) == int(1 * 1.5)
      >>> assert next(sizes) == int(1.5 * 1.5)
    """
    return (int(result) for result in spacy_compounding(start, stop, compound))


def peek(iterator):
    item = next(iterator)
    return item, itertools.chain([item], iterator)


def pad_2d_sequences(seq: Sequence[Any], maxlen: Optional[int] = None,
                     embedding_size: Optional[int] = None) -> np.ndarray:
    """ Like keras.preprocessing.sequence.pad_sequences but for 2d (already embedded) sequences.

    Caveat: this function does not truncate inputs. An error will be raised if the specified maxlen is smaller than the
    actual maximum length in the sequence.

    :param seq: the input sequence
    :param maxlen: the length to which the result will be padded, may be None
    :param embedding_size: the embedding dimension of the input, may be None
    :return: a padded array
    """

    # find the maximum length by looking through the sequence
    if maxlen is None:
        maxlen = -1
        for item in seq:
            maxlen = max(maxlen, len(item))

    # find the embedding dimension by looking through the sequence until there is a non-empty item
    if embedding_size is None:
        for item in seq:
            if len(item) != 0:
                embedding_size = len(item[0])
                break

    result = np.zeros((len(seq), maxlen, embedding_size))
    for i, item in enumerate(seq):
        assert len(item) > 0
        result[i, -len(item):] = item
    return result
