import numpy as np

from .util import pad_string_sequences, unpad_sequences
from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences


def test_pad_string_sequences():
    test_seq = [['apple', 'banana', 'cherry'], ['d', 'e', 'f', 'g'], ['h', 'i', 'j', 'k', 'l', 'q'], ['r']]
    padded, seq_length = pad_string_sequences(test_seq)
    assert len(padded) == 4
    assert len(padded[0]) == 6
    assert padded[0][0] == 'apple'
    assert padded[0][3] == ''
    assert seq_length == [3, 4, 6, 1]


def test_unpad_sequences():
    test_seq = [['apple', 'banana', 'cherry', '', ''], ['d', 'e', 'f', 'g', 'h'], ['i', '', '', '', '', ]]
    seq = unpad_sequences(test_seq, [3, 5, 1])
    assert len(seq) == 3
    assert seq[0] == ['apple', 'banana', 'cherry']


def test_is_reverse_operation():
    test_seq = [[0, 1, 2, 3], [4], [5, 6]]
    padded = keras_pad_sequences(test_seq, padding='post')
    unpadded = unpad_sequences(padded, [4, 1, 2])
    assert [list(item) for item in unpadded] == test_seq
