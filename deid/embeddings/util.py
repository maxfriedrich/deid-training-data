from typing import Sequence, Tuple, List, Any


def pad_string_sequences(seq: Sequence[Sequence[str]]) -> Tuple[List[List[str]], Sequence[int]]:
    """ Like keras.preprocessing.sequence.pad_string_sequences but for strings, and it also returns seq_length. """

    seq_length = [len(item) for item in seq]
    maxlen = max(seq_length)

    result = []
    for i, item in enumerate(seq):
        result.append(list(item) + [''] * (maxlen - seq_length[i]))
    return result, seq_length


def unpad_sequences(padded: Sequence[Any], seq_length: Sequence[int]):
    """ The reverse operation of `keras.preprocessing.sequence.pad_sequences`. """
    assert len(padded) == len(seq_length)
    return [padded[i][:seq_length[i]] for i in range(len(padded))]


# https://stackoverflow.com/a/434328/2623170
def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
