import hashlib

import numpy as np

from . import Embeddings


class DummyEmbeddings(Embeddings):
    @property
    def size(self):
        return 5

    @property
    def std(self):
        return 0.5

    def is_unknown(self, word: str) -> bool:
        return False

    def lookup(self, word: str):
        hashed = int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16)
        five_digits = [int(digit) for digit in str(hashed)[1:6]]  # omitting a possible - at the first index
        return np.array([-0.5 if digit < 5 else 0.5 for digit in five_digits])
