from typing import Sequence, Union
import numpy as np
from .token import Token

Sentence = Sequence[Union[Token, np.ndarray]]
SentenceLabels = Sequence[Sequence[int]]
