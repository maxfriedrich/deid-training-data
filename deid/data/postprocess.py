import itertools
from typing import List, NamedTuple, Sequence, Tuple, Optional

import numpy as np
# noinspection PyProtectedMember
from bs4 import CData, BeautifulSoup

from .token import Token, TOKEN_TYPE


class TaggedTokens(NamedTuple):
    type: str
    tokens: List[Token]
    start: int
    end: int


class TagAssembler:
    def __init__(self, sent_tokens: Sequence[Token]) -> None:
        self.input = sent_tokens
        self.result: List[TaggedTokens] = []
        self.current_tag: Optional[str] = None
        self.current_tag_tokens: List[Token] = []

    def close_current_tag(self) -> None:
        if self.current_tag is not None:
            self.result.append(TaggedTokens(self.current_tag,
                                            self.current_tag_tokens,
                                            self.current_tag_tokens[0].start,
                                            self.current_tag_tokens[-1].end))
            self.current_tag = None
            self.current_tag_tokens = []

    def assemble(self) -> Sequence[TaggedTokens]:
        for t in self.input:
            if t.type == 'O':
                self.close_current_tag()
            elif t.type.startswith('I') and self.current_tag == t.type[2:]:
                self.current_tag_tokens.append(t)
            else:  # B tag or a stray I tag that should be normalized to a B
                self.close_current_tag()
                self.current_tag = t.type[2:]
                self.current_tag_tokens.append(t)

        self.close_current_tag()
        return self.result


def unpad(X, preds) -> Tuple[List, List]:
    assert len(X) == len(preds), f'X and preds have different lengths: {len(X)} != {len(preds)} '
    unpadded_X, unpadded_preds = [], []
    for i in range(len(X)):
        if isinstance(X[i], np.ndarray):
            actual_length = np.sum(X[i].any(axis=1))
            X_start = preds_start = len(X[i]) - actual_length
        else:
            X_start = 0
            preds_start = len(preds[i]) - len(X[i])
        unpadded_X.append(list(X[i][X_start:]))
        unpadded_preds.append(list(preds[i][preds_start:]))
        assert len(unpadded_X[i]) == len(unpadded_preds[i])
    return unpadded_X, unpadded_preds


def postprocess_prediction(X, preds, sents, ind2label_lookup) -> Sequence[Sequence[TaggedTokens]]:
    X, preds = unpad(X, preds)

    result = []
    for i in range(len(X)):
        sent_tokens = []
        for j in range(len(X[i])):
            sent_tokens.append(
                Token(sents[i][j].text, ind2label_lookup(preds[i][j]), sents[i][j].start, sents[i][j].end))
        result.append(sent_tokens)

    return [TagAssembler(sent_tokens).assemble() for sent_tokens in result]


def prediction_to_xml(X, preds, text, sents, ind2label_lookup) -> str:
    preds = postprocess_prediction(X, preds, sents, ind2label_lookup)

    soup = BeautifulSoup('<deIdi2b2><TEXT></TEXT><TAGS></TAGS></deIdi2b2>', features='xml')
    soup.find('TEXT').string = CData(text)
    tags = soup.find('TAGS')
    for i, tagged_tokens in enumerate(itertools.chain.from_iterable(preds)):
        tags.append(soup.new_tag(TOKEN_TYPE[tagged_tokens.type],
                                 id=f'P{i}',
                                 start=tagged_tokens.start,
                                 end=tagged_tokens.end,
                                 TYPE=tagged_tokens.type,
                                 text=text[tagged_tokens.start:tagged_tokens.end]))

    return str(soup)
