import logging
import random
import re
from typing import Optional, Callable, Sequence, NamedTuple, Tuple, Dict, Union

from spacy.lang.en.stop_words import STOP_WORDS

from . import AugmentStrategy, get as get_strategy
from .. import Token, Sentence
from ...embeddings import Embeddings

logger = logging.getLogger()
digit_pattern = '^[0-9]*$'


def default_exclude(word: str) -> bool:
    return word.lower() in STOP_WORDS or bool(re.match('^[.,:;/+\-*=\\\\]*$', word))


def exclude_nothing(_: str) -> bool:
    return False


class AugmentedSentence(NamedTuple):
    original: Sentence
    augmented: Sequence[Sentence]


class Augment:
    def __init__(self, embeddings: Embeddings,
                 strategy: Union[AugmentStrategy, str],
                 digit_strategy: Optional[Union[AugmentStrategy, str]] = None,
                 n_augmentations: int = 1,
                 augment_all: bool = False,
                 augment_max: Optional[int] = None,
                 exclude_unknown: bool = True,
                 exclude: Optional[Callable[[str], bool]] = default_exclude) -> None:
        self.embeddings = embeddings
        self.augment_all = augment_all
        self.exclude_unknown = exclude_unknown
        if isinstance(strategy, str):
            self.strategy = get_strategy(strategy)
        else:
            self.strategy = strategy

        if digit_strategy is None:
            self.digit_strategy = self.strategy
        elif isinstance(digit_strategy, str):
            self.digit_strategy = get_strategy(digit_strategy)
        else:
            self.digit_strategy = digit_strategy

        self.n_augmentations = n_augmentations
        self.augment_max = augment_max if augment_max is not None else 10_000
        self.exclude = exclude if exclude is not None else exclude_nothing

    def __str__(self) -> str:
        return f'<Augment embeddings={self.embeddings.__class__.__name__}, strategy={self.strategy}, ' \
               f'digit_strategy={self.digit_strategy}, n_augmentations={self.n_augmentations}, ' \
               f'augment_all={self.augment_all}, exclude_unknown={self.exclude_unknown}>'

    def _strategy_or_digit_strategy(self, word: str) -> AugmentStrategy:
        if re.match(digit_pattern, word):
            return self.digit_strategy
        return self.strategy

    def _should_be_excluded(self, word, label):
        exclude_because_o = not self.augment_all and label == 'O'
        exclude_because_unknown = self.exclude_unknown and self.embeddings.is_unknown(word)
        return self.exclude(word) or exclude_because_o or exclude_because_unknown

    def lookup_sentence(self, sentence: Sequence[Token]) -> AugmentedSentence:
        """ If the sentence is only O, just look it up. Otherwise:
        - apply the word strategies and keep track of the embedding strategies that need to be applied later
        - look up the sentence
        - apply the embedding strategies

        :param sentence: the input sentence
        :return: an AugmentedSentence object
        """
        original = self.embeddings.lookup_sentence([token.text for token in sentence])
        if not self.augment_all and all([token.type == 'O' for token in sentence]):
            return AugmentedSentence(original, [])

        apply_word_strategies_result = [self.apply_word_strategies(sentence) for _ in range(self.n_augmentations)]
        augment_embeddings, sentences_for_lookup = zip(*apply_word_strategies_result)
        embedded_sentences = self.embeddings.lookup_sentences(sentences_for_lookup)
        augmented = [self.apply_embedding_strategies(augment_embedding, embedded_sent) for
                     augment_embedding, embedded_sent in zip(augment_embeddings, embedded_sentences)]
        return AugmentedSentence(original, augmented)

    def apply_embedding_strategies(self, augment_embedding: Dict[int, AugmentStrategy],
                                   sentence_embeddings: Sentence) -> Sentence:
        sentence_embeddings = list(sentence_embeddings)
        for i, strategy in augment_embedding.items():
            augmented = strategy.augment(sentence_embeddings[i])
            assert len(augmented) == self.embeddings.size
            sentence_embeddings[i] = augmented
        return sentence_embeddings

    def apply_word_strategies(self, sentence: Sequence[Token]) -> Tuple[Dict[int, AugmentStrategy], Sequence[str]]:
        sentence_for_lookup = []
        augment_embedding = {}

        augment_word_ind = []
        for i, token in enumerate(sentence):
            word, label = token.text, token.type
            if not self._should_be_excluded(word, label):
                strategy = self._strategy_or_digit_strategy(word)
                if strategy.augments_words:
                    augment_word_ind.append(i)
                else:
                    augment_embedding[i] = strategy
                    logger.info('deferring strategy %s to augment "%s"', strategy, word)

        if len(augment_word_ind) > self.augment_max:
            augment_word_ind = random.sample(augment_word_ind, self.augment_max)

        for i, token in enumerate(sentence):
            word, label = token.text, token.type
            if i in augment_word_ind:
                strategy = self._strategy_or_digit_strategy(word)
                augmented = strategy.augment(word)
                logger.info('using strategy %s to augment "%s" to "%s"', strategy, word, augmented)
                sentence_for_lookup.append(augmented)
            else:
                sentence_for_lookup.append(word)

        return augment_embedding, sentence_for_lookup
