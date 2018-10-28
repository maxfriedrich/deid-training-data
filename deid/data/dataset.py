import itertools
import logging
import os
from typing import Sequence, Optional, Set, Dict, Tuple, Callable, Iterator

import numpy as np
from tqdm import tqdm

from . import Sentence, SentenceLabels
from .augment import Augment
from .feature import Feature, apply_features
from .read import tokens_from_csv, full_text_for_csv, split_sentences, vocab_from_tokens
from .token import Token, TOKEN_TYPE, HIPAA_TOKEN_TYPE, BINARY_LABEL
from .util import pad_2d_sequences
from ..embeddings import Embeddings
from ..env import env

logger = logging.getLogger()


class DataSet:
    def __init__(self, X: Sequence[Sentence], y: Sequence[SentenceLabels], X_extra: Sequence[np.ndarray],
                 vocab: Set[str], dataset_id: str) -> None:
        self.X = X
        self.y = y
        self.X_extra = X_extra
        self.vocab = vocab
        self.mark_as_used(dataset_id, self.__class__.__name__)

    used_datasets = {}

    @classmethod
    def mark_as_used(cls, dataset_id, classname):
        used_classname = cls.used_datasets.get(dataset_id)
        if used_classname is None:
            cls.used_datasets[dataset_id] = classname
        elif used_classname != classname:
            raise RuntimeWarning(
                f"The dataset {dataset_id} is already in use by a {used_classname}, "
                f"you're trying to use it again in a {classname}.")


class TrainingSet(DataSet):
    def __init__(self,
                 embeddings: Optional[Embeddings] = None,
                 train_set: str = 'train',
                 augment: Optional[Augment] = None,
                 use_short_sentences: bool = False,
                 limit_documents: Optional[int] = None,
                 binary_classification: bool = False,
                 hipaa_only: bool = False,
                 extra_features: Sequence[Feature] = ()) -> None:
        """ Make a training set from the train directory with optional pre-trained embeddings.

         :param embeddings: an Embeddings object used for lookup, or None if the words should be returned
         :param train_set: the directory containing the train xmls (relative to the data directory)
         :param augment: an optional augmentation object or its string description that is used for embeddings lookup
         :param use_short_sentences: set to True for a smaller, faster training set (sentences < 64 tokens)
         :param limit_documents: a limit for the number of xml files to parse, or None if there should be no limit
         :param binary_classification: set to True to skip the BIO classes and use only yes/no labels
         """

        tokens = tokens_from_csv(train_set, limit=limit_documents, binary_classification=binary_classification,
                                 hipaa_only=hipaa_only)
        sents = split_sentences(tokens)

        if use_short_sentences:
            sents = [sent for sent in sents if len(sent) < 64]
        self.X_extra_size = sum(feature.dimension for feature in extra_features)
        X, self.augmented = prepare_sentences(sents, embeddings, augment)

        sent_labels = [[token.type for token in sent] for sent in sents]
        labels = self.labels(binary_classification, hipaa_only)
        self._label2ind = {label: i for i, label in enumerate(labels)}
        self._ind2label = {i: label for i, label in enumerate(labels)}

        y = [[[self.label2ind(label)] for label in sent] for sent in sent_labels]
        self.maxlen = max([len(sent) for sent in X])

        super().__init__(X, y, [apply_features(extra_features, sent) for sent in sents],
                         vocab=vocab_from_tokens(tokens), dataset_id=train_set)

    def label2ind(self, label: str) -> int:
        return self._label2ind[label]

    def ind2label(self, ind: int) -> str:
        return self._ind2label[ind]

    @property
    def output_size(self) -> int:
        return len(self._ind2label)

    @staticmethod
    def labels(binary_classification: bool, hipaa_only: bool) -> Sequence[str]:
        if binary_classification:
            return ['O', 'O', f'B-{BINARY_LABEL}', f'I-{BINARY_LABEL}']

        token_type = HIPAA_TOKEN_TYPE if hipaa_only else TOKEN_TYPE
        bi_labels = [(f'B-{v}', f'I-{v}') for v in sorted(token_type.keys())]
        return ['O', 'O'] + list(itertools.chain.from_iterable(bi_labels))

    @property
    def data_with_augmented(self):
        n_augmentations = len(next(augmented for augmented in self.augmented.values() if len(augmented) > 0))
        augmented_keys = sorted([key for key, value in self.augmented.items() if len(value) > 0])

        augmented_X = (self.augmented[i] for i in sorted(augmented_keys))
        X = list(self.X) + list(itertools.chain.from_iterable(augmented_X))

        augmented_y = ([self.y[i]] * n_augmentations for i in augmented_keys)
        y = list(self.y) + list(itertools.chain.from_iterable(augmented_y))

        augmented_X_extra = ([self.X_extra[i]] * n_augmentations for i in augmented_keys)
        X_extra = list(self.X_extra) + list(itertools.chain.from_iterable(augmented_X_extra))

        return X, y, X_extra


class ValidationSet(DataSet):
    def __init__(self,
                 label2ind: Callable[[str], int],
                 embeddings: Optional[Embeddings] = None,
                 validation_set: str = 'validation',
                 augment: Optional[Augment] = None,
                 use_short_sentences: bool = False,
                 limit_documents: Optional[int] = None,
                 binary_classification: bool = False,
                 hipaa_only: bool = False,
                 extra_features: Sequence[Feature] = ()) -> None:
        """ Makes a validation set from a directory with optional pre-trained embeddings.

        :param embeddings: an Embeddings object used for lookup, or None if no lookup should be performed
        :param validation_set: the directory containing the validation csvs (relative to the data directory)
        :param label2ind: the label -> index lookup function (from the training set)
        :param augment: an optional augmentation object that is used for embeddings lookup
        :param use_short_sentences: set to True for a smaller, faster training set (sentences < 64 tokens)
        :param limit_documents: a limit for the number of csv files to parse, or None if there should be no limit
        :param binary_classification: set to True to skip the BIO classes and use only yes/no labels
        """

        tokens = tokens_from_csv(validation_set, limit=limit_documents, binary_classification=binary_classification,
                                 hipaa_only=hipaa_only)
        sents = split_sentences(tokens)
        if use_short_sentences:
            sents = [sent for sent in sents if len(sent) < 64]

        X, self.augmented = prepare_sentences(sents, embeddings, augment)

        y: Sequence[SentenceLabels] = [[[label2ind(token.type) if label2ind is not None else token.type]
                                        for token in sent] for sent in sents]
        self.sents = [[(token.text, token.type) for token in sent] for sent in sents]
        super().__init__(X, y, [apply_features(extra_features, sent) for sent in sents],
                         vocab=vocab_from_tokens(tokens), dataset_id=validation_set)


class TestSet(DataSet):
    def __init__(self, X: Sequence[Sentence], y: Sequence[SentenceLabels], X_extra, filename: str, text: str,
                 sents: Sequence[Sequence[Token]], vocab: Set[str]) -> None:
        """ You should probably use TestSet.test_sets() instead of calling this directly. """
        super().__init__(X, y, X_extra, vocab, dataset_id=filename)
        self.filename = filename
        self.text = text
        self.sents = sents

    @classmethod
    def test_set_csvs(cls, test_set: str) -> Sequence[str]:
        test_set = os.path.join(env.data_dir, test_set)
        csvs = sorted([os.path.join(test_set, f) for f in os.listdir(test_set) if f.endswith('.csv')])
        if env.limit_validation_documents is None or len(csvs) < env.limit_validation_documents:
            return csvs
        return csvs[:env.limit_validation_documents]

    @classmethod
    def number_of_test_sets(cls, test_set: str = 'validation') -> int:
        test_set = os.path.join(env.data_dir, test_set)
        return len(cls.test_set_csvs(test_set))

    @classmethod
    def test_sets(cls,
                  embeddings: Optional[Embeddings],
                  label2ind: Callable[[str], int],
                  test_set: str = 'validation',
                  binary_classification: bool = False,
                  hipaa_only: bool = False,
                  extra_features: Sequence[Feature] = ()) -> Iterator['TestSet']:
        """ Yields test sets for every csv in the test directory based on the parameters of the training set.

        :param embeddings: an Embeddings object used for lookup, or None if no lookup should be performed
        :param label2ind: the label -> index lookup function (from the training set)
        :param test_set: the directory containing the test csvs (relative to the data directory)
        :param binary_classification: set to True to skip the BIO classes and use only yes/no labels

        :return: a TestSet object
        """
        test_set = os.path.join(env.data_dir, test_set)

        for filename in cls.test_set_csvs(test_set):
            text = full_text_for_csv(filename)
            tokens = tokens_from_csv(filename, binary_classification=binary_classification, hipaa_only=hipaa_only)
            sents = split_sentences(tokens)

            if embeddings is None:
                X = sents
            else:
                X = [embeddings.lookup_sentence([token.text for token in sent]) for sent in sents]
            X_extra = [apply_features(extra_features, sent) for sent in sents]

            sent_labels = [[token.type for token in sent] for sent in sents]
            y = [[[label2ind(label)] for label in sent] for sent in sent_labels]

            if embeddings is not None:
                maxlen = max(len(sent) for sent in sents)
                X = pad_2d_sequences(X, maxlen=maxlen)
                y = pad_2d_sequences(y, maxlen=maxlen)
                X_extra = pad_2d_sequences(X_extra, maxlen=maxlen)

            yield TestSet(X, y, X_extra, filename, text, sents,
                          vocab_from_tokens(tokens))


AlternativesDict = Dict[int, Sequence[Sentence]]


def prepare_sentences(sents: Sequence[Sentence],
                      embeddings: Optional[Embeddings],
                      augment: Optional[Augment]) -> Tuple[Sequence[Sentence],
                                                           Optional[AlternativesDict]]:
    if embeddings is None:  # there is also nothing to augment
        return sents, None

    if augment is None:
        return [embeddings.lookup_sentence([token.text for token in sent]) for sent in sents], None

    X = []
    augmented: Dict[int, Sequence[Sentence]] = {}

    if env.keras_verbose == 1:
        progress = tqdm(enumerate(sents), desc='Augmenting sentences', total=len(sents))
    else:
        progress = enumerate(sents)

    for i, sent in progress:
        augmented_sentence = augment.lookup_sentence(sent)
        X.append(augmented_sentence.original)
        augmented[i] = augmented_sentence.augmented
    return X, augmented


def is_phi_sentence(_, y):
    return any(label[0] > 1 for label in y)
