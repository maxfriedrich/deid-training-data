import numpy as np

from .augment import Augment
from .strategy import AugmentWord, AugmentEmbedding
from ...data import Token
from ...embeddings import DummyEmbeddings


class Ones(AugmentEmbedding):
    def augment(self, word_embedding):
        return np.ones(len(word_embedding))


def test_augment_embeddings():
    embeddings = DummyEmbeddings()
    augment = Augment(embeddings, Ones(), exclude=None, n_augmentations=2)
    sent = [Token.with_text('this'), Token.with_text('is'), Token.with_text('a'), Token.with_text('test')]

    result = augment.lookup_sentence(sent)
    assert len(result.original) == 4
    assert len(result.original[0]) == embeddings.size
    assert len(result.augmented) == 0

    sent = [Token.with_text('this'), Token.with_text('is'), Token.with_text('a', 'B-NAME'),
            Token.with_text('name', 'I-NAME')]
    result = augment.lookup_sentence(sent)
    augmented = result.augmented[0]

    assert len(augmented) == 4
    assert len(augmented[0]) == embeddings.size
    assert np.all(augmented[2] == np.ones(embeddings.size))
    assert np.all(augmented[3] == np.ones(embeddings.size))
    assert len(result.augmented) == 2


class ReplaceWithFixed(AugmentWord):
    def augment(self, word):
        return 'REPLACED'


def test_augment_words():
    embeddings = DummyEmbeddings()
    augment = Augment(embeddings, ReplaceWithFixed(), exclude=None)
    sent = [Token.with_text('replace'), Token.with_text('these', 'B-NAME'), Token.with_text('words', 'I-NAME')]
    result = augment.lookup_sentence(sent).augmented[0]

    assert np.all(result[0] == embeddings.lookup('replace'))

    assert np.any(result[1] != embeddings.lookup('these'))
    assert np.all(result[1] == embeddings.lookup('REPLACED'))

    assert np.any(result[2] != embeddings.lookup('words'))
    assert np.all(result[2] == embeddings.lookup('REPLACED'))


def test_augment_exclude():
    embeddings = DummyEmbeddings()
    augment = Augment(embeddings, Ones())
    sent = [Token.with_text('Please'), Token.with_text('ignore'), Token.with_text('this', 'B-NAME'),
            Token.with_text(':', 'I-NAME'), Token.with_text('stopword', 'I-NAME')]

    result = augment.lookup_sentence(sent).augmented[0]
    assert np.all(result[2] != np.ones(embeddings.size))
    assert np.all(result[3] != np.ones(embeddings.size))
    assert np.all(result[4] == np.ones(embeddings.size))


def test_augment_all():
    embeddings = DummyEmbeddings()
    augment = Augment(embeddings, Ones(), augment_all=True, exclude=None)
    sent = [Token.with_text('Augment'), Token.with_text('all'), Token.with_text('of', 'B-NAME'),
            Token.with_text('these', 'I-NAME')]

    result = augment.lookup_sentence(sent).augmented[0]
    assert np.all(result[0] == np.ones(embeddings.size))
    assert np.all(result[1] == np.ones(embeddings.size))
    assert np.all(result[2] == np.ones(embeddings.size))
    assert np.all(result[3] == np.ones(embeddings.size))


def test_augment_does_not_touch_unknown():
    class DummyEmbeddingsWithUnknownTestWord(DummyEmbeddings):
        def is_unknown(self, word: str):
            return word == 'test'

        def lookup(self, word):
            if word == 'test':
                return np.zeros(self.size)
            return super().lookup(word)

    embeddings = DummyEmbeddingsWithUnknownTestWord()
    augment = Augment(embeddings, Ones(), exclude=None)
    sent = [Token.with_text('This', 'B-NAME'), Token.with_text('is', 'I-NAME'), Token.with_text('another', 'I-NAME'),
            Token.with_text('test', 'I-NAME')]
    result = augment.lookup_sentence(sent).augmented[0]
    assert np.any(result[2] == np.ones(embeddings.size))
    assert np.all(result[3] == np.zeros(embeddings.size))


def test_augment_max():
    embeddings = DummyEmbeddings()
    augment = Augment(embeddings, ReplaceWithFixed(), augment_max=1, exclude=None)
    sent = [Token.with_text('Augment'), Token.with_text('only'), Token.with_text('one', 'B-NAME'),
            Token.with_text('please', 'I-NAME')]
    result = augment.lookup_sentence(sent).augmented[0]
    assert len([r for r in result if np.all(r == embeddings.lookup('REPLACED'))]) == 1
