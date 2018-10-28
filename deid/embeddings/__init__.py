from .embeddings import Embeddings, PrecomputedEmbeddings
from .dummy import DummyEmbeddings
from .elmo import ElmoEmbeddings, TensorFlowElmoEmbeddings, CachedElmoEmbeddings
from .fasttext import FastTextEmbeddings, PreloadFastTextEmbeddings, CachedFastTextEmbeddings
from .glove import GloveEmbeddings
from .matrix import Matrix, EmbeddingSimilarity
from .noise import Noise, GaussianNoise, DropoutNoise, NoiseWrapper


def get(identifier, *args, **kwargs):
    if identifier == 'dummy':
        return DummyEmbeddings()
    elif identifier == 'elmo':
        return ElmoEmbeddings(*args)
    elif identifier == 'elmo-tf':
        return TensorFlowElmoEmbeddings(*args, **kwargs)
    elif identifier == 'glove':
        return GloveEmbeddings(*args, **kwargs)
    elif identifier == 'fasttext':
        return FastTextEmbeddings(*args, **kwargs)
    else:
        raise ValueError('unknown identifier:', identifier)
