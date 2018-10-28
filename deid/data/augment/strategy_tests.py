import numpy as np

from .strategy import NeighborsCache
from ...embeddings import EmbeddingSimilarity


def test_neighbors_cache():
    cache = NeighborsCache('selected')
    assert cache.lookup('test') is None

    neighbors = [EmbeddingSimilarity(1, 'tests', 0.95, np.zeros(10)),
                 EmbeddingSimilarity(1, 'testing', 0.93, np.zeros(10)),
                 EmbeddingSimilarity(1, 'tester', 0.91, np.zeros(10))]

    cache.store('test', neighbors, neighbors[0])
    assert cache.lookup('test') == cache.lookup('test') == neighbors[0]

    cache = NeighborsCache('neighbors')
    assert cache.lookup('test') is None

    cache.store('test', neighbors, neighbors[0])
    assert cache.lookup('test') in neighbors
    assert cache.lookup('test') in neighbors
