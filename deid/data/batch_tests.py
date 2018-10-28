import numpy as np

from .batch import BatchGenerator, StratifiedSampling
from .util import compounding
from ..data import TrainingSet
from ..embeddings import DummyEmbeddings


def test_generator():
    batch_size = 2
    tr = TrainingSet(limit_documents=1, embeddings=DummyEmbeddings())

    generator = BatchGenerator(tr.X, tr.y, batch_size)
    x, y = next(generator)
    assert x.shape[0] == y.shape[0] == batch_size


def test_generator_yields_incomplete_batches():
    def make_array():
        return np.array([[[i for _ in range(10)] for _ in range(3)] for i in range(3)])

    generator = BatchGenerator(make_array(), make_array(), batch_size=2, yield_incomplete_batches=True)
    assert generator.epoch_length == 2
    x, y = next(generator)
    assert x.shape[0] == y.shape[0] == 2

    x, y = next(generator)
    assert x.shape[0] == y.shape[0] == 1

    x, y = next(generator)
    assert x.shape[0] == y.shape[0] == 2

    generator = BatchGenerator(make_array(), make_array(), batch_size=2, yield_incomplete_batches=False)
    assert generator.epoch_length == 1
    x, y = next(generator)
    assert x.shape[0] == y.shape[0] == 2

    x, y = next(generator)
    assert x.shape[0] == y.shape[0] == 2


def test_generator_compounding_batch_size():
    def make_array():
        return np.ones((100, 10, 1))

    generator = BatchGenerator(make_array(), make_array(), batch_size=compounding(1, 20, 1.1),
                              yield_incomplete_batches=False)
    compounding_value = 1

    sum = 0
    print('batch sizes:', generator.epoch_batch_sizes)
    print('epoch length:', generator.epoch_length)
    for i in range(40):  # 1 * 1.1**40 ≈ 45, so it's testing the maximum size as well
        compounding_value = min(20, int(1.1 ** i))
        x, y = next(generator)
        sum += x.shape[0]
        print(f'({i})', x.shape[0], '=', compounding_value, sum)
        assert x.shape[0] == y.shape[0] == int(compounding_value)

    assert compounding_value == 20


def test_generator_yields_permutation():
    def make_array():
        return np.arange(0, 100).reshape((10, 10, 1))

    x, y = make_array(), make_array()
    generator = BatchGenerator(x, y, batch_size=5, yield_indices=True)

    for _ in range(5):  # so we shuffle a couple of times
        batch_x, batch_y, batch_ind = next(generator)
        assert np.all(batch_x[0] == x[batch_ind[0]])


def test_stratified_sampling():
    def make_array():
        arr = np.zeros((100, 10, 1))
        for i in range(100):
            arr[i] = np.ones((10, 1)) * i
        return arr

    x, y = make_array(), make_array()
    generator = StratifiedSampling(x, y, split_condition=lambda x, _: x[-1] >= 20, batch_size=6, yield_indices=True)

    assert generator.epoch_length == 7

    batch_x, batch_y, batch_ind = next(generator)
    assert np.all(batch_x[0] == x[batch_ind[0]])
    assert batch_x.size == 60
    assert batch_x[batch_x >= 20].size == 30  # half of them

    for _ in range(1, generator.epoch_length):
        batch_x, batch_y, batch_ind = next(generator)
    assert batch_x.size == 40  # last batch should be incomplete

    batch_x, batch_y, batch_ind = next(generator)
    assert batch_x.size == 60  # next epoch
