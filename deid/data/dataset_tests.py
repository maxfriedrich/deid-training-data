from . import TrainingSet
from ..embeddings import DummyEmbeddings


def test_training_set():
    tr = TrainingSet(limit_documents=1, embeddings=DummyEmbeddings())
    assert len(tr.X) == len(tr.y)
