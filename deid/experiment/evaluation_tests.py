import os
import tempfile

from . import evaluation
from ..env import Test
from .evaluation import _run_official_evaluation

config = evaluation.env = Test()


def test_run_official_evaluation():
    with tempfile.NamedTemporaryFile() as f:
        # testing the fixtures train_xml directory against itself, resulting in perfect score
        results = _run_official_evaluation(predictions_dir=os.path.join(config.data_dir, 'train_xml'),
                                           test_set='train',
                                           output_file=f.name)
        assert len(f.read().strip()) != 0  # something was written to the evaluation file

    assert results['Token']['precision'] == 1.0
    assert results['Token']['recall'] == 1.0
    assert results['Token']['f1'] == 1.0
