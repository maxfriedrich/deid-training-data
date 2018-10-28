from .basic import basic_experiment
from .alternating import alternating_experiment
from .alternating_evaluation import alternating_evaluation_experiment
from .mtn_evaluation import mtn_evaluation_experiment
from .fake_sentences import fake_sentences_experiment


def get(identifier):
    if identifier == 'basic':
        return basic_experiment
    elif identifier == 'alternating':
        return alternating_experiment
    elif identifier == 'alternating_evaluation':
        return alternating_evaluation_experiment
    elif identifier == 'mtn_evaluation':
        return mtn_evaluation_experiment
    elif identifier == 'fake_sentences':
        return fake_sentences_experiment
    else:
        raise ValueError('unknown identifier:', identifier)
