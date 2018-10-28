def get(identifier):
    if identifier == 'lstm':
        return make_lstm_crf
    elif identifier.startswith('adversarial'):
        return AdversarialModel
    else:
        raise ValueError('unknown identifier:', identifier)


from .adversarial import AdversarialModel
from .deidentifier import make_lstm_crf
