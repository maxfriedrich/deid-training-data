# https://github.com/linxihui/keras-contrib/blob/7c9fc2124f3a6c6d821f2f3c5c437a38072c5ded/keras_contrib/losses/crf_losses.py

# Specify this crf_loss function in the custom_objects dict when loading keras models with a CRF layer
# until https://github.com/keras-team/keras-contrib/pull/272 is merged :)


from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from keras_contrib import backend as K


# noinspection PyProtectedMember,PyProtectedMember,PyProtectedMember,PyProtectedMember
def crf_nll(y_true, y_pred):
    """Usual Linear Chain CRF negative log likelihood. Used for CRF "join" mode. See `layers.CRF` for usage."""
    crf, idx = y_pred._keras_history[:2]
    assert not crf._outbound_nodes, 'When learn_model="join", CRF must be the last layer.'
    if crf.sparse_target:
        y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), crf.units)
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
    return nloglik


# noinspection PyProtectedMember
def crf_loss(y_true, y_pred):
    """General CRF loss function, depending on the learning mode."""
    crf, idx = y_pred._keras_history[:2]
    if crf.learn_mode == 'join':
        return crf_nll(y_true, y_pred)
    else:
        if crf.sparse_target:
            return sparse_categorical_crossentropy(y_true, y_pred)
        else:
            return categorical_crossentropy(y_true, y_pred)
