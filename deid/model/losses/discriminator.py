from keras import backend as K
from keras.losses import binary_crossentropy


def discriminator_loss(y_true, y_pred):
    """ Compares the actual binary crossentropy loss to the random guessing loss (0.6931..., accuracy 0.5) and returns
    the maximum. This is motivated by the fact that our adversarial discriminators should not be worse than random
    guessing, otherwise we could just flip every prediction and get a better discriminator.
    """
    loss = binary_crossentropy(y_true, y_pred)
    random_guessing = -K.log(0.5)
    return K.maximum(loss, random_guessing)
