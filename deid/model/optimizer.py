from keras.optimizers import Adam, Nadam, RMSprop, SGD

# We want to pass custom args to the adversaries. Passing a Keras optimizer string to the compile method won't let us
# select custom args, so we make a subset of optimizers available by string keys here.
optimizers = {'adam': Adam, 'nadam': Nadam, 'rmsprop': RMSprop, 'sgd': SGD}


def get(identifier):
    if identifier in optimizers.keys():
        return optimizers[identifier]
    raise ValueError(f'Unknown optimizer: {identifier}')
