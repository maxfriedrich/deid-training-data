def setup_random():
    import os
    import random

    import numpy as np
    import tensorflow as tf

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1)
    random.seed(2)
    tf.set_random_seed(3)
