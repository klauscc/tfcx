# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2019/09/30
#   description:
#
#================================================================

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class PytorchInitializer(keras.initializers.Initializer):
    """initialize kernel as pytorch"""

    def __init__(self, *args, **kwargs):
        super(PytorchInitializer, self).__init__(*args, **kwargs)
        self.limit = None

    def __call__(self, shape, dtype=None):
        dim = len(shape)
        if dim == 1:    # bias
            if not self.limit:
                self.limit = np.sqrt(1. / shape[0])
        else:    # kernel
            k = 1.0
            for i in range(dim - 1):
                k *= shape[i]
            self.limit = np.sqrt(1 / k)

        return K.random_uniform(shape, -self.limit, self.limit)


def pytorch_init_batch_norm_layer():
    """batch norm with the same params as pytorch.
    Returns: TODO

    """
    return keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)


def pytorch_initial_layer(layer, *args, **kwargs):
    """TODO: Docstring for pytorch_initial_layer.

    Args:
        layer: tf.keras.layers.Layer

    Returns: TODO

    """
    initializer = PytorchInitializer()

    if "kernel_initializer" not in kwargs:
        kwargs["kernel_initializer"] = initializer
    if "bias_initializer" not in kwargs:
        kwargs["bias_initializer"] = initializer
    return layer(*args, **kwargs)
