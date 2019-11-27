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

from .pytorch_initial import PytorchInitializer


class Conv2D(keras.layers.Conv2D):
    """docstring for Conv2D"""

    def __init__(self, *args, **kwargs):

        initializer = PytorchInitializer()

        if "kernel_initializer" not in kwargs:
            kwargs["kernel_initializer"] = initializer
        if "bias_initializer" not in kwargs:
            kwargs["bias_initializer"] = initializer
        super(Conv2D, self).__init__(*args, **kwargs)


class Conv2DTranspose(keras.layers.Conv2DTranspose):
    """docstring for Conv2DTranspose"""

    def __init__(self, *args, **kwargs):

        initializer = PytorchInitializer()

        if "kernel_initializer" not in kwargs:
            kwargs["kernel_initializer"] = initializer
        if "bias_initializer" not in kwargs:
            kwargs["bias_initializer"] = initializer
        super(Conv2DTranspose, self).__init__(*args, **kwargs)


class Conv1D(keras.layers.Conv1D):
    """docstring for Conv1D"""

    def __init__(self, *args, **kwargs):

        initializer = PytorchInitializer()

        if "kernel_initializer" not in kwargs:
            kwargs["kernel_initializer"] = initializer
        if "bias_initializer" not in kwargs:
            kwargs["bias_initializer"] = initializer
        super(Conv1D, self).__init__(*args, **kwargs)
