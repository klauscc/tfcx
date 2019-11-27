# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2019/09/26
#   description:
#
#================================================================

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from .. import layers as mi_layers
from ..layers.conv_layers import *


def pytorch_conv_initializer(shape, dtype=None):
    dim = len(shape)
    c_in = shape[-2]
    k = 1.0
    for i in range(dim - 1):
        k *= shape[i]
    limit = np.sqrt(1 / k)
    return K.random_uniform(shape, -limit, limit)


class Pix2PixUnet(tf.keras.layers.Layer):
    """Unet
    
    Args:
        init_filters: Int. The initial filter number.
        kernel_size: Int. Kernel size used by conv layers.
        num_downsample: Int. The downsample times. Default to 3.
        **kwargs: The other arguments passed to the Layer.
    
    """

    def __init__(self,
                 init_filters,
                 output_channels,
                 kernel_size,
                 norm_type="batchnorm",
                 num_downsample=4,
                 **kwargs):
        super(Pix2PixUnet, self).__init__(**kwargs)

        self.down_stack = []
        self.up_stack = []
        self.down_stack = [
            downsample(init_filters, kernel_size, norm_type, first_down=True),    # / (64,64,64) 
            downsample(init_filters * 2, kernel_size, norm_type),    # / (32,32,128) 
            downsample(init_filters * 4, kernel_size, norm_type),    #  / (16,16,256) 
        ]

        self.bridge = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D((2, 2), padding="same"),
            conv_norm_relu(init_filters * 8, kernel_size=kernel_size, norm_type=norm_type),
            Conv2DTranspose(init_filters * 4, kernel_size, strides=2, padding='same')
        ])

        self.up_stack = [
            self.bridge,    # (16,16,256)
            upsample(init_filters * 2, kernel_size, norm_type),    # / (32,32,128) 
            upsample(init_filters, kernel_size, norm_type),    # / (64,64,64) 
        ]

        self.last = tf.keras.Sequential([
            Conv2D(init_filters, kernel_size, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
            tf.keras.layers.ReLU(),
            Conv2D(output_channels, kernel_size, padding="same")
        ])    # (64,64,output_channels)

        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, training=None):

        skips = []
        for down in self.down_stack:
            x = down(x, training)
            skips.append(x)

        skips = reversed(skips)
        for up, skip in zip(self.up_stack, skips):
            x = up(x, training)
            x = self.concat([x, skip])
        x = self.last(x, training)
        return x


def conv_norm_relu(filters,
                   *args,
                   kernel_size=3,
                   norm_type="batchnorm",
                   apply_norm=True,
                   apply_dropout=False,
                   **kwargs):
    net = tf.keras.Sequential()
    if norm_type.lower() == "batchnorm":
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type.lower() == "instancenorm":
        norm_layer = InstanceNormalization
    else:
        norm_layer = None
    net.add(Conv2D(filters, kernel_size, padding="same", use_bias=False, *args, **kwargs))
    if norm_layer and apply_norm:
        net.add(norm_layer(momentum=0.9, epsilon=1e-5))
    net.add(tf.keras.layers.ReLU())
    if apply_dropout:
        net.add(tf.keras.layers.Dropout(0.3))

    return net


def downsample(filters, size, norm_type='batchnorm', apply_norm=True, first_down=False):
    """Downsamples an input.
  Conv2D => Batchnorm => LeakyRelu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer
  Returns:
    Downsample Sequential Model
  """
    # initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    if not first_down:
        result.add(tf.keras.layers.MaxPool2D((2, 2), padding="same"))
    result.add(Conv2D(filters, size, strides=1, padding='same', use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.ReLU())

    return result


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

    # initializer = tf.random_normal_initializer(0., 0.02)

    if norm_type == "batchnorm":
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == "instancenorm":
        norm_layer = InstanceNormalization
    else:
        norm_layer = None

    result = tf.keras.Sequential()
    result.add(Conv2D(filters * 2, size, strides=1, padding="same", use_bias=False))
    if norm_layer:
        result.add(norm_layer(momentum=0.9, epsilon=1e-5))
    result.add(tf.keras.layers.ReLU())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(Conv2DTranspose(filters, size, strides=2, padding='same'))
    return result
