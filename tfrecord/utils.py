# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2019/09/23
#   description:
#
#================================================================
import tensorflow as tf


def reshape_back(d, key):
    """reshape back tensor of key.

    Args:
        d: Dict of Tensor. The d must contain key: `key/data`, `key/shape`
        key: The key of reshaped.

    Returns: Tensor of the shape `key/shape`.

    """
    return tf.reshape(d[key + "/data"], tf.sparse.to_dense(d[key + "/shape"]))
