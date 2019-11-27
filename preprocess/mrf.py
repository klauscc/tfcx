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


def normalize_immrf(immrf):
    """normalize the immrf

    Args:
        immrf: Tensor of shape [h, w, c] 

    Returns: normalized immrf. The same shape as input.

    """
    mean = tf.math.reduce_mean(immrf**2, axis=[-1], keepdims=True) * 2
    immrf = immrf / (mean**0.5) / 36.0
    # mean = tf.math.reduce_sum(immrf**2, axis=[-1], keepdims=True)
    # immrf = immrf / (mean**0.5)
    return immrf


def normalize_tmap(tmap):
    """normalize the tmap to [0,1] 

    Args:
        tmap: Tensor of shape [h,w,2]. 

    Returns: normalized tmap. The same shape as input.

    """
    max_value = [5000, 500]
    return tmap / max_value


def denormalize_tmap(tmap):
    """denormalize tmap to the original scale.

    Args:
        tmap: Tensor of shape [h,w,2]. 

    Returns: denormalized tmap. The same shape as input.

    """
    return tmap * [5000, 500]
