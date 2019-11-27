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

from .. import record_feature
from .. import utils


def mrf_slice_example(slice):
    """convert mrf slice data to tensorflow

    Args:
        slice: Dict. Contains keys:
                immrf/data, immrf/path,
                tmap/data, tmap/path,
                mask/data, mask/path

    Returns: TODO

    """
    immrf = slice["immrf/data"]
    tmap = slice["tmap/data"]
    mask = slice["mask/data"]
    slice_path = slice["immrf/path"]
    tmap_path = slice["tmap/path"]
    mask_path = slice["mask/path"]

    feature = {
        "immrf/data": record_feature.bytes_feature(immrf.tostring()),
        "immrf/shape": record_feature.int64_list_feature(immrf.shape),
        "immrf/path": record_feature.bytes_feature(tf.compat.as_bytes(slice_path)),
        "tmap/data": record_feature.bytes_feature(tmap.tostring()),
        "tmap/shape": record_feature.int64_list_feature(tmap.shape),
        "tmap/path": record_feature.bytes_feature(tf.compat.as_bytes(tmap_path)),
        "mask/data": record_feature.bytes_feature(mask.tostring()),
        "mask/shape": record_feature.int64_list_feature(mask.shape),
        "mask/path": record_feature.bytes_feature(tf.compat.as_bytes(mask_path)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_single_example(serialized_example):
    """parse serialized example to the original type

    Args:
        serialized_example: protobuf data.

    Returns: A dict.

    """
    feature_description = {
        "immrf/data": tf.io.FixedLenFeature([], tf.string),
        "immrf/shape": tf.io.VarLenFeature(tf.int64),
        "immrf/path": tf.io.FixedLenFeature([], tf.string),
        "tmap/data": tf.io.FixedLenFeature([], tf.string),
        "tmap/shape": tf.io.VarLenFeature(tf.int64),
        "tmap/path": tf.io.FixedLenFeature([], tf.string),
        "mask/data": tf.io.FixedLenFeature([], tf.string),
        "mask/shape": tf.io.VarLenFeature(tf.int64),
        "mask/path": tf.io.FixedLenFeature([], tf.string),
    }
    slice = tf.io.parse_single_example(serialized_example, feature_description)
    for key in ["immrf", "tmap", "mask"]:
        slice[key + "/data"] = tf.io.decode_raw(slice[key + "/data"], out_type=tf.float32)
        slice[key + "/data"] = utils.reshape_back(slice, key)
    return slice
