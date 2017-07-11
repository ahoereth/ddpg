import tensorflow as tf
import numpy as np


def to_tuple(*args):
    result = []
    for arg in args:
        try:
            result.extend(arg)
        except TypeError:
            result.extend((arg,))
    return tuple(result)


def selu(x):
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def to_tf_dtype(dtype):
    np2tf = {
        np.float: tf.float32,
        np.uint8: tf.uint8,
    }
    return np2tf.get(dtype, dtype)
