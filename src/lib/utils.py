import tensorflow as tf


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
