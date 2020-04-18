"""Methods for regularization to produce sparse networks.

L2 regularization mostly penalizes the weight magnitudes without introducing sparsity.
L1 regularization promotes sparsity.
L1/2 promotes sparsity even more than L1. However, it can be difficult to train due to non-convexity and exploding
gradients close to 0. Thus, we introduce a smoothed L1/2 regularization to remove the exploding gradients."""

import tensorflow as tf


def l1(input_tensor):
    if type(input_tensor) == list:
        return sum([l1(tensor) for tensor in input_tensor])
    return tf.reduce_sum(tf.abs(input_tensor))


def l2_norm(input_tensor):
    if type(input_tensor) == list:
        return sum([l2_norm(tensor) for tensor in input_tensor])
    return tf.reduce_sum(tf.square(input_tensor))


def l12_norm(input_tensor):
    """L1/2, or L0.5, norm. Note that the gradients go to infinity as the weight approaches 0, so this regularization
    is unstable during training. Use l12_smooth instead."""
    if type(input_tensor) == list:
        return sum([l12_norm(tensor) for tensor in input_tensor])
    return tf.reduce_sum(tf.pow(tf.abs(input_tensor), 0.5))


def piecewise_l12_l2(input_tensor, a=0.05):
    if type(input_tensor) == list:
        return sum([piecewise_l12_l2(tensor, a) for tensor in input_tensor])
    l2 = tf.square(input_tensor)
    l12 = tf.pow(tf.abs(input_tensor), 0.5)
    return tf.reduce_sum(tf.where(input_tensor < a, l2, l12))


def l12_smooth(input_tensor, a=0.05):
    """Smoothed L1/2 norm"""
    if type(input_tensor) == list:
        return sum([l12_smooth(tensor) for tensor in input_tensor])

    smooth_abs = tf.where(tf.abs(input_tensor) < a,
                          tf.pow(input_tensor, 4)/(-8*a**3) + tf.square(input_tensor)*3/4/a + 3*a/8,
                          tf.abs(input_tensor))
    return tf.reduce_sum(tf.sqrt(smooth_abs))
