#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import tensorflow as tf


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def EfficientCommbine(enc, pred, enc_len, pred_len):
    """ Ref: IMPROVING RNN TRANSDUCER MODELING FOR END-TO-END SPEECH RECOGNITION

    Instead, we implement the combination sequence by sequence. 
    Then we concatenate all zn instead of paralleling them, which means we convert z into a two-dimension tensor (􏰀Nn=1 Tn ∗ Un), D). 

    Args:
        enc: A 3-D Tensor of floats. The dimensions should be (B, T, V),
                     B is batch index, T is the time index, V is the encode units size.
        pred: A 3-D Tensor of floats. The dimensions should be (B, U, V),
                     B is batch index, U is the prediction network sequence length, V is the decode units size.
        enc_len: A 1-D Tensor of ints, the number of time steps
                       for each sequence in the minibatch.
        pred_len: A 1-D Tensor of ints, the length of each label
                       for each example in the minibatch.
    Returns:
        A 2-D Tensor of floats.  The dimensions should be (Sum(T_i*U_i), V),
                                 T is the time index, U is the prediction network sequence
                                 length, and V indexes over activations for each
                                 symbol in the alphabet.
    """
    def combination_with_index(enc, pred, enc_len, pred_len, index):
        t = enc_len[index]
        u = pred_len[index]

        # [B,T,F] -> [T_i,F]
        enc_i = enc[index, :t, :]
        # [B,U+1,F] -> [U_i,F]
        pred_i = pred[index, :u, :]
        # [T_i,F] => [T_i, 1, F]
        enc_i = tf.expand_dims(enc_i, axis=1)
        # [U_i,F] => [1, U_i, F]
        pred_i = tf.expand_dims(pred_i, axis=0)
        # [T_i, 1, F] => [T_i, U_i, F]
        enc_i = tf.tile(enc_i, tf.stack([1, u, 1]))
        # [1, U_i, F] => [T_i, U_i, F]
        pred_i = tf.tile(pred_i, tf.stack([t, 1, 1]))

        # Plus and reshape
        comb_i = tf.nn.relu(enc_i+pred_i)

        # [T_i, U_i, F] => [T_i*U_i, F]
        return tf.reshape(comb_i, shape=[t*(u), x.enc_i.as_list()[-1]])

    # commbine
    batch_size, = shape_list(enc_len)
    i = tf.constant(1)
    combination = combination_with_index(enc, pred, enc_len, pred_len, 0)

    def body(i, combination):
        comb_i = combination_with_index(enc, pred, enc_len, pred_len, i)
        combination = tf.concat(
            [combination, comb_i], axis=0)  # [T_i*U_i, F] => [Sum(T_i*U_i), F]
        return i+1, combination

    _, combination = tf.while_loop(
        lambda i, a: tf.less(i, batch_size),
        body,
        [i, combination])

    # [Sum(T_i*U_i), F]
    return combination
