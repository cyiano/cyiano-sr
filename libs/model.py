from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import math
from libs.utils import bicubic_interpolation_2d
import tensorflow as tf
import tensorflow.contrib.slim as slim

def lrelu(x, trainbable=None):
    alpha = 0.2
    return tf.maximum(alpha * x, x)

def prelu(x, trainable=True):
    dim = x.get_shape()[-1]
    alpha = tf.get_variable('alpha', dim, dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=trainable)
    out = tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)
    return out

def PS(x, r):
    bs, a, b, c = x.get_shape().as_list()
    x = tf.reshape(x, (bs, a, b, r, r))  # bsize, a, b, 1, 1
    x = tf.transpose(x, (0, 1, 2, 4, 3))
    x = tf.split(x, a, axis=1)  # a, [bsize, b, r, r]
    x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)  # bsize, b, a*r, r
    x = tf.split(x, b, axis=1)  # b, [bsize, a*r, r]
    x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)  # bsize, a*r, b*r
    return tf.reshape(x, (bs, a*r, b*r, 1))

def pixel_shuffle_layer(x, r, n_split=None):
    xc = tf.split(x, n_split, axis=3)
    x = tf.concat([PS(x_, r) for x_ in xc], 3)
    return x

def residual_arg_scope( is_training=False,
                        need_bn=True, 
                        weight_decay=0.0001,
                        batch_norm_decay=0.95,
                        batch_norm_epsilon=1e-5,
                        batch_norm_scale=True,
                        normalizer_fn=None,
                        reuse=False):

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    if need_bn is True:
        normalizer_fn = slim.batch_norm

    with slim.arg_scope(
        [slim.conv2d],
        padding='SAME',
        weights_regularizer=None, #slim.l2_regularizer(weight_decay),
        weights_initializer=slim.xavier_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        reuse=reuse):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

def SR_model(x, scale, num_blocks, is_training=True, reuse=tf.AUTO_REUSE):

    # with slim.arg_scope([slim.conv2d],
    #                     trainable=is_training,
    #                     reuse=reuse):
    #     (_, hei, wid, _) = x.get_shape().as_list()
    #     x = slim.repeat(x, 6, slim.conv2d, 64, [3, 3], scope='conv0')
    #     x = slim.conv2d(x, 32, [3, 3], activation_fn=tf.nn.tanh, scope='conv1')
    #     x = slim.conv2d(x, 12, [3, 3], activation_fn=tf.nn.tanh, scope='conv2')
    #     x = pixel_shuffle_layer(x, 2, 3)
    #     x = slim.conv2d(x, 12, [3, 3], activation_fn=tf.nn.tanh, scope='conv3')
    #     x = pixel_shuffle_layer(x, 2, 3)
    #     return x

    x = tf.convert_to_tensor(x)

    if x.get_shape().ndims == 3:
        x = tf.expand_dims(x, 0)

    (_, hei, wid, _) = x.get_shape().as_list()

    with slim.arg_scope([slim.conv2d],
                        kernel_size=[3, 3],
                        activation_fn=tf.nn.relu,
                        reuse=reuse):
        with tf.variable_scope('Residual'):
            #x_bic = bicubic_interpolation_2d(x, (hei*scale, wid*scale), endpoint=True)
            x_bic = tf.image.resize_bicubic(x, (hei*scale, wid*scale))
            # innitial conv layer to extract feature map
            x = slim.conv2d(x, 64, scope='conv0')
            # skip = x

            # build residual block, you can choose whether to use batch normalization
            with slim.arg_scope(residual_arg_scope(is_training=is_training, need_bn=False, reuse=reuse)):
                for i in range(num_blocks):
                    f_channel = 64 + math.floor((64 * i) / num_blocks + 0.5)
                    mid = x
                    x = slim.conv2d(x, f_channel, scope='block{}_conv1'.format(i+1))
                    x = slim.conv2d(x, f_channel, scope='block{}_conv2'.format(i+1))
                    # x *= 0.1
                    x = tf.add(x, slim.conv2d(mid, f_channel, kernel_size=[1, 1], activation_fn=None, scope='block{}_skipconv'.format(i+1)))

            x = slim.conv2d(x, 64, scope='conv1')
            # x = tf.add(skip, x)

        with tf.variable_scope('Subpixel'):
            # pixel shuffle layers
            if scale == 2:
                x = slim.conv2d(x, 12, activation_fn=tf.nn.tanh, scope='ps2')
                x = pixel_shuffle_layer(x, 2, 3)

            if scale == 3:
                x = slim.conv2d(x, 27, activation_fn=tf.nn.tanh, scope='ps3')
                x = pixel_shuffle_layer(x, 3, 3)

            if scale == 4:
                x = slim.conv2d(x, 12, activation_fn=tf.nn.tanh, scope='ps4_1')
                x = pixel_shuffle_layer(x, 2, 3)

                x = slim.conv2d(x, 12, activation_fn=tf.nn.tanh, scope='ps4_2')
                x = pixel_shuffle_layer(x, 2, 3)

    x = tf.add(x, x_bic)
    return x