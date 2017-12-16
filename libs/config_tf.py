from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#######################
#    Dataset Flags    #
#######################

tf.app.flags.DEFINE_integer(
    'HR_size', 112, 'Train image size')

tf.app.flags.DEFINE_integer(
    'scale', 4, 'Train image size'
    )

tf.app.flags.DEFINE_string(
    'image_dir', 'Images',
    'The directory where the jpg-format dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'num_shards', 10,
    'The number of threads used to create the batches.')

#######################
#     Model Flags     #
#######################

tf.app.flags.DEFINE_integer(
    'batch_size', 8,
    'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_blocks', 4, 'Train image size.')

tf.app.flags.DEFINE_bool(
    'use_perceptual_loss', False,
    'Whether to use perceptual loss.')

#######################
#     Train Flags     #
#######################

tf.app.flags.DEFINE_string(
    'testdir', "Images/test",
    'The directory where the jpg-format dataset files are stored.')

tf.app.flags.DEFINE_bool(
    'ckpt', True,
    'Where the training is begun without loading checkpoint.')


FLAGS = tf.app.flags.FLAGS