from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
from libs import config_tf
import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs.model import SR_model

FLAGS = tf.app.flags.FLAGS

def _preprocess_for_image(image, hei, wid, chn):
    hr = tf.reshape(image, (1, hei, wid, chn))
    lr_hei, lr_wid = int(hei/FLAGS.scale), int(wid/FLAGS.scale)
    lr = tf.image.resize_bicubic(hr, (lr_hei, lr_wid))

    hr = tf.cast(tf.squeeze(hr, axis=0), tf.float32) / 255.
    lr = tf.cast(tf.squeeze(lr, axis=0), tf.float32) / 255.
    return hr, lr

def _tfrecords_reader(tfrecords_filename, num_epochs):

    if not isinstance(tfrecords_filename, list):
        tfrecords_filename = [tfrecords_filename]

    filename_queue = tf.train.string_input_producer(
        tfrecords_filename, num_epochs=num_epochs)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={'hr': tf.FixedLenFeature([], tf.string),
                  'img_id': tf.FixedLenFeature([], tf.int64),
    })

    LR_size = int(FLAGS.HR_size / FLAGS.scale)

    hr = tf.decode_raw(features['hr'], tf.uint8)
    hr, lr = _preprocess_for_image(hr, FLAGS.HR_size, FLAGS.HR_size, 3)

    img_id = tf.cast(features['img_id'], tf.int32)

    return hr, lr, img_id

def get_dataset_batches(dataset_dir, split_name, batch_size=None, file_pattern=None):

    assert split_name in ['train', 'val']

    if file_pattern is None:
        file_pattern = 'datasets_' + split_name + '*.tfrecord'
    tfrecords = glob.glob(dataset_dir + '/' + file_pattern)

    if split_name == 'train':
        hr_sets, lr_sets, id_sets = _tfrecords_reader(tfrecords, 10000)
        hr, lr, img_id = tf.train.shuffle_batch([hr_sets, lr_sets, id_sets],
                                                batch_size=batch_size,
                                                num_threads=2,
                                                capacity=20000,
                                                min_after_dequeue=1000)
    else:
        hr_sets, lr_sets, id_sets = _tfrecords_reader(tfrecords, 1)
        [hr, lr, img_id] = [tf.expand_dims(x, axis=0) for x in [hr_sets, lr_sets, id_sets]]

    return hr, lr, img_id
