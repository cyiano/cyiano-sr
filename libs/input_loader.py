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
    image = tf.reshape(image, (hei, wid, chn))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def _tfrecords_reader(tfrecords_filename):

    if not isinstance(tfrecords_filename, list):
        tfrecords_filename = [tfrecords_filename]

    filename_queue = tf.train.string_input_producer(
        tfrecords_filename, num_epochs=10000)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={
            'hr': tf.FixedLenFeature([], tf.string),
            'lr': tf.FixedLenFeature([], tf.string),
            'img_id': tf.FixedLenFeature([], tf.int64),
    })

    LR_size = int(FLAGS.HR_size / FLAGS.scale)

    hr = tf.decode_raw(features['hr'], tf.uint8)
    hr = _preprocess_for_image(hr, FLAGS.HR_size, FLAGS.HR_size, 3)
    lr = tf.decode_raw(features['lr'], tf.uint8)
    lr = _preprocess_for_image(lr, LR_size, LR_size, 3)

    img_id = tf.cast(features['img_id'], tf.int32)

    return hr, lr, img_id
  
def get_dataset_batches(dataset_dir, split_name, batch_size=None, file_pattern=None):

    assert split_name in ['train', 'val']

    if file_pattern is None:
        file_pattern = 'datasets_' + split_name + '*.tfrecord'
    tfrecords = glob.glob(dataset_dir + '/' + file_pattern)

    hr_sets, lr_sets, id_sets = _tfrecords_reader(tfrecords)
    hr, lr, img_id = tf.train.shuffle_batch([hr_sets, lr_sets, id_sets], 
                                            batch_size=batch_size,
                                            num_threads=2,
                                            capacity=20000,
                                            min_after_dequeue=1000)

    return hr, lr, img_id
