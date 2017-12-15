from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import pdb
import glob
import numpy as np

from libs.utils import *
from libs import config_tf
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

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
    tfrecords_filename, num_epochs=1)

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
  
def get_dataset_batches(dataset_dir, batch_size=None, file_pattern=None):

  if file_pattern is None:
    file_pattern = 'datasets_val' + '*.tfrecord'
  tfrecords = glob.glob(dataset_dir + '/' + file_pattern)

  hr_sets, lr_sets, id_sets = _tfrecords_reader(tfrecords)
  hr, lr, img_id = tf.train.shuffle_batch([hr_sets, lr_sets, id_sets], 
                                          batch_size=batch_size,
                                          num_threads=2,
                                          capacity=20000,
                                          min_after_dequeue=1000)

  return hr, lr, img_id

def validate():

    # Loading the data
    hr, lr, img_id = get_dataset_batches('Images', 1)
    PSNR_SR = []
    PSNR_BICUBIC = []

    # build the SR model
    model = SR_model([lr, hr], [lr, hr], scale=FLAGS.scale, num_blocks=FLAGS.num_blocks, drop_rate=1.0, is_training=False)

    gt_shape = hr.get_shape().as_list()
    
    model.sess.run(tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    ))

    coord = tf.train.Coordinator()
    threads = []
    tf.train.start_queue_runners(sess=model.sess, coord=coord)

    model.restore('checkpoint/sr_model')
    cnt = 1

    try:
        while True:
            sys.stdout.write('\r>> Inferencing the {} th images...'.format(cnt))
            sys.stdout.flush()
            lr_np, sr_np, hr_np = model.sess.run([model.x, model.y, model.gt])
            lr_np, sr_np, hr_np = lr_np[0], sr_np[0], hr_np[0]
            srbic_np = cv2.resize(lr_np, (112, 112), interpolation=cv2.INTER_CUBIC)
            # LR = cv2.resize(LR[0], (96,96))
            # LR = np.expand_dims(LR, axis=2)
            # img_concat = np.concatenate([LR, HR[0], SR[0]], axis=1)
            # cv2.imwrite('result/{}.jpg'.format(cnt), img_concat*255)
            PSNR_SR.append(compute_psnr(sr_np, hr_np))
            PSNR_BICUBIC.append(compute_psnr(srbic_np, hr_np))
            cnt += 1
            if cnt >= 1000:
                break

    except tf.errors.OutOfRangeError:
        coord.request_stop()
    finally:
        coord.request_stop()
        coord.join(threads)

    sys.stdout.write('\n')
    sys.stdout.flush()

    print(np.mean(PSNR_SR))
    print(np.mean(PSNR_BICUBIC))

if __name__ == '__main__':
    validate()