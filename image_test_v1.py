from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import pdb
import math
import numpy as np
import random

from libs.utils import *
from libs import config_tf
import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.model import SR_model

FLAGS = tf.app.flags.FLAGS

def image_preprocessing(img):

    (height, width, channel) = img.shape
    img = tf.expand_dims(tf.convert_to_tensor(img), axis=0)

    hr_hei = FLAGS.scale * math.floor(height / FLAGS.scale)
    hr_wid = FLAGS.scale * math.floor(width / FLAGS.scale)
    lr_hei, lr_wid = int(hr_hei/FLAGS.scale), int(hr_wid/FLAGS.scale)

    hr = tf.image.resize_image_with_crop_or_pad(img, hr_hei, hr_wid)

    lr = tf.image.resize_bicubic(hr, (lr_hei, lr_wid))
    hr_bic = tf.image.resize_bicubic(lr, (hr_hei, hr_wid))

    hr = tf.cast(hr, tf.float32) / 255.
    hr_bic = tf.cast(hr_bic, tf.float32) / 255.
    lr = tf.cast(lr, tf.float32) / 255.

    return hr, hr_bic, lr

def image_preprocessing_without_downsampling(img):

    (height, width, channel) = img.shape
    lr = tf.expand_dims(tf.convert_to_tensor(img), axis=0)
    hr_bic = tf.image.resize_bicubic(lr, (height*FLAGS.scale, width*FLAGS.scale))

    hr_bic = tf.cast(hr_bic, tf.float32) / 255.
    lr = tf.cast(lr, tf.float32) / 255.

    return lr, hr_bic

def export_img_series(img_name, img_list):
    img_concat = np.concatenate(img_list, axis=1)
    img_concat = cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_name, img_concat)

def validate(mode):

    PSNR_SR, PSNR_BICUBIC = [], []
    filelist = [FLAGS.testdir + '/'+ name for name in os.listdir(FLAGS.testdir)]

    fake = np.random.random((1, 1, 1, 3)).astype('float32')
    fake_s = SR_model(fake, FLAGS.scale, FLAGS.num_blocks, is_training=False)

    sess = tf.Session()
    sess.run(tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    ))

    sr_restorer = tf.train.Saver()
    sr_restorer.restore(sess, 'checkpoint/sr_model.ckpt')

    if mode == 0:
        for i in range(len(filelist)):

            sys.stdout.write('\r>> Inferencing the {}/{} th images......'.format(i+1, len(filelist)))
            sys.stdout.flush()

            image_np = cv2.cvtColor(cv2.imread(filelist[i]), cv2.COLOR_BGR2RGB)
            hr_tf, hr_bic_tf, lr_tf = image_preprocessing(image_np)
            sr_tf = SR_model(lr_tf, FLAGS.scale, FLAGS.num_blocks, is_training=False)
            lr, sr, hr_bic, hr = sess.run([lr_tf, sr_tf, hr_bic_tf, hr_tf])

            sr = (sr*255).astype('uint8')
            hr_bic = (hr_bic*255).astype('uint8')
            hr = (hr*255).astype('uint8')

            export_img_series('result/{}.jpg'.format(i), [hr_bic[0], sr[0], hr[0]])
            PSNR_SR.append(compute_psnr(sr[0], hr[0], 8))
            PSNR_BICUBIC.append(compute_psnr(hr_bic[0], hr[0], 8))

        sys.stdout.write('\n')
        sys.stdout.flush()

        print(np.mean(PSNR_SR))
        print(np.mean(PSNR_BICUBIC))

    else:
        for i in range(len(filelist)):

            sys.stdout.write('\r>> Inferencing the {}/{} th images......'.format(i+1, len(filelist)))
            sys.stdout.flush()

            image_np = cv2.cvtColor(cv2.imread(filelist[i]), cv2.COLOR_BGR2RGB)
            lr_tf, hr_bic_tf = image_preprocessing(image_np)
            sr_tf = SR_model(lr_tf, FLAGS.scale, FLAGS.num_blocks, is_training=False)
            lr, sr, hr_bic = sess.run([lr_tf, sr_tf, hr_bic_tf])

            sr = (sr*255).astype('uint8')
            hr_bic = (hr_bic*255).astype('uint8')
            export_img_series('result/{}.jpg'.format(i), [hr_bic[0], sr[0]])

            sys.stdout.write('\n')
            sys.stdout.flush()

if __name__ == '__main__':
    validate(mode=0)
