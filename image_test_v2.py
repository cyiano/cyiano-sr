from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import sys
import numpy as np
from time import strftime, gmtime

import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs import config_tf
from libs.vgg import get_vgg_fn
from libs.model import SR_model
from libs.input_loader import get_dataset_batches

FLAGS = tf.app.flags.FLAGS

def get_psnr(predictions, labels, bit):
    def log10(x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    predictions = tf.cast(predictions, tf.float32)
    labels = tf.cast(labels, tf.float32)
    mse = tf.reduce_mean(tf.square(predictions - labels), axis=[1, 2])
    psnr = tf.reduce_mean(10 * log10((2**bit-1) * (2**bit-1) / mse))
    return psnr

#############################
#     The main function     #
#############################
def val_record_inference():

    hr, lr, img_id = get_dataset_batches('Images', 'val')
    sr = SR_model(lr, FLAGS.scale, FLAGS.num_blocks, is_training=False)
    hr_bic = tf.image.resize_bicubic(lr, [FLAGS.HR_size, FLAGS.HR_size])
    psnr = get_psnr(tf.cast(sr*255, tf.uint8), tf.cast(hr*255, tf.uint8), 8)
    psnr_bic = get_psnr(tf.cast(hr_bic*255, tf.uint8), tf.cast(hr*255, tf.uint8), 8)

    sess = tf.Session()
    sess.run(tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    ))

    '''
    Create the coord and threads which is neccessary in queue
    operations.
    '''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    '''
    Restore the model.
    '''
    sr_restorer = tf.train.Saver()
    sr_restorer.restore(sess, 'checkpoint/sr_model.ckpt')

    '''
    Starting the inference using queue.
    '''
    metric_list = []
    cnt = 1
    print('\n\tTraining begin....\n')

    try:
        while True:
            sys.stdout.write('\r>> Inferencing the {} th images......'.format(cnt))
            sys.stdout.flush()
            psnr_np, psnr_bic_np, lr_np, sr_np, hr_bic_np, hr_np = \
                sess.run([psnr, psnr_bic, lr, sr, hr_bic, hr])
            metric_list.append([psnr_np, psnr_bic_np])
            cnt += 1
    except tf.errors.OutOfRangeError:
        coord.request_stop()
    finally:
        coord.request_stop()
        coord.join(threads)

    sys.stdout.write('\n\n')
    sys.stdout.flush()

    metric_list = np.mean(np.asarray(metric_list), axis=0)
    print('Validation ----- PSNR: {0:.8f}, PSNR using bicubic: {1:.8f}\n'.format(metric_list[0], metric_list[1]))

if __name__ == '__main__':
    val_record_inference()
