import os
import sys
import cv2
import pdb
import numpy as np

from libs.utils import *
from libs import config_tf
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

from libs.model import SR_model

FLAGS = tf.app.flags.FLAGS

def preproccessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hr = img.astype('float32') / 255
    return np.expand_dims(hr, 0)

def export_img(img_name, img_list):
    img_concat = np.concatenate(img_list, axis=1)
    img_concat = cv2.cvtColor(img_concat*255, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_name, img_concat)

def validate():

    PSNR_SR = []
    PSNR_BICUBIC = []
    filelist = os.listdir('Images/test') 

    # In fact variable lr and hr have no use in this module.
    lr = np.random.random((1, 16, 16, 3)).astype('float32')
    hr = np.random.random((1, 16, 16, 3)).astype('float32')
    model = SR_model([lr, hr], [lr, hr], scale=FLAGS.scale, num_blocks=FLAGS.num_blocks, drop_rate=1.0, is_training=False)
    model.sess.run(tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    ))
    model.restore('checkpoint/sr_model')


    for i in range(len(filelist)):
        sys.stdout.write('\r>> Inferencing the {}/{} th images......'.format(i+1, len(filelist)))
        sys.stdout.flush()
        img = cv2.imread('Images/test/'+filelist[i])
        img = preproccessing(img)
        (_, hei, wid, _) = img.shape

        sr = model.predict(img, need_restore=False)

        sr_bic = cv2.resize(img[0], (4*wid, 4*hei), interpolation=cv2.INTER_CUBIC)
        export_img('result/{}.jpg'.format(i), [sr_bic, sr[0]])

    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    validate()
