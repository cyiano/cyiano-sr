import os
import sys
import cv2
import pdb
import numpy as np
import random

from libs.utils import *
from libs import config_tf
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from libs.model import SR_model

FLAGS = tf.app.flags.FLAGS

def modcrop(imgs, modulo):
    if np.size(imgs.shape) == 4:
        (_, sheight, swidth, _) = imgs.shape
        sheight = sheight - np.mod(sheight, modulo)
        swidth = swidth - np.mod(swidth, modulo)
        imgs = imgs[:, 0:sheight, 0:swidth, :]
    elif np.size(imgs.shape) == 3:
        (sheight, swidth, _) = imgs.shape
        sheight = sheight - np.mod(sheight, modulo)
        swidth = swidth - np.mod(swidth, modulo)
        imgs = imgs[0:sheight, 0:swidth, :]
    else:
        (sheight, swidth) = imgs.shape
        sheight = sheight - np.mod(sheight, modulo)
        swidth = swidth - np.mod(swidth, modulo)
        imgs = imgs[0:sheight, 0:swidth]
    return imgs

def preproccessing(img):

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hr = modcrop(img, FLAGS.scale).astype('float32') / 255

    (hei, wid, _) = hr.shape
    lr_hei = int(hei/FLAGS.scale)
    lr_wid = int(wid/FLAGS.scale)
    
    # lr= cv2.resize(hr, (lr_wid, lr_hei), interpolation=cv2.INTER_CUBIC)
    lr = [cv2.resize(hr, (lr_wid, lr_hei), interpolation=cv2.INTER_CUBIC),
        cv2.resize(hr, (lr_wid, lr_hei), interpolation=cv2.INTER_LINEAR),
        cv2.resize(hr, (lr_wid, lr_hei), interpolation=cv2.INTER_NEAREST),
        cv2.resize(hr, (lr_wid, lr_hei), interpolation=cv2.INTER_AREA)][random.randint(0, 3)]

    hr = np.expand_dims(hr, 0)
    lr= np.expand_dims(lr, 0)

    return hr, lr

def export_img(img_name, img_list):
    img_concat = np.concatenate(img_list, axis=1)
    # img_concat = cv2.cvtColor(img_concat*255, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_name, img_concat*255)

def validate():

    PSNR_SR = []
    PSNR_BICUBIC = []
    # FLAGS.testdir = 'new data/1'
    filelist = os.listdir(FLAGS.testdir) 

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
        hr = cv2.imread(FLAGS.testdir + '/'+filelist[i])
        # hr = hr[50:98, 50:98, :]
        hr, lr = preproccessing(hr)
        (_, hei, wid, _) = hr.shape

        sr = model.predict(lr, need_restore=False)

        sr_bic = cv2.resize(lr[0], (wid, hei), interpolation=cv2.INTER_CUBIC)
        export_img('result/{}.jpg'.format(i), [sr_bic, sr[0], hr[0]])

        PSNR_SR.append(compute_psnr(sr[0], hr[0]))
        PSNR_BICUBIC.append(compute_psnr(sr_bic, hr[0]))

    sys.stdout.write('\n')
    sys.stdout.flush()

    print(np.mean(PSNR_SR))
    print(np.mean(PSNR_BICUBIC))

if __name__ == '__main__':
    validate()
