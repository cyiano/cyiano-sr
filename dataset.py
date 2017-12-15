import numpy as np
import random
import math
import cv2
import sys
import pdb
import os
import warnings

# import skimage
# from skimage import img_as_float, img_as_uint
# from skimage.io import imread
# from skimage.transform import rescale, resize

import tensorflow as tf
from libs import config_tf

FLAGS = tf.app.flags.FLAGS

def modcrop(imgs, modulo):
  if np.size(imgs.shape) == 3:
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

def img_rotate(img):
  rot = [0, 90, 180, 270]
  (hei, wei, chn) = img.shape
  img_rot = np.empty((4, hei, wei, chn))
  for k in range(4):
    M = cv2.getRotationMatrix2D((hei/2., wei/2.), rot[k], 1)
    img_rot[k] = cv2.warpAffine(img, M, (hei, wei))
  return img_rot

def random_blur(img, alpha=4, seed=None):
  f = random.randint(1, alpha)
  img = cv2.blur(img, (f, f))
  return img

def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'datasets_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id+1, FLAGS.num_shards)
  return os.path.join(dataset_dir, output_filename)

def _to_tfexample(hr, lr, img_id):
  return tf.train.Example(features=tf.train.Features(feature={
    'hr': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hr.tobytes()])),
    'lr': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lr.tobytes()])),
    'img_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_id])),
    }))

class tfrecord_data(object):

  def __init__(self):
    self.HR_size = FLAGS.HR_size
    self.LR_size = int(self.HR_size / FLAGS.scale)
    self.stride = int(self.HR_size / 1)
    self.folder = FLAGS.image_dir + '/' + 'train'
    self.patch_generator()

  def patch_generator(self):
    
    split_name = ['train', 'val']
    filelist = os.listdir(self.folder)
    random.shuffle(filelist)

    length = len(filelist)
    train_ratio = 0.9
    dataset_list = [filelist[:math.ceil(length * train_ratio)], filelist[math.ceil(length * train_ratio):]]

    for k in range(2):
      self.cnt = 1
      split_list = dataset_list[k]
      num_per_shard = int(math.ceil(len(split_list) / FLAGS.num_shards))

      for shard_id in range(FLAGS.num_shards):
        output_filename = _get_dataset_filename('Images', split_name[k], shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(split_list))

          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(split_list), shard_id))
            sys.stdout.flush()

            hr_img = cv2.imread(self.folder+'/'+split_list[i])
            (hei, wid, _) = hr_img.shape

            # sub_hr = cv2.resize(hr_img, (self.HR_size, self.HR_size), interpolation=cv2.INTER_CUBIC)
            
            # sub_lr = [cv2.resize(sub_hr, (self.LR_size, self.LR_size), interpolation=cv2.INTER_CUBIC),
            #           cv2.resize(sub_hr, (self.LR_size, self.LR_size), interpolation=cv2.INTER_LINEAR),
            #           cv2.resize(sub_hr, (self.LR_size, self.LR_size), interpolation=cv2.INTER_NEAREST),
            #           cv2.resize(sub_hr, (self.LR_size, self.LR_size), interpolation=cv2.INTER_AREA)][random.randint(0, 3)]
            # sub_lr = random_blur(sub_lr)

            # example = _to_tfexample(sub_hr, sub_lr, self.cnt)

            # tfrecord_writer.write(example.SerializeToString())
            # self.cnt += 1

            for x in range(0, hei - self.HR_size + 1, self.stride):
              for y in range(0, wid - self.HR_size + 1, self.stride):

                x_end = min(x + self.HR_size, hei-6)
                x_begin = x_end - self.HR_size
                y_end = min(y + self.HR_size, wid-6)
                y_begin = y_end - self.HR_size

                sub_hr = hr_img[x_begin:x_end, y_begin:y_end, :]
                
                sub_lr = [cv2.resize(sub_hr, (self.LR_size, self.LR_size), interpolation=cv2.INTER_CUBIC),
                          cv2.resize(sub_hr, (self.LR_size, self.LR_size), interpolation=cv2.INTER_LINEAR),
                          cv2.resize(sub_hr, (self.LR_size, self.LR_size), interpolation=cv2.INTER_NEAREST),
                          cv2.resize(sub_hr, (self.LR_size, self.LR_size), interpolation=cv2.INTER_AREA)][random.randint(0, 3)]
                sub_lr = random_blur(sub_lr)
                # cv2.imwrite('lr{}.jpg'.format(self.cnt), sub_lr)
                # cv2.imwrite('hr{}.jpg'.format(self.cnt), sub_hr)

                example = _to_tfexample(sub_hr, sub_lr, self.cnt)

                tfrecord_writer.write(example.SerializeToString())
                self.cnt += 1

      sys.stdout.write('\n')
      sys.stdout.flush()

      print('\nPatch size: {}\n'.format(self.cnt-1))

def convert():
  warnings.filterwarnings("ignore")
  traind = tfrecord_data()
  # traind.patch_generator()

if __name__ == '__main__':
  convert()  
