import numpy as np
import random
import math
import cv2
import sys
import pdb
import os

import tensorflow as tf
from libs import config_tf

FLAGS = tf.app.flags.FLAGS

class tfrecord_data(object):

    def __init__(self):
        self.HR_size = FLAGS.HR_size
        self.LR_size = int(self.HR_size / FLAGS.scale)
        self.stride = int(self.HR_size / 1)
        self.folder = FLAGS.image_dir + '/' + 'train'
        self.patch_generator()

    def _get_dataset_filename(self, dataset_dir, split_name, shard_id):
        output_filename = 'datasets_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id+1, FLAGS.num_shards)
        return os.path.join(dataset_dir, output_filename)

    def _to_tfexample(self, hr, img_id):
        return tf.train.Example(features=tf.train.Features(feature={
            'hr': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hr.tobytes()])),
            'img_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_id])),
        }))

    def _single_image_export(self, file_path, tfrecord_writer):

        hr_img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        (hei, wid, _) = hr_img.shape

        for x in range(0, hei - self.HR_size + 1, self.stride):
            for y in range(0, wid - self.HR_size + 1, self.stride):

                x_end = min(x + self.HR_size, hei)
                x_begin = x_end - self.HR_size
                y_end = min(y + self.HR_size, wid)
                y_begin = y_end - self.HR_size

                sub_hr = hr_img[x_begin:x_end, y_begin:y_end, :]

                example = self._to_tfexample(sub_hr, self.cnt)

                tfrecord_writer.write(example.SerializeToString())
                self.cnt += 1

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
                output_filename = self._get_dataset_filename('Images', split_name[k], shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(split_list))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(split_list), shard_id))
                        sys.stdout.flush()
                        self._single_image_export(self.folder+'/'+split_list[i], tfrecord_writer)

            sys.stdout.write('\n')
            sys.stdout.flush()

if __name__ == '__main__':
    traind = tfrecord_data()
    print('>> Done!')
