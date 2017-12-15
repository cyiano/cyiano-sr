import os
import cv2
import pdb
import math
import numpy as np
from time import strftime, gmtime
import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs import config_tf
from tensorflow.python.ops import control_flow_ops
from libs.utils import *
from libs.vgg import get_vgg_fn
FLAGS = tf.app.flags.FLAGS

def lrelu(x, trainbable=None):
    alpha = 0.2
    return tf.maximum(alpha * x, x)

def prelu(x, trainable=True):
    dim = x.get_shape()[-1]
    alpha = tf.get_variable('alpha', dim, dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=trainable)
    out = tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)
    return out

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PS(x, r):
    bs, a, b, c = x.get_shape().as_list()
    x = tf.reshape(x, (bs, a, b, r, r))  # bsize, a, b, 1, 1
    x = tf.transpose(x, (0, 1, 2, 4, 3))
    x = tf.split(x, a, axis=1)  # a, [bsize, b, r, r]
    x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)  # bsize, b, a*r, r
    x = tf.split(x, b, axis=1)  # b, [bsize, a*r, r]
    x = tf.concat([tf.squeeze(x_, axis=1) for x_ in x], 2)  # bsize, a*r, b*r
    return tf.reshape(x, (bs, a*r, b*r, 1))

def pixel_shuffle_layer(x, r, n_split=None):
    xc = tf.split(x, n_split, axis=3)
    x = tf.concat([PS(x_, r) for x_ in xc], 3)
    return x

def residual_arg_scope( is_training=False,
                        need_bn=True, 
                        weight_decay=0.0001,
                        batch_norm_decay=0.95,
                        batch_norm_epsilon=1e-5,
                        batch_norm_scale=True,
                        normalizer_fn=None,
                        reuse=False):

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    if need_bn is True:
        normalizer_fn = slim.batch_norm

    with slim.arg_scope(
        [slim.conv2d],
        padding='SAME',
        weights_regularizer=None, #slim.l2_regularizer(weight_decay),
        weights_initializer=slim.xavier_initializer(),
        activation_fn=None,
        normalizer_fn=normalizer_fn,
        reuse=reuse):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

class SR_model:

    def __init__(self, training_set, validation_set, scale, num_blocks, drop_rate, is_training):

        self.scale = scale
        self.is_training = is_training
        self.num_blocks = num_blocks
        # self.x = x
        # self.gt = gt
        [self.x, self.gt] = training_set
        [self.x_val, self.gt_val] = validation_set
        self.drop_rate = drop_rate
        self.y = self.forward(self.x, self.num_blocks)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.sess = tf.Session()

        self.saver = tf.train.Saver()
        self.restorer = tf.train.Saver()

    def forward(self, x, num_blocks, drop_rate=None, reuse=False):

        # with slim.arg_scope([slim.conv2d],
        #                     trainable=self.is_training,
        #                     reuse=reuse):
        #     (_, hei, wid, _) = x.get_shape().as_list()
        #     skip = tf.image.resize_bicubic(x, (hei*FLAGS.scale, wid*FLAGS.scale))
        #     x = slim.repeat(x, 6, slim.conv2d, 64, [3, 3], scope='conv0')
        #     x = slim.conv2d(x, 32, [3, 3], activation_fn=tf.nn.tanh, scope='conv1')
        #     x = slim.conv2d(x, 12, [3, 3], activation_fn=tf.nn.tanh, scope='conv2')
        #     x = pixel_shuffle_layer(x, 2, 3)
        #     x = slim.conv2d(x, 12, [3, 3], activation_fn=tf.nn.tanh, scope='conv3')
        #     x = pixel_shuffle_layer(x, 2, 3)
        #     return x + skip

        if drop_rate is None:
            drop_rate = self.drop_rate

        x = tf.convert_to_tensor(x)
        (_, hei, wid, _) = x.get_shape().as_list()
        start = tf.image.resize_bicubic(x, (hei*FLAGS.scale, wid*FLAGS.scale))

        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=None,
                            reuse=reuse):
            with tf.variable_scope('SR', reuse=reuse):
                with tf.variable_scope('Residual'):
                    # innitial conv layer to extract feature map
                    x = slim.conv2d(x, 64, [3, 3], activation_fn=tf.nn.relu, scope='conv0')
                    # skip = x

                    # build residual block, you can choose whether to use batch normalization
                    with slim.arg_scope(residual_arg_scope(is_training=self.is_training, need_bn=False)):
                        for i in range(num_blocks):
                            mid = x
                            x = slim.conv2d(x, 64, [3, 3], scope='block{}_conv1'.format(i+1))
                            x = tf.nn.relu(x)
                            x = slim.conv2d(x, 64, [3, 3], scope='block{}_conv2'.format(i+1))
                            x *= 0.1
                            x = tf.add(mid, x)

                    x = slim.conv2d(x, 64, [3, 3], scope='conv1')
                    # x = tf.add(skip, x)

                with tf.variable_scope('Subpixel'):
                    # pixel shuffle layers
                    if self.scale == 2:
                        x = slim.conv2d(x, 12, [3, 3], scope='ps2')
                        x = tf.nn.tanh(x)
                        x = pixel_shuffle_layer(x, 2, 3)

                    if self.scale == 3:
                        x = slim.conv2d(x, 27, [3, 3], scope='ps3')
                        x = tf.nn.tanh(x)
                        x = pixel_shuffle_layer(x, 3, 3)

                    if self.scale == 4:
                        x = slim.conv2d(x, 12, [3, 3], scope='ps4_1')
                        x = tf.nn.tanh(x)
                        x = pixel_shuffle_layer(x, 2, 3)

                        x = slim.conv2d(x, 12, [3, 3], scope='ps4_2')
                        x = tf.nn.tanh(x)
                        x = pixel_shuffle_layer(x, 2, 3)
        x = tf.add(x, start)
        return x

    def get_l1_loss(self, predictions, labels):
        l1_loss = tf.losses.absolute_difference(labels=labels, predictions=predictions)
        return l1_loss
    
    def get_l2_loss(self, predictions, labels):
        l2_loss = tf.losses.mean_squared_error(predictions=predictions, labels=labels)
        return l2_loss

    def get_perceptual_loss(self, predictions, labels):
        vgg_pred1, vgg_pred2 = get_vgg_fn(predictions)
        vgg_labels1, vgg_labels2 = get_vgg_fn(labels)
        p_loss1 = tf.losses.mean_squared_error(predictions=vgg_pred1, labels=vgg_labels1)
        p_loss2 = tf.losses.mean_squared_error(predictions=vgg_pred2, labels=vgg_labels2)
        return p_loss1 + p_loss2

    def get_psnr(self, predictions, labels):
        mse = tf.reduce_mean(tf.square(predictions - labels), axis=(1, 2))
        psnr = tf.reduce_mean(-10 * log10(mse))
        return psnr

    def save(self, step, root_dir):
        print('\n  ################################\n      Saving the model ......\n  ################################\n')
        save_name = os.path.join('checkpoint', root_dir, 'sr_model')
        self.saver.save(self.sess, save_name)

    def restore(self, checkpoint_path=None):
        print('Restoring the model.....')
        if checkpoint_path is None:
            checkpoint_path = tf.train.latest_checkpoint('checkpoint')
        self.restorer.restore(self.sess, checkpoint_path)
        print('Successfully restored the model!')

    def predict(self, x, need_restore=False):
        '''
        Use the trained model to predict super-resolution image, 
        note that the parameter 'x' should be type 'numpy.
        ndarray'.
        '''
        if need_restore is True:
            self.restore()

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        if x.get_shape().ndims == 3:
            x = tf.expand_dims(x, 0)
        elif x.get_shape().ndims == 4:
            x = x
        else:
            raise 'Dimension of the x not correct!'

        y = self.forward(x, self.num_blocks, reuse=True)
        return self.sess.run(y)

    def train(self, train_mode, num_epoch, lr_rate, need_restore=True):
        '''
        x: the input of SR model.
        gt: the ground truth corresponding to x.
        we use the standard mse loss as the optimization target.
        '''
        assert train_mode in ['residual', 'ps', 'all']

        self.loss_mse = self.get_l2_loss(self.y, self.gt)
        self.loss_psnr = self.get_psnr(self.y, self.gt)
        self.loss_p = self.get_perceptual_loss(self.y, self.gt)
        self.loss_total = self.loss_mse + 1000*self.loss_p

        vars_all = tf.trainable_variables()
        vars_sr = [v for v in vars_all if 'Residual' in v.name]
        vars_subpixel = [v for v in vars_all if 'Subpixel' in v.name]
        vars_vgg = [v for v in vars_all if 'vgg_19' in v.name]

        for var in vars_all:
            print(var)

        if train_mode is 'residual':
            var_list = vars_sr
        elif train_mode is 'ps':
            var_list = vars_subpixel
        elif train_mode is 'all':
            var_list = vars_all

        tf.summary.scalar('mse_loss', self.loss_mse)
        tf.summary.scalar('perceptual_loss', self.loss_p)
        tf.summary.scalar('total_loss', self.loss_total)

        '''
        Set the global step, and learning methods.
        '''
        update_bns = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        updates = tf.group(*update_bns)
        self.loss_total = control_flow_ops.with_dependencies([updates], self.loss_total)
        opt = tf.train.AdamOptimizer(learning_rate=lr_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        train_op = slim.learning.create_train_op(self.loss_total, opt, global_step=self.global_step)

        '''
        Initalize all the tf variables. Note that 'tf.local_varia-
        bles_initializer()' is needed if you use parameter 'num_e-
        poch' in 'tf.train.string_input_producer'.
        '''
        self.sess.run(tf.group(
            tf.local_variables_initializer(),
            tf.global_variables_initializer()
        ))

        '''
        Gather all the tf.summary operations.
        '''
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('checkpoint', graph=self.sess.graph)

        '''
        Create the coord and threads which is neccessary in queue 
        operations.
        '''
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        '''
        Restore the model.
        '''
        vgg_restorer = tf.train.Saver(var_list = vars_vgg)
        vgg_restorer.restore(self.sess, 'checkpoint/vgg_19.ckpt')

        if need_restore is True:
            self.restore('checkpoint/sr_model')

        '''
        Starting the training iteration, along with the model sa-
        ving and loss values printing.
        '''
        metric_list = []
        print('\n\tTraining begin....\n')

        for epoch in range(num_epoch):

            _, summary_str, loss_mse, loss_p, loss_total, step, x_np, y_np, gt_np = \
                self.sess.run([train_op, summary_op, 
                              self.loss_mse, self.loss_p, self.loss_total,
                              self.global_step, self.x, self.y, self.gt])  
            # x_np = cv2.cvtColor((x_np[0]*255).astype('uint8'), cv2.COLOR_BGR2RGB)
            # cv2.imwrite('h1.jpg', x_np) 
            # y_np = cv2.cvtColor((y_np[0]*255).astype('uint8'), cv2.COLOR_BGR2RGB)
            # cv2.imwrite('h2.jpg', y_np)
            # gt_np = cv2.cvtColor((gt_np[0]*255).astype('uint8'), cv2.COLOR_BGR2RGB)
            # cv2.imwrite('h3.jpg', gt_np)
            # pdb.set_trace()
            summary_writer.add_summary(summary_str, step)
            metric_list.append([loss_mse, loss_p, loss_total, compute_psnr(y_np, gt_np)])

            # Print the average MSE and PSNR.
            if step % 100 == 0:
                metric_list = np.asarray(metric_list)
                metric_list = np.mean(metric_list, axis=0)
                print('{0} ----- epoch: {1}, MSE loss: {2:.8f}, Peceptual loss: {3:.8f}, Total loss: {4:.8f}, PSNR: {5:.8f}'.format(
                    strftime('%Y-%m-%d %H:%M:%S'), step, metric_list[0], metric_list[1], metric_list[2], metric_list[3]))
                metric_list = []
            
            # Save the model.
            if step % 1000 == 0:
                self.y_val = self.forward(self.x_val, self.num_blocks, reuse=True)
                yval_np, gtval_np = self.sess.run([self.y_val, self.gt_val])
                loss_psnr_val = compute_psnr(yval_np, gtval_np)
                print('Validation ----- PSNR: {0:.8f}'.format(loss_psnr_val))
                
            if step % 5000 == 0:
                self.save(step, strftime('ckpt-%Y%m%d-%H%M%S-')+'-Step-'+str(step))
                
            # End up the coord.
            if self.coord.should_stop():
                self.coord.request_stop()
                self.coord.join(self.threads)
