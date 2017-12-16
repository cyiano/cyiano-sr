from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np
from time import strftime, gmtime

import tensorflow as tf
import tensorflow.contrib.slim as slim

from libs import config_tf
from libs.vgg import get_vgg_fn
from libs.model import SR_model
from libs.input_loader import get_dataset_batches

FLAGS = tf.app.flags.FLAGS

##############################################
#     Define the loss between two images     #
##############################################
def get_l1_loss(predictions, labels):
    l1_loss = tf.losses.absolute_difference(labels=labels, predictions=predictions)
    return l1_loss

def get_l2_loss(predictions, labels):
    l2_loss = tf.losses.mean_squared_error(predictions=predictions, labels=labels)
    return l2_loss

def get_psnr(predictions, labels):
    def log10(x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    mse = tf.reduce_mean(tf.square(predictions - labels), axis=[1, 2])
    psnr = tf.reduce_mean(-10 * log10(mse))
    return psnr

def get_perceptual_loss(predictions, labels):
    vgg_pred1, vgg_pred2 = get_vgg_fn(predictions)
    vgg_labels1, vgg_labels2 = get_vgg_fn(labels)
    p_loss1 = tf.losses.mean_squared_error(predictions=vgg_pred1, labels=vgg_labels1)
    p_loss2 = tf.losses.mean_squared_error(predictions=vgg_pred2, labels=vgg_labels2)
    return p_loss1 + p_loss2

def get_loss(predictions, labels, weights):

    loss_mse = get_l2_loss(predictions, labels)
    tf.summary.scalar('mse_loss', loss_mse)
    
    if FLAGS.use_perceptual_loss is True:
        loss_p = get_perceptual_loss(predictions, labels)  
        loss_total = loss_mse + weights * loss_p
        tf.summary.scalar('perceptual_loss', loss_p)
        tf.summary.scalar('total_loss', loss_total)
        return loss_total
    else:
        return loss_mse

#############################
#     The main function     #
#############################
def train(train_conf):

    lr_rate = tf.placeholder(dtype=tf.float32, shape=[], name='lr_rate')
    hr, lr, hr_bic, img_id = get_dataset_batches('Images', 'train', FLAGS.batch_size)
    hr_val, lr_val, hr_bic_val, img_id_val = get_dataset_batches('Images', 'val', FLAGS.batch_size)

    sr = SR_model(lr, hr_bic, FLAGS.scale, FLAGS.num_blocks, is_training=True)

    loss = get_loss(sr, hr, weights=100)
    psnr = get_psnr(sr, hr)

    vars_all = tf.trainable_variables()
    vars_sr = [v for v in vars_all if 'Residual' in v.name]
    vars_subpixel = [v for v in vars_all if 'Subpixel' in v.name]
    if FLAGS.use_perceptual_loss is True:
        vars_vgg = [v for v in vars_all if 'vgg_19' in v.name]

    for var in vars_all:
        print(var)

    '''
    Set the global step, and learning methods.
    '''
    global_step = tf.Variable(0, name='global_step', trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=lr_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    train_op = slim.learning.create_train_op(loss, opt, global_step=global_step)

    '''
    Initalize all the tf variables. Note that 'tf.local_varia-
    bles_initializer()' is needed if you use parameter 'num_e-
    poch' in 'tf.train.string_input_producer'.
    '''
    sess = tf.Session()
    sess.run(tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    ))

    '''
    Gather all the tf.summary operations.
    '''
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('checkpoint', graph=sess.graph)

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
    if FLAGS.ckpt is True:
        sr_restorer.restore(sess, 'checkpoint/sr_model.ckpt')
    if FLAGS.use_perceptual_loss is True:
        vgg_restorer = tf.train.Saver(var_list = vars_vgg)
        vgg_restorer.restore(sess, 'checkpoint/vgg_19.ckpt')

    '''
    Starting the training iteration, along with the model sa-
    ving and loss values printing.
    '''
    metric_list = []
    print('\n\tTraining begin....\n')

    for cnt in range(len(train_conf['epoch'])):
        for epoch in range(train_conf['epoch'][cnt]):

            _, summary_str, loss_np, psnr_np, step, lr_np, sr_np, hr_np = \
                sess.run([train_op, summary_op, loss, psnr, global_step, lr, sr, hr], 
                        feed_dict={lr_rate: train_conf['lr_rate'][cnt]})  
            summary_writer.add_summary(summary_str, step)
            metric_list.append([loss_np, psnr_np])
            # Print the average MSE and PSNR.
            if step % 100 == 0:
                metric_list = np.asarray(metric_list)
                metric_list = np.mean(metric_list, axis=0)
                print('{0} ----- epoch: {1}, Total loss: {2:.8f}, PSNR: {3:.8f}'.format(
                    strftime('%Y-%m-%d %H:%M:%S'), step, metric_list[0], metric_list[1]))
                metric_list = []

            # Save the model.
            if step % 1000 == 0:
                sr_val = SR_model(lr_val, hr_bic_val, FLAGS.scale, FLAGS.num_blocks, is_training=False)
                # sr_bic = tf.image.resize_bicubic(lr_val, [FLAGS.HR_size, FLAGS.HR_size])
                psnr_val = get_psnr(sr_val, hr_val)
                psnr_bic = get_psnr(hr_bic_val, hr_val)
                loss1, loss2 = sess.run([psnr_val, psnr_bic])
                print('Validation ----- PSNR: {0:.8f}, PSNR using bicubic: {1:.8f}\n'.format(loss1, loss2))

            if step % 5000 == 0:
                # self.save(step, strftime('ckpt-%Y%m%d-%H%M%S-')+'-Step-'+str(step))
                print('\n  ################################\n      Saving the model ......\n  ################################\n')
                # save_name = os.path.join('checkpoint', root_dir, 'sr_model.ckpt')
                sr_restorer.save(sess, 'checkpoint/sr_model.ckpt')

            # End up the coord.
            if coord.should_stop():
                coord.request_stop()
                coord.join(threads)

if __name__ == '__main__':
    train_conf = {
        'epoch': [10000, 10000, 10000],
        'lr_rate': [1e-3, 2*1e-4, 1e-4],
    }
    train(train_conf)