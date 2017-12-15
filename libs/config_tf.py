import tensorflow as tf

#######################
#    Dataset Flags    #
#######################

tf.app.flags.DEFINE_integer(
    'HR_size', 112, 'Train image size')

tf.app.flags.DEFINE_integer(
    'scale', 4, 'Train image size')

tf.app.flags.DEFINE_string(
    'image_dir', 'Images',
    'The directory where the jpg-format dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'num_shards', 10,
    'The number of threads used to create the batches.')

#######################
#     Model Flags     #
#######################

tf.app.flags.DEFINE_integer(
    'batch_size', 32,
    'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_blocks', 8, 'Train image size')

#######################
#     Train Flags     #
#######################

tf.app.flags.DEFINE_float(
    'lr_rate', 1e-4, 'Learning rate.')

tf.app.flags.DEFINE_string(
    'testdir', 'Images/test',
    'The directory where the jpg-format dataset files are stored.')



FLAGS = tf.app.flags.FLAGS