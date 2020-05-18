import sys
sys.path.append('../')
import os.path as osp
import global_macros
import tensorflow as tf
import numpy as np
from Unet2 import Unet2
from tensorflow.python.platform import flags
from utils import TF2FLRD, print_flag

FLAGS = flags.FLAGS

# Dataset Options:
flags.DEFINE_integer('batch_size', 32, 'Size of a batch')
#flags.DEFINE_bool('single', False, 'whether to debug by training on a single image')

# Base Model class Mandatory:
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_integer('epoch_num', 200, 'Number of Epochs to train on')
flags.DEFINE_integer('resume_iter', -1,
    'iteration to resume training from, -1 means not resuming')
flags.DEFINE_string('ckptdir', osp.join(global_macros.CKPT_ROOT, "Unet2"),
    'location where models will be stored')
flags.DEFINE_string('logdir', osp.join(global_macros.LOGGER_ROOT, "Unet2"),
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'exp', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 50,'save outputs every so many batches')
## Saver options:
flags.DEFINE_integer('max_to_keep', 30, 'maximum number of models to keep')
flags.DEFINE_integer('keep_checkpoint_every_n_hours', 3, 'check point intervals')

# Model specific:
flags.DEFINE_float('lr', 1e-3, 'Learning for training')
flags.DEFINE_integer('test_interval', 1000,'evaluate outputs every so many batches')
flags.DEFINE_bool('is_pad', True, 'Use padding for convolution or not')
flags.DEFINE_float('dropout_val', 0.6, 'Drop out value')

# Execution:
flags.DEFINE_integer('num_gpus', 1, 'number of gpus to train on')
flags.DEFINE_integer('data_workers', 4,
    'Number of different data workers to load data in parallel')


def parse(example):
    shapeX = (2,1,41,141)
    shapeY = (1,1,41,141)
    features = {
            'X': tf.FixedLenFeature(shapeX, tf.float32),
            'Y': tf.FixedLenFeature(shapeY, tf.float32)
        }
    data = tf.parse_single_example(example, features)

    datax = tf.reshape(data['X'][:,:,:40,:136], [2, 40, 136])
    datay = tf.reshape(data['Y'][:,:,:40,:136], [1, 40, 136])
    return datax, datay


def main():
    #print_flag(FLAGS)
    file_comment = "_sfc"
    years_train = [2014, 2015, 2016]
    years_val = [2017]
    years_test = [2018]

    ipaths_train = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + file_comment) for i in years_train]
    ipaths_val = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + file_comment) for i in years_val]
    ipaths_test = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + file_comment) for i in years_test]

    with tf.Session() as sess:
        with tf.name_scope('data'):
            iter_train = TF2FLRD(ipaths_train, batchsize=FLAGS.batch_size, buffersize=200, parse=parse)
            iter_val = TF2FLRD(ipaths_val, batchsize=40, buffersize=100, parse=parse)
            iter_test = TF2FLRD(ipaths_test, batchsize=40, buffersize=100, parse=parse)
        sess.run(iter_train.initializer)
        sess.run(iter_val.initializer)
        sess.run(iter_test.initializer)
        #with tf.device('/device:GPU:0'): #'/device:GPU:0'  '/cpu:0'
        model = Unet2(sess=sess, FLAGS=FLAGS)
        model.run(iter_data=iter_train, iter_val=iter_val)
        result = model.run(iter_data=iter_test, train=False, load=True)

if __name__ == "__main__":
    main()
