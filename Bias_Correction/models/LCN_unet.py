#!/usr/bin/python
import tensorflow as tf
import random
import numpy as np
import os.path as osp
from models.utils import variable_summaries, base_model, getLoss, getEvaluate
import config

from models.layers import upconv3d_layer, conv_batch_relu3d_layer, centre_crop_and_concat_layer, LCN3D_layer

# define model parameter as macros here:
X_SHAPE = config.X_SHAPE
Y_SHAPE = config.Y_SHAPE

base_n_filt = 4
base_init = tf.truncated_normal_initializer(stddev=0.1)
reg_init = tf.contrib.layers.l2_regularizer(scale=0.1)

class LCN_unet(base_model):
    def __init__(self, sess, FLAGS, file_comment=''):
            self.file_comment=file_comment
            self.reg = 0
            base_model.__init__(self, sess, FLAGS)
            
    def _buildnet(self):
        temp_only = self.FLAGS.temp_only

        self.x = tf.placeholder(tf.float32, shape=X_SHAPE, name='X')
        self.y = tf.placeholder(tf.float32, shape=Y_SHAPE, name='Y')
        
        if not temp_only:
            x0 = self.x[:, :X_SHAPE[1] // 2, :, :, :]
            x1 = self.x[:, X_SHAPE[1] // 2: X_SHAPE[1], :, :, :]
            x = self.x
            y = self.y
            predshape = [-1, Y_SHAPE[1], Y_SHAPE[3], Y_SHAPE[4]]
        else:
            x0 = self.x[:, 0: 1, :, :, :]
            x1 = self.x[:, X_SHAPE[1] // 2: X_SHAPE[1] // 2 + 1, :, :, :]
            x = tf.concat([x0, x1], axis=1)
            y = self.y[:, 0: 1, :, :, :]
            predshape = [-1, 1, Y_SHAPE[3], Y_SHAPE[4]]
            
        with tf.variable_scope("LCN0", reuse=False, regularizer=reg_init):
            
            lcn_kernel = ( int(self.FLAGS.lcn_kernel[0]), int(self.FLAGS.lcn_kernel[1]), int(self.FLAGS.lcn_kernel[2]) )
            self.myprint("LCN0: the kernel size is: " + str(lcn_kernel))
            out, reg0 = LCN3D_layer(x1, channels = 1, kernel=lcn_kernel, namespace ='conv_local3',
                                                  regularize=self.FLAGS.regularize, alpha=self.FLAGS.alpha, with_affine=True)

        with tf.variable_scope("Unet", reuse=False, initializer=base_init,
                                                 regularizer=reg_init):
            conv_0_1 = conv_batch_relu3d_layer(out, base_n_filt, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            conv_0_2 = conv_batch_relu3d_layer(conv_0_1, base_n_filt * 2, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            # Level one
            max_1_1 = tf.layers.max_pooling3d(conv_0_2, [1, 2, 2], [1, 2, 2],
                                                                                data_format="channels_first")  # Stride, Kernel previously [2,2,2]
            # pool_size:An integer or tuple/list of 3 integers: (pool_depth, pool_height, pool_width)

            conv_1_1 = conv_batch_relu3d_layer(max_1_1, base_n_filt * 2, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            conv_1_2 = conv_batch_relu3d_layer(conv_1_1, base_n_filt * 4, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            # Level two
            max_2_1 = tf.layers.max_pooling3d(conv_1_2, [1, 2, 2], [1, 2, 2],
                                                                                data_format="channels_first")  # Stride, Kernel previously [2,2,2]
            conv_2_1 = conv_batch_relu3d_layer(max_2_1, base_n_filt * 4, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            conv_2_2 = conv_batch_relu3d_layer(conv_2_1, base_n_filt * 8, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            # Level three
            max_3_1 = tf.layers.max_pooling3d(conv_2_2, [1, 2, 2], [1, 2, 2],
                                                                                data_format="channels_first")  # Stride, Kernel previously [2,2,2]
            conv_3_1 = conv_batch_relu3d_layer(max_3_1, base_n_filt * 8, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            conv_3_2 = conv_batch_relu3d_layer(conv_3_1, base_n_filt * 16, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            # Level two
            up_conv_3_2 = upconv3d_layer(conv_3_2, base_n_filt * 16)#, kernel=2,
                                                                        #stride=[1, 2, 2])  # Stride previously [2,2,2]
            concat_2_1 = centre_crop_and_concat_layer(conv_2_2, up_conv_3_2)
            conv_2_3 = conv_batch_relu3d_layer(concat_2_1, base_n_filt * 8, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            conv_2_4 = conv_batch_relu3d_layer(conv_2_3, base_n_filt * 8, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            # Level one
            up_conv_2_1 = upconv3d_layer(conv_2_4, base_n_filt * 8)#, kernel=2,
                                                                        #stride=[1, 2, 2])  # Stride previously [2,2,2]
            concat_1_1 = centre_crop_and_concat_layer(conv_1_2, up_conv_2_1)
            conv_1_3 = conv_batch_relu3d_layer(concat_1_1, base_n_filt * 4, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            conv_1_4 = conv_batch_relu3d_layer(conv_1_3, base_n_filt * 4, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            # Level zero
            up_conv_1_0 = upconv3d_layer(conv_1_4, base_n_filt * 4)#, kernel=2,
                                                                        #stride=[1, 2, 2])  # Stride previously [2,2,2]
            concat_0_1 = centre_crop_and_concat_layer(conv_0_2, up_conv_1_0)
            conv_0_3 = conv_batch_relu3d_layer(concat_0_1, base_n_filt * 2, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)
            conv_0_4 = conv_batch_relu3d_layer(conv_0_3, base_n_filt * 2, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad)

            conv_0_5 = tf.layers.conv3d(conv_0_4, y.shape[1], [1, 1, 1], [1, 1, 1], padding='same',
                                                                    data_format="channels_first")  # 1 instead of OUTPUT_CLASSES
            out = conv_0_5
        
        with tf.variable_scope("LCN0", reuse=False, regularizer=reg_init):
        
            lcn_kernel = ( int(self.FLAGS.lcn_kernel[0]), int(self.FLAGS.lcn_kernel[1]), int(self.FLAGS.lcn_kernel[2]) )
            self.myprint("LCN1: the kernel size is: " + str(lcn_kernel))
            out, reg1 = LCN3D_layer(out, channels = 1, kernel=lcn_kernel, namespace ='conv_local3',
                                            regularize=self.FLAGS.regularize, alpha=self.FLAGS.alpha, with_affine=True)
        self.pred = out + x1
            
        with tf.name_scope('train'):
            self.loss = getLoss(y, self.pred) + reg0 + reg1
            self.evaluate_op = getEvaluate(y, self.pred, self.file_comment)
            self.trainer = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Ensure correct ordering for batch-norm to work
            with tf.control_dependencies(self.extra_update_ops):
                self.train_op = self.trainer.minimize(self.loss)  # = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            self.summary_op = tf.summary.merge_all()