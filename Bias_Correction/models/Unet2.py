#!/usr/bin/python
import tensorflow as tf
import random
import numpy as np
import os.path as osp
from models.utils import variable_summaries, base_model, getLoss, getEvaluate
import config

X_SHAPE = config.X_SHAPE
Y_SHAPE = config.Y_SHAPE
base_n_filt = 4
base_init = tf.truncated_normal_initializer(stddev=0.1)  # Initialise weights
reg_init = tf.contrib.layers.l2_regularizer(scale=0.1)  # Initialise regularisation (was useful)


class Unet2(base_model):

    def __init__(self, sess, FLAGS, file_comment=''):
        self.file_comment=file_comment
        base_model.__init__(self, sess, FLAGS)


    def conv_batch_relu2d(self, tensor, filters, kernel=[3, 3], stride=[1, 1], is_training=True):

        if self.FLAGS.is_pad:    padding = 'same'
        else:   padding = 'valid'

        conv = tf.layers.conv2d(tensor, filters, kernel_size=kernel, strides=stride, padding=padding,
                                kernel_initializer=base_init, kernel_regularizer=reg_init,
                                data_format="channels_first")  # , data_format = "NCDHW"
        conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)  # axis 1 because of channels first
        conv = tf.nn.relu(conv)
        return conv


    def upconvolve2d(self, tensor, filters, kernel=[2, 2], stride=[2, 2], scale=4, activation=None):

        if self.FLAGS.is_pad:    padding = 'same'
        else:    padding = 'valid'

        conv = tf.layers.conv2d_transpose(tensor, filters, kernel_size=kernel, strides=stride, padding=padding,
                                          use_bias=False,
                                          kernel_initializer=base_init, kernel_regularizer=reg_init,
                                          data_format="channels_first")
        return conv


    def centre_crop_and_concat(self, prev_conv, up_conv):
        # If concatenating two different sized Tensors, centre crop the first Tensor to the right size and concat
        # Needed if you don't have padding
        p_c_s = prev_conv.get_shape()
        u_c_s = up_conv.get_shape()
        offsets = np.array([0, 0, (p_c_s[2] - u_c_s[2]) // 2, (p_c_s[3] - u_c_s[3]) // 2], dtype=np.int32)
        size = np.array([-1, p_c_s[1], u_c_s[2], u_c_s[3]], np.int32)
        prev_conv_crop = tf.slice(prev_conv, offsets, size)
        up_concat = tf.concat((prev_conv_crop, up_conv), 1)
        return up_concat


    def _buildnet(self):
        with tf.variable_scope('2DUnet'):
            self.x = tf.placeholder(tf.float32, shape=X_SHAPE)  # batch none to input it later
            self.y = tf.placeholder(tf.float32, shape=Y_SHAPE)


            x0 = self.x[:, :X_SHAPE[1]//2, :, :, :]
            x1 = self.x[:, X_SHAPE[1]//2: X_SHAPE[1], :, :, :]

            x0d2 = tf.reshape(x0, [-1, X_SHAPE[1]*X_SHAPE[2]//2, X_SHAPE[3], X_SHAPE[4]])
            x1d2 = tf.reshape(x1, [-1, X_SHAPE[1]*X_SHAPE[2]//2, X_SHAPE[3], X_SHAPE[4]])

            xd2 = tf.reshape(self.x, [-1, X_SHAPE[1]*X_SHAPE[2], X_SHAPE[3], X_SHAPE[4]])
            yd2 = tf.reshape(self.y, [-1, Y_SHAPE[1]*Y_SHAPE[2], Y_SHAPE[3], Y_SHAPE[4]])



            conv_0_1 = self.conv_batch_relu2d(xd2, base_n_filt, is_training=self.FLAGS.train)
            conv_0_2 = self.conv_batch_relu2d(conv_0_1, base_n_filt * 2, is_training=self.FLAGS.train)
            # Level one
            max_1_1 = tf.layers.max_pooling2d(conv_0_2, [2, 2], [2, 2], data_format="channels_first")
            conv_1_1 = self.conv_batch_relu2d(max_1_1, base_n_filt * 2, is_training=self.FLAGS.train)
            conv_1_2 = self.conv_batch_relu2d(conv_1_1, base_n_filt * 4, is_training=self.FLAGS.train)
            # Level two
            max_2_1 = tf.layers.max_pooling2d(conv_1_2, [2, 2], [2, 2], data_format="channels_first")
            conv_2_1 = self.conv_batch_relu2d(max_2_1, base_n_filt * 4, is_training=self.FLAGS.train)
            conv_2_2 = self.conv_batch_relu2d(conv_2_1, base_n_filt * 8, is_training=self.FLAGS.train)
            # Level three
            max_3_1 = tf.layers.max_pooling2d(conv_2_2, [2, 2], [2, 2], data_format="channels_first")
            conv_3_1 = self.conv_batch_relu2d(max_3_1, base_n_filt * 8, is_training=self.FLAGS.train)
            conv_3_2 = self.conv_batch_relu2d(conv_3_1, base_n_filt * 16, is_training=self.FLAGS.train)
            # Level two
            up_conv_3_2 = self.upconvolve2d(conv_3_2, base_n_filt * 16, kernel=[2, 2], stride=[2, 2])
            concat_2_1 = self.centre_crop_and_concat(conv_2_2, up_conv_3_2)
            conv_2_3 = self.conv_batch_relu2d(concat_2_1, base_n_filt * 8, is_training=self.FLAGS.train)
            conv_2_4 = self.conv_batch_relu2d(conv_2_3, base_n_filt * 8, is_training=self.FLAGS.train)
            # Level one
            up_conv_2_1 = self.upconvolve2d(conv_2_4, base_n_filt * 8, kernel=[2, 2], stride=[2, 2])
            concat_1_1 = self.centre_crop_and_concat(conv_1_2, up_conv_2_1)
            conv_1_3 = self.conv_batch_relu2d(concat_1_1, base_n_filt * 4, is_training=self.FLAGS.train)
            conv_1_4 = self.conv_batch_relu2d(conv_1_3, base_n_filt * 4, is_training=self.FLAGS.train)
            # Level zero
            up_conv_1_0 = self.upconvolve2d(conv_1_4, base_n_filt * 4, kernel=[2, 2], stride=[2, 2])
            concat_0_1 = self.centre_crop_and_concat(conv_0_2, up_conv_1_0)
            concat_0_2 = self.centre_crop_and_concat(concat_0_1, xd2)
            conv_0_3 = self.conv_batch_relu2d(concat_0_2, base_n_filt * 2, is_training=self.FLAGS.train)
            conv_0_4 = self.conv_batch_relu2d(conv_0_3, base_n_filt * 2, is_training=self.FLAGS.train)
            conv_0_5 = tf.layers.conv2d(conv_0_4, 1, [1, 1], [1, 1], padding='same', data_format="channels_first")
            self.pred = conv_0_5 + x1d2

        with tf.name_scope('train'):
            self.loss = getLoss(yd2, self.pred)
            self.evaluate_op = self.loss
            self.trainer = tf.train.AdamOptimizer(learning_rate=self.FLAGS.lr)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Ensure correct ordering for batch-norm to work
            with tf.control_dependencies(self.extra_update_ops):
                self.train_op = self.trainer.minimize(self.loss)  # = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            self.summary_op = tf.summary.merge_all()
