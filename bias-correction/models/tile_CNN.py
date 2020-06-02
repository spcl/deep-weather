#!/usr/bin/python
import tensorflow as tf
import random
import numpy as np
import os.path as osp
from models.utils import variable_summaries, base_model, getLoss, getEvaluate
import config

from models.layers import tile_conv_layer

X_SHAPE = config.X_SHAPE
Y_SHAPE = config.Y_SHAPE

init = tf.truncated_normal_initializer(stddev=1e-4)
base_init = tf.truncated_normal_initializer(stddev=1e-4)


class TileCNN(base_model):
    def __init__(self, sess, FLAGS, file_comment=''):
        self.file_comment = file_comment
        base_model.__init__(self, sess, FLAGS)

    def _buildnet(self):
        temp_only = self.FLAGS.temp_only

        self.x = tf.placeholder(tf.float32, shape=X_SHAPE)
        self.y = tf.placeholder(tf.float32, shape=Y_SHAPE)

        if not temp_only:
            x0 = self.x[:, :X_SHAPE[1] // 2, :, :, :]
            x1 = self.x[:, X_SHAPE[1] // 2: X_SHAPE[1], :, :, :]
            x = self.x
            y = self.y
        else:
            x0 = self.x[:, 0: 1, :, :, :]
            x1 = self.x[:, X_SHAPE[1] // 2: X_SHAPE[1] // 2 + 1, :, :, :]
            x = tf.concat([x0, x1], axis=1)
            y = self.y[:, 0: 1, :, :, :]

        xd2 = tf.reshape(self.x, [-1, X_SHAPE[1]*X_SHAPE[2], X_SHAPE[3], X_SHAPE[4]])
        x0d2 = tf.reshape(x0, [-1, X_SHAPE[1]*X_SHAPE[2]//2, X_SHAPE[3], X_SHAPE[4]])
        x1d2 = tf.reshape(x1, [-1, X_SHAPE[1]*X_SHAPE[2]//2, X_SHAPE[3], X_SHAPE[4]])

        xd2 = tf.reshape(self.x, [-1, X_SHAPE[1]*X_SHAPE[2], X_SHAPE[3], X_SHAPE[4]])
        yd2 = tf.reshape(self.y, [-1, Y_SHAPE[1]*Y_SHAPE[2], Y_SHAPE[3], Y_SHAPE[4]])

        with tf.variable_scope("tile_CNN", reuse=False, initializer=base_init, regularizer=None):
            if Y_SHAPE[2] == 1:
                lcn_kernel = int(self.FLAGS.lcn_kernel[0])
                self.pred = tf.layers.conv2d(x1d2, 1, kernel_size=lcn_kernel, strides=[1, 1],
                    padding = 'SAME', data_format="channels_first")
                self.pred = self.pred + x1d2
                self.loss = getLoss(y, self.pred)

            else:
                lcn_kernel = (
                int(self.FLAGS.lcn_kernel[0]), int(self.FLAGS.lcn_kernel[1]), int(self.FLAGS.lcn_kernel[2]))
                self.myprint("TileCNN: the kernel size is: " + str(lcn_kernel))
                div = (2,3)
                self.pred = tile_conv_layer(x1, div, 1, kernel_size=lcn_kernel, strides=[1, 1, 1], data_format="channels_first")
                self.pred = self.pred + x1
                self.loss = getLoss(y, self.pred)
                self.evaluate_op = getEvaluate(y, self.pred, self.file_comment)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate = self.FLAGS.lr).minimize(self.loss)
            self.summary_op = tf.summary.merge_all()
