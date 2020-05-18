#!/usr/bin/python
import tensorflow as tf
import random
import numpy as np
import os.path as osp
from models.utils import variable_summaries, base_model, getLoss, getEvaluate
import config

# define model parameter as macros here:
X_SHAPE = config.X_SHAPE
Y_SHAPE = config.Y_SHAPE

reg_init = tf.contrib.layers.l2_regularizer(scale=0.1)

class one2one(base_model):
    def __init__(self, sess, FLAGS, file_comment=''):
        self.file_comment=file_comment
        base_model.__init__(self, sess, FLAGS)

    def one2oneLayer(self, input, namespace):
        with tf.name_scope(namespace):
            shape = Y_SHAPE[1:]
            weights = tf.Variable(tf.zeros(shape), name = 'weights')
            bias = tf.Variable(tf.zeros(shape), name = 'bias')
            out = tf.multiply(input, weights) + bias
            with tf.name_scope('summary'):
                variable_summaries(weights, name = 'weights')
                variable_summaries(bias, name = 'bias')
        return out

    def _buildnet(self):
        self.x = tf.placeholder(tf.float32, shape=X_SHAPE, name='X')
        self.y = tf.placeholder(tf.float32, shape=Y_SHAPE, name='Y')

        x0 = self.x[:, :X_SHAPE[1]//2, :, :, :]
        x1 = self.x[:, X_SHAPE[1]//2: X_SHAPE[1], :, :, :]

        with tf.name_scope('one2one'):
            self.pred = self.one2oneLayer(x1, "l1") + x1
            self.loss = getLoss(self.y, self.pred)
            self.evaluate_op = getEvaluate(self.y, self.pred, self.file_comment)
            with tf.name_scope('summary'):
                tf.summary.scalar('MSE loss', self.loss)
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate = self.FLAGS.lr).minimize(self.loss)
        self.summary_op = tf.summary.merge_all()
