#!/usr/bin/python
import tensorflow as tf
import random
import numpy as np
import os.path as osp
from models.utils import variable_summaries, base_model, getLoss, getEvaluate, getLossSlice
import config

# define model parameter as macros here:
X_SHAPE = config.X_SHAPE
Y_SHAPE = config.Y_SHAPE

class refmse(base_model):
    def __init__(self, sess, FLAGS, file_comment=''):
        self.file_comment=file_comment
        base_model.__init__(self, sess, FLAGS)

    def _buildnet(self):
        self.x = tf.placeholder(tf.float32, shape=X_SHAPE, name='X')
        self.y = tf.placeholder(tf.float32, shape=Y_SHAPE, name='Y')
        with tf.name_scope('MSE_loss'):
            y = self.y
            self.pred = getLossSlice(self.x[:, X_SHAPE[1]//2: X_SHAPE[1], :, :, :])
            self.loss = getLoss(y, self.pred, single_layer=True)
            self.evaluate_op = getEvaluate(y, self.pred, self.file_comment, single_layer=True)
            with tf.name_scope('summary'):
                tf.summary.scalar('MSE loss', self.loss)
        with tf.name_scope('train'):
            self.train_op = self.loss
        self.summary_op = tf.summary.merge_all()
