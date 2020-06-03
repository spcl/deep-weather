#!/usr/bin/python
import tensorflow as tf
import random
import numpy as np
import os.path as osp
from models.utils import variable_summaries, base_model, getCrop, getLoss, getEvaluate
import config

from models.layers import affine_layer, LCN2D_layer, LCN3D_layer

# define model parameter as macros here:
X_SHAPE = config.X_SHAPE
Y_SHAPE = config.Y_SHAPE

base_init = tf.truncated_normal_initializer(stddev=0.1)
reg_init = tf.contrib.layers.l2_regularizer(scale=0.1)


class LCN(base_model):
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
        else:
            x0 = self.x[:, 0: 1, :, :, :]
            x1 = self.x[:, X_SHAPE[1] // 2: X_SHAPE[1] // 2 + 1, :, :, :]
            x = tf.concat([x0, x1], axis=1)
            y = self.y[:, 0: 1, :, :, :]

        with tf.variable_scope("LCN", reuse=False):
            lcn_kernel = self.FLAGS.lcn_kernel
            if len(lcn_kernel) == 3:
                lcn_kernel = (int(self.FLAGS.lcn_kernel[0]), int(self.FLAGS.lcn_kernel[1]), int(self.FLAGS.lcn_kernel[2]))
            elif len(lcn_kernel) == 2:
                print("Converting 2D lcn_kernel to 3D with height 1")
                lcn_kernel = ( 1, int(self.FLAGS.lcn_kernel[0]), int(self.FLAGS.lcn_kernel[1]))
            self.myprint("LCN: the kernel size is: " + str(lcn_kernel))
            self.pred, reg = LCN3D_layer(x1, channels=1, kernel=lcn_kernel, namespace='conv_local3',
                                    regularize=self.FLAGS.regularize, alpha=self.FLAGS.alpha, with_affine=True)
            self.pred = self.pred + getCrop(x1)
            self.loss = tf.losses.mean_squared_error(getCrop(y), self.pred) + reg
            self.evaluate_op = getEvaluate(self.y, self.pred, self.file_comment, single_layer=True)

            with tf.name_scope('summary'):
                tf.summary.scalar('MSE loss', self.loss)

        self.build_train_op()
        self.summary_op = tf.summary.merge_all()


if __name__ == "__main__":
    ic = 2
    xshape=[5,ic,2,2]
    tensor = tf.Variable(tf.random_uniform(xshape, 0, 1, dtype=tf.float32, seed=0))
    #tensor = tf.constant(value=1, dtype=tf.float32, shape=xshape)
    kernel = 3
    channels = 2

    padsize = (kernel - 1) // 2
    shape = tensor.get_shape()
    c = int(shape[1])
    w = int(shape[2])
    h = int(shape[3])
    oc = channels

    paddings = tf.constant([ [0,0], [0,0], [padsize, padsize], [padsize, padsize] ])
    paddedtensor = tf.pad(tensor, paddings, "SYMMETRIC")

    weight_shape = [c, w, h, oc, kernel, kernel]
    bias_shape = [w, h, oc]
    #weights = tf.get_variable('x', shape=[2, 4], trainable = True, initializer=tf.constant_initializer(1/(c*kernel*kernel)))
    norm_factor = tf.constant(c*kernel*kernel, dtype=tf.float32)
    weights = tf.Variable(tf.ones(weight_shape, dtype=tf.float32), name = 'conv_local2_weights')
    bias = tf.Variable(tf.zeros(bias_shape, dtype=tf.float32), name = 'conv_local2_bias')

    result = [None] * oc
    for k in range (oc):
        for i in range (kernel):
            for j in range (kernel):
                if i==0 and j==0:
                    result[k] = tf.reshape(weights[:,:,:,k,i,j], [c, w, h]) * tf.reshape(paddedtensor[:, :, i:i+w , j:j+h], [-1, c, w, h]) + bias[:,:,k]
                else:
                    result[k] = result[k] + tf.reshape(weights[:,:,:,k,i,j], [c, w, h]) * tf.reshape(paddedtensor[:, :, i:i+w , j:j+h], [-1, c, w, h])

    oc_result = tf.concat([tf.reshape(t, [-1, 1, w, h]) for t in result], axis = 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # npw = sess.run(weights)
        # npb = sess.run(bias)
        # print(npw)
        # print(npb)
        # tr = sess.run(tensor)
        ptr = sess.run(paddedtensor)
        r = sess.run(oc_result)
        #r1 = sess.run(result1)

        item = 0

        val = 0
        xpos = 0
        ypos = 0
        for k in range(ic):
            for i in range(kernel):
                for j in range(kernel):
                    val += ptr[item, k, xpos+i, ypos+j]

        # print(tr[1,:,:,:])
        print(ptr[item,:,:,:])
        print(r[item,:,:,:])
        #print(r1[item,:,:,:])
        print(val)
