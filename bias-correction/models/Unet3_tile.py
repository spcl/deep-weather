#!/usr/bin/python
import tensorflow as tf
import numpy as np
from models.utils import variable_summaries, general_model, getLoss, getEvaluate, getCrop
from models.layers import conv_tile_batch_relu3d_layer, LCN3D_layer
from models.models import simple_conv_tile, unet3_l1_tile
import config

from models.models import unet3_l3, unet3_l2, unet3_l1, simple_conv

X_SHAPE = config.X_SHAPE
Y_SHAPE = config.Y_SHAPE
base_init = tf.truncated_normal_initializer(stddev=1e-2)  # Initialise weights
reg_init = tf.contrib.layers.l2_regularizer(scale=0.1)  # Initialise regularisation (was useful)


class Unet3_tile(general_model):
    def _buildnet(self):
        use_LCN = self.FLAGS.use_LCN
        self.varlist = None
        if self.FLAGS.recompute:
            self.varlist = []
        self.x = tf.placeholder(tf.float32, shape=X_SHAPE)
        self.y = tf.placeholder(tf.float32, shape=Y_SHAPE)

        # random crop:
        div = [9, 3]

        x, y = self.data_crop(div=div, crop=self.FLAGS.crop, stack=self.FLAGS.crop_stack)

        if self.FLAGS.img_emb:
            emb_img = tf.Variable(tf.zeros([1, 1] + list(X_SHAPE[2:])), name='emb_img')
            x = tf.concat([x, tf.tile(emb_img, [tf.shape(x)[0], 1, 1, 1, 1])], axis=1)

        # reshape for 1,2,3 layers unet
        x1 = x[:, X_SHAPE[1]//2: X_SHAPE[1], :, :, :]

        with tf.variable_scope("Unet", reuse=False, initializer=base_init, regularizer=reg_init):
            if self.FLAGS.unet_levels == 1:
                pred = unet3_l1_tile(self.FLAGS.nfilters, x, y, deconv=False, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad, varlist=self.varlist)
            elif self.FLAGS.unet_levels == 0:
                pred = simple_conv_tile(self.FLAGS.nfilters, x, y, deconv=False, is_training=self.FLAGS.train, is_pad=self.FLAGS.is_pad, varlist=self.varlist)
            else:
                assert 0, "Unet levels not supported, only 0,1,2,3 are supported"

        if use_LCN:
            lcn_kernel = (int(self.FLAGS.lcn_kernel[0]), int(self.FLAGS.lcn_kernel[1]), int(self.FLAGS.lcn_kernel[2]))
            out, reg = LCN3D_layer(pred, channels=1, kernel=lcn_kernel, namespace='conv_local3',
                               regularize=self.FLAGS.regularize, alpha=self.FLAGS.alpha, with_affine=True)
            self.pred = out + getCrop(x1)
        else:
            reg = 0
            self.pred = getCrop(pred + x1)

        self.loss = tf.losses.mean_squared_error(getCrop(y), self.pred) + reg
        variable_summaries(self.loss, 'loss')
        self.evaluate_op = getEvaluate(y, self.pred, self.file_comment, single_layer=True)
        self.build_train_op()
        self.summary_op = tf.summary.merge_all()
