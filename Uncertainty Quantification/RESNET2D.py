#!/usr/bin/env python3
# model
import numpy as np
import tensorflow as tf
import parameters
import scipy

# Continuous Ranked Probability Score between a ground truth and a Gaussian distribution
# PRE
# std: Standard deviation of gaussian distribution
# dif: Difference between gaussian distribution mean and ground truth
# POST
# returns CRPS
def CRPS(std,dif):
    rp = np.sqrt(np.pi)
    r2 = np.sqrt(2)
    return dif*scipy.special.erf(dif/(r2*std)) + std/rp*(-1+r2*np.exp(-np.power(dif,2)/(2*np.power(std,2))))

#Overriding py_func
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


class Model():
    def inc(self, max_1_1, base_filt, is_training=True):
        # Dilated Convolutions can be ued here
        conv_1_2_a3 = tf.layers.conv2d(max_1_1, base_filt, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                       kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                       data_format="channels_first")
        conv_1_2_a5 = tf.layers.conv2d(max_1_1, base_filt, kernel_size=[5, 5], strides=[1, 1], dilation_rate=1, padding='same',
                                       kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                       data_format="channels_first")
        conv_1_2_a7 = tf.layers.conv2d(max_1_1, base_filt, kernel_size=[7, 7], strides=[1, 1], dilation_rate=1, padding='same',
                                       kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                       data_format="channels_first")

        conv_1_2 = tf.concat([conv_1_2_a3, conv_1_2_a5, conv_1_2_a7], 1)
        conv_1_2 = tf.layers.batch_normalization(conv_1_2, training=is_training, axis=1)
        conv_1_2 = tf.nn.relu(conv_1_2)
        conv_1_2 = tf.concat([conv_1_2,max_1_1],1)
        conv_1_2 = tf.layers.conv2d(conv_1_2, base_filt, kernel_size=[1, 1], strides=[1, 1], padding='same', kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,                                                                                  data_format="channels_first")
        return conv_1_2

    def conv_batch_relu(self, tensor, filters, kernel=[3, 3], stride=[1, 1], is_training=True):
        padding = 'valid'
        if self.should_pad: padding = 'same'

        conv = tf.layers.conv2d(tensor, filters, kernel_size=kernel, strides=stride, padding=padding,
                                kernel_initializer=self.base_init, kernel_regularizer=self.reg_init,
                                data_format="channels_first")
        conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)  # axis 1 because of channels first
        conv = tf.nn.relu(conv)
        return conv

    def __init__(self, timesteps = parameters.NR_TIMESTEPS, base_filt=parameters.BASEFILTER_SIZE, in_depth=parameters.INPUT_DEPTH,
                 out_depth=parameters.OUTPUT_DEPTH, in_sizel=parameters.INPUT_SIZEL, in_sizew=parameters.INPUT_SIZEW,
                 out_sizel=parameters.OUTPUT_SIZEL, out_sizew=parameters.OUTPUT_SIZEW,
                 learning_rate=parameters.LEARNING_RATE, print_shapes=True, should_pad=True,
                 nr_chan=parameters.NR_CHANNELS, p_temp = parameters.PREDICT_TEMP, do_concat = parameters.CONCAT_NWP, do_crps = parameters.DO_CRPS):

        # Initialize weights & regularization
        self.base_init = tf.truncated_normal_initializer(stddev=0.00001)
        self.reg_init = tf.contrib.layers.l2_regularizer(scale=0.0000001)

        self.should_pad = should_pad

        with tf.variable_scope('RESNET2D'):
            self.training = tf.placeholder(tf.bool)
            self.do_print = print_shapes
            # Define placeholders for feed_dict
            self.model_input = tf.placeholder(tf.float32, shape=(
            None, timesteps, nr_chan, in_depth, in_sizew, in_sizel))  # batch none to input it later
            if do_crps:
                self.model_cout = tf.placeholder(tf.float32, shape=(None, 1, out_sizew, out_sizel))
            else:
                self.model_cout = tf.placeholder(tf.float32, shape=(None, 2, 7, out_depth, out_sizew, out_sizel))
            if self.do_print:
                print('Input features shape', self.model_input.get_shape())
                print('Output shape', self.model_cout.get_shape())


            # Feature selection + NWP initial Post-Processing steps
            tp = self.model_input[:,:,:7,0,:,:]
            tp = tf.reshape(tp, shape=(-1,timesteps*7,out_sizew, out_sizel))
            cp = self.conv_batch_relu(tp,3,kernel=[1,1],stride=[1,1], is_training=self.training)
            t0 = self.model_input[:,:,7:,0,:,:]
            if p_temp:
                tgo = t0[:, :, None, 0, :, :]
            else:
                tgo = t0[:, :, None, 5, :, :]/900 # Additional normalization heuristic of geopotential spread -> faster training
            t0 = tf.concat([t0[:,:,:5,:,:],tgo,t0[:,:,None,6,:,:]],axis = 2)
            inp = tf.reshape(t0, shape=(-1, timesteps*7, out_sizew, out_sizel))
            inp = tf.concat([inp,cp], axis=1)

            if do_crps:
                out = self.model_cout
            else:
                if p_temp:
                    out = self.model_cout[:,0,:,:,:,:]
                else:
                    out = self.model_cout[:,None,0,5,0,:,:]

            c3 = self.conv_batch_relu(inp, base_filt*4, is_training=self.training)

            # Inception style layers
            c1 = inp
            co = c3
            for i in range(10):
                c2 = tf.concat([c3, c1], axis=1)
                c3 = self.inc(c2, base_filt*4+i*4, is_training=self.training)

            c3 = tf.concat([co,c3],1)

            # Prediction Combination
            c3 = self.conv_batch_relu(c3, 8, kernel=[3, 3], stride=[1, 1], is_training=self.training)
            conv_0 = tf.layers.conv2d(c3, 4, [1, 1], [1, 1],kernel_initializer=self.base_init, padding='same',
                                        data_format="channels_first")
            if do_concat:
                conv_0 = tf.concat([conv_0,tgo[:,2,:,:,:]],axis=1) # concatenate original NWP prediction

            conv_1 = tf.layers.conv2d(conv_0, 1, [1, 1], [1, 1],kernel_initializer=self.base_init, padding='same',
                                        data_format="channels_first")
            self.predictions = tf.math.abs(conv_1) if do_crps else conv_1
            if not p_temp:
                self.predictions *= 900
            if self.do_print:
                print('Model output shape', self.predictions.get_shape())

            # Losses for metric aggregation
            self.linf_loss = tf.reduce_max(
                tf.abs(tf.subtract(self.predictions, out)))
            if do_crps: # Calculate CRPS loss
                self.loss = tf.reduce_mean(tf.py_function(func=CRPS, inp=[self.predictions, out],
                                                          Tout=tf.float32))
                self.loss2 = tf.reduce_mean(tf.py_function(func=CRPS, inp=[self.predictions, out],
                                                           Tout=tf.float32)) + tf.losses.get_regularization_loss()
            else:       # Calculate MSE & SSIM loss
                self.loss = tf.losses.mean_squared_error(out, self.predictions)
                if p_temp:
                    self.loss2 = -tf.reduce_mean(tf.image.ssim(tf.transpose(self.predictions, [0, 2, 3, 1]), tf.transpose(out, [0, 2, 3, 1]), max_val=4.3001)) + tf.losses.get_regularization_loss() # max val set to 4.3001 for temperature, evaluated from data
                else:
                    self.loss2 = -tf.reduce_mean(tf.image.ssim(tf.transpose(self.predictions,[0,2,3,1]), tf.transpose(out,[0,2,3,1]),max_val=1)) + tf.losses.get_regularization_loss() # max val 4.3001 for geopotential

            self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.extra_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS)  # Ensure correct ordering for batch-norm to work
            with tf.control_dependencies(self.extra_update_ops):
                self.train_op = self.trainer.minimize(
                   self.loss2)
