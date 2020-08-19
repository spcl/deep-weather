#!/usr/bin/python
import tensorflow as tf
import numpy as np
from models.utils import variable_summaries, general_model, getLoss, getEvaluate, getCrop, getSSIM
import config
from models.models import unet3_l3, unet3_l2, unet3_l1, simple_conv
from models.layers import LCN3D_layer
import scipy

X_SHAPE = (None, 1, 361, 720, 5)
Y_SHAPE = (None, 1, 361, 720)

base_init = tf.truncated_normal_initializer(stddev=1e-1)  # Initialise weights
reg_init = tf.contrib.layers.l2_regularizer(scale=0.1)  # Initialise regularisation (was useful)


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

def reduce_var(x, axis=None, keepdims=False):
        m = tf.reduce_mean(x, axis=axis, keep_dims=True)
        devs_squared = tf.square(x - m)
        return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


class emos(general_model):
    def _buildnet(self):
        self.varlist = None
        if self.FLAGS.recompute:
            self.varlist = []
        self.x = tf.placeholder(tf.float32, shape=X_SHAPE)
        self.y = tf.placeholder(tf.float32, shape=Y_SHAPE)
        #out = [None] * 5
        #for i in range(5):
        #    out[i] = tf.placeholder(tf.float32, shape=Y_SHAPE)
        

        bias = tf.Variable(tf.zeros([1,1,1,1,1]), name="a", dtype=tf.float32)
        weights = tf.Variable(tf.ones([1,1,1,1,5]), name="b", dtype=tf.float32)
        std_bias = tf.Variable(tf.zeros([1,1,1,1,1]), name="c", dtype=tf.float32)
        std_weight = tf.Variable(tf.ones([1,1,1,1,1]), name="d", dtype=tf.float32)

        #for i in range(5):
        #    out[i] = self.x[:,:,:,:,i] * weights[i]
        #outvec = tf.stack(out, axis=4)

        outvec = self.x * weights

        stddev = tf.sqrt(reduce_var(self.x, 4) * std_weight + std_bias)
        mean = tf.reduce_mean(outvec, 4) + bias
        diff = mean - self.y

        self.pred = tf.reduce_mean(self.x, 4)
        self.loss = tf.reduce_mean(tf.py_function(func=CRPS, inp=[stddev, diff], Tout=tf.float32))
        
        variable_summaries(self.loss, 'loss')
        self.build_train_op()
        self.summary_op = tf.summary.merge_all()


