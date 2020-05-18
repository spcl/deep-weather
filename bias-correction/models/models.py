import tensorflow as tf
import numpy as np
from models.layers import upconv_op, conv_batch_relu3d_layer, centre_crop_and_concat_layer, conv_tile_batch_relu3d_layer, LCN3D_layer


def simple_conv(base_n_filt, x, y, deconv=False, kernel_conv=[3,3,3], kernel_deconv=[1,3,3], is_training=True, is_pad=False, varlist=None):
    # Level zero
    conv_0_1 = conv_batch_relu3d_layer(x, base_n_filt, kernel=kernel_conv, is_training=is_training, is_pad=is_pad)
    conv_0_2 = conv_batch_relu3d_layer(conv_0_1, base_n_filt * 2, kernel=kernel_conv, is_training=is_training, is_pad=is_pad)
    conv_0_3 = conv_batch_relu3d_layer(conv_0_2, base_n_filt * 2, is_training=is_training, kernel=kernel_conv, is_pad=is_pad)

    if varlist is not None:
        varlist.append(conv_0_3)

    conv_0_4 = conv_batch_relu3d_layer(conv_0_3, base_n_filt * 2, is_training=is_training, kernel=kernel_conv, is_pad=is_pad)
    conv_0_5 = tf.layers.conv3d(conv_0_4, y.shape[1], [1, 1, 1], [1, 1, 1], padding='same',
                                data_format="channels_first")  # 1 instead of OUTPUT_CLASSES
    return conv_0_5


def simple_conv_tile(base_n_filt, x, y, deconv=False, kernel_conv=[1,3,3], kernel_deconv=[1,3,3], is_training=True, is_pad=False, varlist=None, div = (9, 3)):
    # Level zero
    conv_0_1 = conv_tile_batch_relu3d_layer(x, div, base_n_filt, kernel_size=kernel_conv, strides=[1, 1, 1])
    conv_0_2 = conv_tile_batch_relu3d_layer(conv_0_1, div, base_n_filt * 2, kernel_size=kernel_conv, strides=[1, 1, 1])
    conv_0_3 = conv_tile_batch_relu3d_layer(conv_0_2, div, base_n_filt * 2, kernel_size=kernel_conv, strides=[1, 1, 1])
    conv_0_4 = conv_tile_batch_relu3d_layer(conv_0_3, div, base_n_filt * 2, kernel_size=kernel_conv, strides=[1, 1, 1])
    conv_0_5 = tf.layers.conv3d(conv_0_4, y.shape[1], [1, 1, 1], [1, 1, 1], padding='same', data_format="channels_first")
    return conv_0_5


def unet3_l1(base_n_filt, x, y, deconv=False, kernel_conv=[3,3,3], kernel_deconv=[1,3,3], is_training=True, is_pad=False, varlist=None):

    # Level zero
    conv_0_1 = conv_batch_relu3d_layer(x, base_n_filt, kernel=kernel_conv, is_training=is_training, is_pad=is_pad)
    conv_0_2 = conv_batch_relu3d_layer(conv_0_1, base_n_filt * 2, kernel=kernel_conv, is_training=is_training, is_pad=is_pad)
    if varlist is not None:
        varlist.append(conv_0_2)
    # Level one
    max_1_1 = tf.layers.max_pooling3d(conv_0_2, [1, 2, 2], [1, 2, 2], data_format="channels_first")
    conv_1_1 = conv_batch_relu3d_layer(max_1_1, base_n_filt * 2, kernel=kernel_conv, is_training=is_training,  is_pad=is_pad)
    conv_1_2 = conv_batch_relu3d_layer(conv_1_1, base_n_filt * 4, kernel=kernel_conv, is_training=is_training, is_pad=is_pad)

    # Level zero
    up_conv_1_0 = upconv_op(conv_1_2, base_n_filt * 4, deconv=deconv, kernel=kernel_deconv)
    concat_0_1 = centre_crop_and_concat_layer(conv_0_2, up_conv_1_0)

    if varlist is not None:
        varlist.append(concat_0_1)
    conv_0_3 = conv_batch_relu3d_layer(concat_0_1, base_n_filt * 2, is_training=is_training, kernel=kernel_conv, is_pad=is_pad)
    conv_0_4 = conv_batch_relu3d_layer(conv_0_3, base_n_filt * 2, is_training=is_training, kernel=kernel_conv, is_pad=is_pad)
    conv_0_5 = tf.layers.conv3d(conv_0_4, y.shape[1], [1, 1, 1], [1, 1, 1], padding='same', data_format="channels_first")
    return conv_0_5


def unet3_l1_tile(base_n_filt, x, y, deconv=False, kernel_conv=[1,3,3], kernel_deconv=[1,3,3], is_training=True, is_pad=False, varlist=None, div = (9, 3)):
    # Level zero
    conv_0_1 = conv_tile_batch_relu3d_layer(x, div, base_n_filt, kernel_size=kernel_conv, strides=[1, 1, 1])
    conv_0_2 = conv_tile_batch_relu3d_layer(conv_0_1, div, base_n_filt * 2, kernel_size=kernel_conv, strides=[1, 1, 1])

    max_1_1 = tf.layers.max_pooling3d(conv_0_2, [1, 2, 2], [1, 2, 2], data_format="channels_first")
    conv_1_1 = conv_batch_relu3d_layer(max_1_1, base_n_filt * 2, kernel=kernel_conv, is_training=is_training,  is_pad=is_pad)
    conv_1_2 = conv_batch_relu3d_layer(conv_1_1, base_n_filt * 4, kernel=kernel_conv, is_training=is_training, is_pad=is_pad)

    # Level zero
    up_conv_1_0 = upconv_op(conv_1_2, base_n_filt * 4, deconv=deconv, kernel=kernel_deconv)
    concat_0_1 = centre_crop_and_concat_layer(conv_0_2, up_conv_1_0)

    conv_0_3 = conv_tile_batch_relu3d_layer(concat_0_1, div, base_n_filt * 2, kernel_size=kernel_conv, strides=[1, 1, 1])
    conv_0_4 = conv_tile_batch_relu3d_layer(conv_0_3, div, base_n_filt * 2, kernel_size=kernel_conv, strides=[1, 1, 1])
    conv_0_5 = tf.layers.conv3d(conv_0_4, y.shape[1], [1, 1, 1], [1, 1, 1], padding='same', data_format="channels_first")
    return conv_0_5


def unet3_l2(base_n_filt, x, y, deconv=False, kernel_conv=[3,3,3], kernel_deconv=[1,3,3], is_training=True, is_pad=False, varlist=None):

    # Level zero
    conv_0_1 = conv_batch_relu3d_layer(x, base_n_filt, kernel=kernel_conv, is_training=is_training, is_pad=is_pad)
    conv_0_2 = conv_batch_relu3d_layer(conv_0_1, base_n_filt * 2, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    if varlist is not None:
        varlist.append(conv_0_2)
    # Level one
    max_1_1 = tf.layers.max_pooling3d(conv_0_2, [1, 2, 2], [1, 2, 2],
                                      data_format="channels_first")  # Stride, Kernel previously [2,2,2]
    # pool_size:An integer or tuple/list of 3 integers: (pool_depth, pool_height, pool_width)

    conv_1_1 = conv_batch_relu3d_layer(max_1_1, base_n_filt * 2, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    conv_1_2 = conv_batch_relu3d_layer(conv_1_1, base_n_filt * 4, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    if varlist is not None:
        varlist.append(conv_1_2)

    # Level two
    max_2_1 = tf.layers.max_pooling3d(conv_1_2, [1, 2, 2], [1, 2, 2],
                                      data_format="channels_first")  # Stride, Kernel previously [2,2,2]
    conv_2_1 = conv_batch_relu3d_layer(max_2_1, base_n_filt * 4, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    conv_2_2 = conv_batch_relu3d_layer(conv_2_1, base_n_filt * 8, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    # Level one
    up_conv_2_1 = upconv_op(conv_2_2, base_n_filt * 8, deconv=deconv, kernel=kernel_deconv)  # , kernel=2,
    # stride=[1, 2, 2])  # Stride previously [2,2,2]
    concat_1_1 = centre_crop_and_concat_layer(conv_1_2, up_conv_2_1)

    if varlist is not None:
        varlist.append(concat_1_1)

    conv_1_3 = conv_batch_relu3d_layer(concat_1_1, base_n_filt * 4, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    conv_1_4 = conv_batch_relu3d_layer(conv_1_3, base_n_filt * 4, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    # Level zero
    up_conv_1_0 = upconv_op(conv_1_4, base_n_filt * 4, deconv=deconv, kernel=kernel_deconv)

    concat_0_1 = centre_crop_and_concat_layer(conv_0_2, up_conv_1_0)
    if varlist is not None:
        varlist.append(concat_0_1)
    conv_0_3 = conv_batch_relu3d_layer(concat_0_1, base_n_filt * 2, is_training=is_training, kernel=kernel_conv,
                                       is_pad=is_pad)
    conv_0_4 = conv_batch_relu3d_layer(conv_0_3, base_n_filt * 2, is_training=is_training, kernel=kernel_conv,
                                       is_pad=is_pad)
    conv_0_5 = tf.layers.conv3d(conv_0_4, y.shape[1], [1, 1, 1], [1, 1, 1], padding='same',
                                data_format="channels_first")  # 1 instead of OUTPUT_CLASSES
    return conv_0_5



def unet3_l3(base_n_filt, x, y, deconv=False, kernel_conv=[3,3,3], kernel_deconv=[1,3,3], is_training=True, is_pad=False, varlist=None):

    # Level zero
    conv_0_1 = conv_batch_relu3d_layer(x, base_n_filt, kernel=kernel_conv, is_training=is_training, is_pad=is_pad)
    conv_0_2 = conv_batch_relu3d_layer(conv_0_1, base_n_filt * 2, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    if varlist is not None:
        varlist.append(conv_0_2)

    # Level one
    max_1_1 = tf.layers.max_pooling3d(conv_0_2, [1, 2, 2], [1, 2, 2],
                                      data_format="channels_first")  # Stride, Kernel previously [2,2,2]
    # pool_size:An integer or tuple/list of 3 integers: (pool_depth, pool_height, pool_width)

    conv_1_1 = conv_batch_relu3d_layer(max_1_1, base_n_filt * 2, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    conv_1_2 = conv_batch_relu3d_layer(conv_1_1, base_n_filt * 4, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    if varlist is not None:
        varlist.append(conv_1_2)

    # Level two
    max_2_1 = tf.layers.max_pooling3d(conv_1_2, [1, 2, 2], [1, 2, 2],
                                      data_format="channels_first")  # Stride, Kernel previously [2,2,2]
    conv_2_1 = conv_batch_relu3d_layer(max_2_1, base_n_filt * 4, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    conv_2_2 = conv_batch_relu3d_layer(conv_2_1, base_n_filt * 8, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)

    if varlist is not None:
        varlist.append(conv_2_2)

    # Level three
    max_3_1 = tf.layers.max_pooling3d(conv_2_2, [1, 2, 2], [1, 2, 2],
                                      data_format="channels_first")  # Stride, Kernel previously [2,2,2]
    conv_3_1 = conv_batch_relu3d_layer(max_3_1, base_n_filt * 8, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    conv_3_2 = conv_batch_relu3d_layer(conv_3_1, base_n_filt * 16, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    # Level two
    up_conv_3_2 = upconv_op(conv_3_2, base_n_filt * 16, deconv=deconv, kernel=kernel_deconv)  # , kernel=2,
    # stride=[1, 2, 2])  # Stride previously [2,2,2]
    concat_2_1 = centre_crop_and_concat_layer(conv_2_2, up_conv_3_2)

    if varlist is not None:
        varlist.append(concat_2_1)

    conv_2_3 = conv_batch_relu3d_layer(concat_2_1, base_n_filt * 8, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    conv_2_4 = conv_batch_relu3d_layer(conv_2_3, base_n_filt * 8, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    # Level one
    up_conv_2_1 = upconv_op(conv_2_4, base_n_filt * 8, deconv=deconv, kernel=kernel_deconv)  # , kernel=2,
    # stride=[1, 2, 2])  # Stride previously [2,2,2]
    concat_1_1 = centre_crop_and_concat_layer(conv_1_2, up_conv_2_1)

    if varlist is not None:
        varlist.append(concat_1_1)

    conv_1_3 = conv_batch_relu3d_layer(concat_1_1, base_n_filt * 4, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    conv_1_4 = conv_batch_relu3d_layer(conv_1_3, base_n_filt * 4, kernel=kernel_conv, is_training=is_training,
                                       is_pad=is_pad)
    # Level zero
    up_conv_1_0 = upconv_op(conv_1_4, base_n_filt * 4, deconv=deconv, kernel=kernel_deconv)

    concat_0_1 = centre_crop_and_concat_layer(conv_0_2, up_conv_1_0)

    if varlist is not None:
        varlist.append(concat_0_1)

    conv_0_3 = conv_batch_relu3d_layer(concat_0_1, base_n_filt * 2, is_training=is_training, kernel=kernel_conv,
                                       is_pad=is_pad)
    conv_0_4 = conv_batch_relu3d_layer(conv_0_3, base_n_filt * 2, is_training=is_training, kernel=kernel_conv,
                                       is_pad=is_pad)
    conv_0_5 = tf.layers.conv3d(conv_0_4, y.shape[1], [1, 1, 1], [1, 1, 1], padding='same',
                                data_format="channels_first")  # 1 instead of OUTPUT_CLASSES
    return conv_0_5