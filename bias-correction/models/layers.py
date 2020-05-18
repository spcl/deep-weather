import tensorflow as tf
import numpy as np
from models.utils import variable_summaries


def conv3Dpad(tensor, n_filters, kernel_size, strides=[1, 1, 1], data_format="channels_first", to_pad = True, pad_type="SYMMETRIC"):
    if to_pad:
        padsize = [None]*3
        for i in range(3):
            assert(kernel_size[i]%2 == 1), "conv_local: even filter size not supported"
            padsize[i] = (kernel_size[i] - 1) // 2
        paddings = tf.constant([[0,0], [0,0], [padsize[0], padsize[0]], [padsize[1], padsize[1]], [padsize[2], padsize[2]]])
        paddedtensor = tf.pad(tensor, paddings, pad_type)
    else:
        paddedtensor = tensor
    out = tf.layers.conv3d(paddedtensor, n_filters, kernel_size=kernel_size, strides=strides,
                     padding='VALID', data_format=data_format)
    return out

def tile_conv_layer(tensor, div, n_filters, kernel_size, strides=[1, 1, 1], data_format="channels_first", pad_type="SYMMETRIC"):
    padsize = [None]*3
    for i in range(3):
        assert(kernel_size[i]%2 == 1), "conv_local: even filter size not supported"
        padsize[i] = (kernel_size[i] - 1) // 2
    _, c, l, w, h = tensor.get_shape()
    divx, divy = div
    out_tensors = [[None for indj in range(divy)] for jndi in range(divx)]
    w = int(w)
    h = int(h)
    for ix in range(divx):
        for iy in range(divy):
            lx = max(ix * w//divx - padsize[1], 0)
            hx = min( (ix+1) * w//divx + padsize[1] , w )
            ly = max(iy * h//divy - padsize[2] , 0)
            hy = min( (iy+1) * h//divy + padsize[2], h )
            paddings = tf.constant([[0, 0], [0, 0], [padsize[0], padsize[0]],
                                    [(lx <= 0) * padsize[1], (hx >= w) * padsize[1]],
                                    [(ly <= 0) * padsize[2], (hy >= h) * padsize[2]]])
            localtensor = tf.pad( tensor[:,:,:,lx:hx,ly:hy], paddings, pad_type)
            out_tensors[ix][iy] = conv3Dpad(localtensor, n_filters, kernel_size, strides=strides, data_format=data_format, to_pad=False)
    # print(out_tensors)
    latitude_results = [None for iter in range(divy)]
    for indj in range(divy):
        latitude_results[indj] = tf.concat( [out_tensors[iter][indj] for iter in range(divx)], axis=3)
    # print(latitude_results)
    out = tf.concat(latitude_results, axis=4)
    # print(out.get_shape())
    return out


def conv_tile_batch_relu3d_layer(tensor, div, n_filters, kernel_size, strides=[1, 1, 1], data_format="channels_first", pad_type="SYMMETRIC", is_training=True):
    conv = tile_conv_layer(tensor, div, n_filters, kernel_size, strides=strides, data_format=data_format, pad_type=pad_type)
    conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)  # axis 1 because of channels first
    conv = tf.nn.relu(conv)
    return conv


def R2p1D_layer(tensor, num_filters = (8, 1), filter_size = (3, 3), height_reuse=tf.AUTO_REUSE, pad_type="SYMMETRIC"):
    _, num_channels, height, latitude, longitude = tensor.shape
    input_layer_ = tf.transpose(tensor, perm=[0,2,3,4,1])
    input_layer_ = tf.reshape(input_layer_, shape=[-1, latitude, longitude, num_channels])

    for size, num_filter in zip(filter_size, num_filters):
        with tf.variable_scope(str(size)+'_'+str(num_filter), reuse=tf.AUTO_REUSE):
            # spatial 2D
            layer2D_out = tf.layers.conv2d(inputs=input_layer_,
                                           filters=num_filter,
                                           kernel_size=[size, size],
                                           padding="same",
                                           activation=tf.nn.relu)
            # layer2D_out = tf.layers.max_pooling2d(inputs=layer2D_out, pool_size=[2, 2], strides=2, padding='same')

            # height 1D
            _, latitude, longitude, num_channels = layer2D_out.shape
            layer2D_out = tf.reshape(layer2D_out, [-1, height, latitude*longitude, num_channels])
            layer2D_out = tf.transpose(layer2D_out, perm = [0, 2, 1, 3])
            layer2D_out = tf.reshape(layer2D_out, [-1, height, num_channels])
            paddings = tf.constant([[0, 0], [(size-1)//2, (size-1)//2], [0, 0]])
            layer2D_out = tf.pad(layer2D_out, paddings, pad_type)

            height_result_list = [None for i in range(height)]
            with tf.variable_scope("conv1D", reuse=tf.AUTO_REUSE) as scope:
                for i in range(height):
                    height_result_list[i] = tf.layers.conv1d(inputs=layer2D_out[:, i:i+size, :],
                                                             filters=num_filter,
                                                             kernel_size=size,
                                                             padding="valid", reuse=height_reuse)
                                                    # activation=tf.nn.relu
                layer1D_out = tf.concat(height_result_list, axis=1)
            layer1D_out = tf.reshape(layer1D_out, [-1, latitude*longitude, height, num_channels])
            layer1D_out = tf.transpose(layer1D_out, perm = [0, 2, 1, 3])
            input_layer_ = tf.reshape(layer1D_out, [-1, latitude, longitude, num_channels])
            print(input_layer_.get_shape())

    input_layer_ =  tf.reshape(input_layer_, [-1, height, latitude, longitude, num_channels])
    input_layer_ = tf.transpose(input_layer_, perm=[0, 4, 1, 2, 3])
    print(input_layer_.get_shape())
    return input_layer_


def upconv_op(tensor, filters, deconv=False, kernel=[2, 2, 2], stride=[1, 2, 2], scale=4, activation=None, is_pad=True):
    if deconv:
        out = upconv3d_layer(tensor, filters, kernel=kernel, stride=stride, scale=scale, activation=activation, is_pad=is_pad)
    else:
        out = upsample_conv2d_layer(tensor, filters, kernel=kernel[1:], upsize=stride[1:], scale=scale, activation=activation, is_pad=is_pad)
    return out


def upconv3d_layer(tensor, filters, kernel=[2, 2, 2], stride=[1, 2, 2], scale=4, activation=None, is_pad=True):
    padding = 'valid'
    if is_pad:
        padding = 'same'
    # conv = tf.layers.conv3d(tensor, filters, kernel, stride, padding = 'same',
    #                                 kernel_initializer = base_init, kernel_regularizer = reg_init)
    # use_bias = False is a tensorflow bug
    conv = tf.layers.conv3d_transpose(tensor, filters, kernel_size=kernel, strides=stride, padding=padding,
                                      use_bias=False,
                                      # kernel_initializer=base_init, kernel_regularizer=reg_init,
                                      data_format="channels_first")
    return conv


def upsample_conv2d_layer(tensor, filters, kernel=[3, 3], upsize=[2, 2], scale=4, activation=None, is_pad=True):
    """
    Upsample and do the convolution. This is to replace the deconvolution operation
    2D conv is implemented because the height stride should be 1
    """
    padding = 'valid'
    if is_pad:
        padding = 'same'
    _, num_channels, height, latitude, longitude = tensor.shape
    tensor = tf.transpose(tensor, perm=[0, 2, 3, 4, 1])
    tensor = tf.reshape(tensor, shape=[-1, latitude, longitude, num_channels])
    # upsample
    newlatitude = latitude*upsize[0]
    newlongitude = longitude*upsize[1]
    conv = tf.image.resize_images(tensor, [newlatitude, newlongitude])
    # 2D conv
    conv = tf.layers.conv2d(conv, filters, kernel_size=kernel, strides=[1,1], padding=padding,
                     data_format="channels_last")
    out = tf.reshape(conv, [-1, height, newlatitude, newlongitude, num_channels])
    out = tf.transpose(out, perm=[0, 4, 1, 2, 3])
    return out


def conv_batch_relu3d_layer(tensor, filters, kernel=[3, 3, 3], stride=[1, 1, 1], is_training=True, is_pad=True):
    # Produces the conv_batch_relu3d combination as in the paper
    padding = 'valid'
    if is_pad:
        padding = 'same'
    conv = tf.layers.conv3d(tensor, filters, kernel_size=kernel, strides=stride, padding=padding,
                            # kernel_initializer=base_init, kernel_regularizer=reg_init,
                            data_format="channels_first")  # , data_format = "NCDHW"
    conv = tf.layers.batch_normalization(conv, training=is_training, axis=1)  # axis 1 because of channels first
    conv = tf.nn.relu(conv)
    return conv


def centre_crop_and_concat_layer(prev_conv, up_conv):
    # If concatenating two different sized Tensors, centre crop the first Tensor to the right size and concat
    # Needed if you don't have padding
    p_c_s = prev_conv.get_shape()
    u_c_s = up_conv.get_shape()
    offsets = np.array([0, 0, (p_c_s[2] - u_c_s[2]) // 2, (p_c_s[3] - u_c_s[3]) // 2,
                        (p_c_s[4] - u_c_s[4]) // 2], dtype=np.int32)
    size = np.array([-1, p_c_s[1], u_c_s[2], u_c_s[3], u_c_s[4]], np.int32)
    prev_conv_crop = tf.slice(prev_conv, offsets, size)
    up_concat = tf.concat((prev_conv_crop, up_conv), 1)
    return up_concat


def affine_layer(tensor, no_bias=True):
    _, c, l, _, _ = tensor.get_shape()
    shape = (1, c, l, 1, 1)
    weight= tf.Variable(tf.ones(shape), name = 'weights')
    if no_bias:
        result = tensor * weight
    else:
        bias = tf.Variable(tf.zeros(shape, name = 'bias'))
        result = tensor * weight + bias
    return result


def LCN2D_layer(tensor, channels = 1, kernel=3, namespace ='conv_local2', name='conv_local2_result', with_affine=True):
    # tensor: batch, c, w, h
    with tf.name_scope(namespace):
        assert(kernel%2 == 1), "conv_local: even filter size not supported"
        padsize = (kernel - 1) // 2
        _, c, w, h = tensor.get_shape()
        oc = channels

    paddings = tf.constant([ [0,0], [0,0], [padsize, padsize], [padsize, padsize] ])
    paddedtensor = tf.pad(tensor, paddings, "SYMMETRIC")

    weight_shape = [c, w, h, oc, kernel, kernel]
    bias_shape = [w, h, oc]
    #weights = tf.get_variable('x', shape=[2, 4], trainable = True, initializer=tf.constant_initializer(1/(c*kernel*kernel)))
    weights = tf.Variable(tf.zeros(weight_shape, dtype=tf.float32), name = 'conv_local2_weights')
    bias = tf.Variable(tf.zeros(bias_shape, dtype=tf.float32), name = 'conv_local2_bias')

    result = [None] * oc
    for k in range(oc):
        for i in range(kernel):
            for j in range(kernel):
                tensor_crop = paddedtensor[:, :, i:i+w, j:j+h]
                weight_filter = weights[:,:,:,k,i,j]
                lcn = tf.reduce_sum(weight_filter * tensor_crop, axis=1)
                if i==0 and j==0:
                    result[k] = lcn + bias[:,:,k]
                else:
                    result[k] = result[k] + lcn
    with tf.name_scope('summary'):
        variable_summaries(weights, name = 'weights')
        variable_summaries(bias, name = 'bias')
    oc_result = tf.concat([tf.reshape(t, [-1, 1, w, h]) for t in result], axis = 1)
    if with_affine:
        oc_result = affine_layer(oc_result)
    return tf.identity(oc_result, name=name)


def LCN3D_layer(tensor, channels = 1, kernel=(1, 3, 3), namespace ='conv_local3', name='conv_local3_result', regularize=False, alpha=0.01, with_affine=True):
    assert(channels==1), "For now only support 1 output channel"
    # tensor: batch, c, l, w, h
    with tf.name_scope(namespace):
        padsize = [None]*3
        for i in range(3):
            assert(kernel[i]%2 == 1), "conv_local: even filter size not supported"
            padsize[i] = (kernel[i] - 1) // 2
        _, c, l, w, h = tensor.get_shape()
        oc = channels

    paddings = tf.constant([ [0,0], [0,0], [padsize[0], padsize[0]], [padsize[1], padsize[1]], [padsize[2], padsize[2]] ])
    paddedtensor = tf.pad(tensor, paddings, "SYMMETRIC")

    weight_shape = [c, l, w, h, oc]
    bias_shape = [l, w, h, oc]
    weights = [ [[None]*kernel[0] for i in range(kernel[1])] for j in range(kernel[2])]
    for i in range (kernel[1]):
        for j in range (kernel[2]):
            for p in range(kernel[0]):
                new_weights = tf.Variable(tf.zeros(weight_shape, dtype=tf.float32), name = 'conv_local3_weights')
                # init_weight_val = tf.zeros(weight_shape,dtype=tf.float32)
                # new_weights = tf.get_variable("conv_local3_weights", shape=weight_shape, dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
                weights[i][j][p] = new_weights
    bias = tf.Variable(tf.zeros(bias_shape, dtype=tf.float32), name = 'conv_local3_bias')

    result = [None] * oc
    for k in range (oc):
        for i in range (kernel[1]):
            for j in range (kernel[2]):
                for p in range(kernel[0]):
                    tensor_crop = paddedtensor[:, :, p:p+l, i:i+w, j:j+h]
                    weight_filter = weights[i][j][p][:, :, :, :, k]
                    lcn = tf.reduce_sum(weight_filter * tensor_crop, axis=1)
                    if i==0 and j==0 and p==0:
                        result[k] = lcn + bias[:, :, :, k]
                    else:
                        result[k] = result[k] + lcn
    with tf.name_scope('summary'):
        variable_summaries(weights, name = 'weights')
        variable_summaries(bias, name = 'bias')
    oc_result = tf.stack([tf.reshape(t, [-1, l, w, h]) for t in result], axis=1)


    # regularization term calculation
    reg = 0
    if regularize:
        # reg1, reg2, reg3 = 0, 0, 0
        # # along longitude dimension
        # for p in range(kernel[0]):
        #     for i in range(kernel[1]):
        #         for j in range(kernel[2] - 1):
        #             regval = tf.reduce_mean(tf.square(weights[i][j+1][p] - weights[i][j][p]))
        #             if p==0 and i==0 and j==0:
        #                 reg1 = regval
        #             else:
        #                 reg1 = regval + reg1
        #
        # # along latitude dimension
        # for p in range(kernel[0]):
        #     for i in range(kernel[1] - 1):
        #         for j in range(kernel[2]):
        #             regval = tf.reduce_mean(tf.square(weights[i+1][j][p] - weights[i][j][p]))
        #             if p==0 and i==0 and j==0:
        #                 reg2 = regval
        #             else:
        #                 reg2 = regval + reg2
        #
        # # along height dimension
        # for p in range(kernel[0] - 1):
        #     for i in range(kernel[1]):
        #         for j in range(kernel[2]):
        #             regval = tf.reduce_mean(tf.square(weights[i][j][p+1] - weights[i][j][p]))
        #             if p==0 and i==0 and j==0:
        #                 reg3 = regval
        #             else:
        #                 reg3 = regval + reg3
        #
        # a, b, c = kernel
        # nterms = a*b*(c-1) + b*c*(a-1) + c*a*(b-1) # normalized by the number of terms
        # reg = alpha*(reg1+reg2+reg3)/nterms
        reg1 = 0
        reg2 = 0
        for p in range(kernel[0]):
            for i in range(kernel[1]):
                for j in range(kernel[2]):
                    reg1 += tf.losses.absolute_difference(weights[i][j][p], tf.roll(weights[i][j][p], shift=1, axis=2))
                    reg2 += tf.losses.absolute_difference(weights[i][j][p], tf.roll(weights[i][j][p], shift=1, axis=3))
        reg_bias1 = tf.losses.absolute_difference(bias, tf.roll(bias, shift=1, axis=1))
        reg_bias2 = tf.losses.absolute_difference(bias, tf.roll(bias, shift=1, axis=2))
        reg = (reg1 + reg2 + reg_bias1 + reg_bias2) / (2 * kernel[0] * kernel[1] * kernel[2] + 2)

    if with_affine:
        oc_result = affine_layer(oc_result)

    return tf.identity(oc_result, name=name), reg
