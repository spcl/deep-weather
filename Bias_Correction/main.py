#!/usr/bin/python
import sys
import tensorflow as tf
import numpy as np
import pickle
import os.path as osp
from tensorflow.python.platform import flags
import global_macros
import args
import config
from data.dataset import TF2FLRD


X_SHAPE = config.X_SHAPE
Y_SHAPE = config.Y_SHAPE
pos_emb_len = config.pos_emb_len
seed = 30


def GetModelAndOptNames():
    if len(sys.argv) < 2:
        print('USAGE: main.py [Model]\n')
        print('Model options: LCN, one2one, unet2, unet3, conv-deconv etc.')
        sys.exit(1)
    modelname = sys.argv[1]
    return modelname


def main(predict=False):

    modelname = GetModelAndOptNames()
    FLAGS = args.getFlag(modelname)

    file_comment = config.file_comment
    years_train = np.arange(1998, 2014)
    years_val = np.arange(2014, 2016)
    years_test = np.arange(2016, 2018)

    ipaths_train = [(global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_train]
    ipaths_val = [(global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_val]
    ipaths_test = [(global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_test]

    mean_path = global_macros.TF_DATA_DIRECTORY + "/mean_" + file_comment + ".npy"
    std_path = global_macros.TF_DATA_DIRECTORY + "/std_" + file_comment + ".npy"

    parsefcn = config.parsefcn

    tf.reset_default_graph()
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    # config_options = config.get_sess_options()

    with tf.Session() as sess:
        model = args.getModel(modelname, FLAGS, sess)
        # train
        with tf.name_scope('data'):
            iter_train = TF2FLRD(ipaths_train, batchsize=FLAGS.batch_size, buffersize=FLAGS.batch_size * 3, parse=parsefcn)
            iter_val = TF2FLRD(ipaths_val, batchsize=FLAGS.batch_size, buffersize=FLAGS.batch_size * 3, shuffle=False, parse=parsefcn)
            iter_test = TF2FLRD(ipaths_test, batchsize=FLAGS.batch_size, buffersize=FLAGS.batch_size * 3, shuffle=False, parse=parsefcn)

        if predict:
            model.get_predictions(iter_test, file_comment)
        else:
            model.run(iter_data=iter_train, iter_val=iter_val)
        # test
        # if yearly_test:
        #     testLossL = []
        #     for testyear in ipaths_test:
        #         print("Test year: ", testyear)
        #         iter_test = TF2FLRD(testyear, batchsize=FLAGS.batch_size, buffersize=FLAGS.batch_size * 3, shuffle=False,
        #                             parse=parsefcn)
        #         loss = model.run(iter_data=iter_test, train=False, load=False)
        #         testLossL.append(loss)
        #     print("The loss array is: ", testLossL)
        # else:
        #     iter_test = TF2FLRD(ipaths_test, batchsize=FLAGS.batch_size, buffersize=FLAGS.batch_size * 3, shuffle=False,
        #                         parse=parsefcn)
        #     loss = model.run(iter_data=iter_test, train=False, load=False)
        #     print("The test loss is: ", loss)


def plotHeatmap():
    modelname = GetModelAndOptNames()
    FLAGS = args.getFlag(modelname)

    file_comment = config.file_comment
    years_train = np.arange(1998, 2014)
    years_val = np.arange(2014, 2016)
    years_test = np.arange(2016, 2018)

    ipaths_train = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_train]
    ipaths_val = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_val]
    ipaths_test = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_test]

    mean_path = global_macros.TF_DATA_DIRECTORY + "/mean_" + file_comment + ".npy"
    std_path = global_macros.TF_DATA_DIRECTORY + "/std_" + file_comment + ".npy"

    mean = np.load(mean_path)
    std = np.load(std_path)

    parsefcn = config.parsefcn

    tf.reset_default_graph()
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    # config_options = args.get_sess_options(mem_frac=FLAGS.mem_frac)

    with tf.Session() as sess:
        model = args.getModel(modelname, FLAGS, sess)
        # train
        with tf.name_scope('data'):
            iter_train = TF2FLRD(ipaths_train, batchsize=FLAGS.batch_size, buffersize=FLAGS.batch_size * 3, parse=parsefcn)
            iter_val = TF2FLRD(ipaths_val, batchsize=FLAGS.batch_size, buffersize=FLAGS.batch_size * 3, shuffle=False, parse=parsefcn)
            iter_test = TF2FLRD(ipaths_test, batchsize=FLAGS.batch_size, buffersize=FLAGS.batch_size * 3, shuffle=False, parse=parsefcn)
        x1, y, pred, date = model.get_one_predict(iter_data=iter_test)

        types = config.getTypes()
        x1 = x1 * std[types[0], 0, 0, 0] + mean[types[0], 0, 0, 0]
        y = y * std[types[0], 0, 0, 0] + mean[types[0], 0, 0, 0]
        pred = pred * std[types[0], 0, 0, 0] + mean[types[0], 0, 0, 0]

        sample = [x1, y, pred]
        dumpdir = osp.join(model.FLAGS.logdir, model.FLAGS.exp)
        with open(osp.join(dumpdir, "sample" + date), 'wb') as f:
            pickle.dump(sample, f)
    return


if __name__ == "__main__":
    plot = 0
    if plot:
        plotHeatmap()
    else:
        main(predict=True)
