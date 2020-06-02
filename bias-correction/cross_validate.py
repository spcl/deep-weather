#!/usr/bin/python
import sys
import tensorflow as tf
import numpy as np

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


def getFlags(modelname):
    return args.getFlag(modelname)


def cross_validate(modelname, FLAGS):
    Nfold = 1
    years = np.arange(1999, 2018)
    years_train_list = [np.arange(1999, 2014)]
    years_val_list = [np.arange(2014, 2016)]  # 2014, 2015
    years_test_list = [np.arange(2016, 2018)]  # 2016, 2017

    ref_loss_list = []
    loss_list = []
    improve_list = []
    val_loss_list = []

    parsefcn = config.parsefcn

    for i in range(Nfold):
        print("Starting {} fold: ".format(i + 1))
        file_comment = config.file_comment
        years_train = years_train_list[i]
        years_val = years_val_list[i]
        years_test = years_test_list[i]

        ipaths_train = [(global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_train]
        ipaths_val = [(global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_val]
        ipaths_test = [(global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_test]

        tf.reset_default_graph()
        np.random.seed(seed)
        tf.random.set_random_seed(seed)

        with tf.Session() as sessref:
            with tf.name_scope('data'):
                iter_test = TF2FLRD(ipaths_test, batchsize=FLAGS.batch_size,
                                        buffersize=FLAGS.batch_size * 3, shuffle=False, parse=parsefcn)
            refmse = args.getModel("refmse", FLAGS, sessref)
            ref_loss = refmse.run(iter_data=iter_test, train=False, load=False)

        tf.reset_default_graph()
        np.random.seed(seed)
        tf.random.set_random_seed(seed)
        with tf.Session() as sess:
            with tf.name_scope('data'):
                iter_train = TF2FLRD(ipaths_train, batchsize=FLAGS.batch_size,
                                        buffersize=FLAGS.batch_size * 3, parse=parsefcn)
                iter_val = TF2FLRD(ipaths_val, batchsize=FLAGS.batch_size,
                                        buffersize=FLAGS.batch_size * 3, shuffle=False, parse=parsefcn)
                iter_test = TF2FLRD(ipaths_test, batchsize=FLAGS.batch_size,
                                        buffersize=FLAGS.batch_size * 3, shuffle=False, parse=parsefcn)
            model = args.getModel(modelname, FLAGS, sess)
            val_loss = model.run(iter_data=iter_train, iter_val=iter_val)
            loss = model.run(iter_data=iter_test, train=False, load=False)

        improve = (ref_loss - loss) / ref_loss
        val_loss_list.append(val_loss)
        ref_loss_list.append(ref_loss)
        loss_list.append(loss)
        improve_list.append(improve)

    if not FLAGS.L1_loss:
        print("Reference RMSE: ", ref_loss_list)
        print("Model vval RMSE: ", val_loss_list)
        print("Model test RMSE: ", loss_list)
        # rms_ref_list = np.sqrt(np.array(ref_loss_list))
        # rms_list = np.sqrt(np.array(loss_list))
        # rms_improve_list = (rms_ref_list - rms_list)/rms_ref_list
        improve_rmse_list = []
        for i in range(len(ref_loss_list)):
            improve_rmse_list.append(ref_loss_list[i] - loss_list[i])
        print("RMSE improvement value:", improve_rmse_list)
        print("RMSE improvement percentage list:", improve_list)
        # print("RMSE improvement list:", rms_improve_list)
        avg_improve = np.mean(improve_list) * 100
        print("The average RMSE improvement percentage is {:.4f}%".format(avg_improve))
        # print( "The average RMSE improvement is {:.4f}%".format( np.mean(rms_improve_list) * 100 ) )
    else:
        print("Reference L1 loss: ", ref_loss_list)
        print("Model L1 loss: ", loss_list)
        print("L1 improvement value:", improve)
        improve_L1_list = []
        for i in range(len(ref_loss_list)):
            improve_L1_list.append(ref_loss_list[i] - loss_list[i])
        avg_improve = np.mean(improve_list) * 100
        print("The average RMSE improvement percentage is {:.4f}%".format(avg_improve))
    return avg_improve


def main():
    modelname = GetModelAndOptNames()
    FLAGS = getFlags(modelname)
    args.print_flag(FLAGS)
    cross_validate(modelname, FLAGS)

if __name__ == "__main__":
    main()
