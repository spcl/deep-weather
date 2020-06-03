#!/usr/bin/python
import sys
import tensorflow as tf
import numpy as np
from eccodes import *

from tensorflow.python.platform import flags
import global_macros
import config
import args
from data.dataset import TF2FLRD
from matplotlib import pyplot as plt
import pickle

X_SHAPE = config.X_SHAPE
Y_SHAPE = config.Y_SHAPE
pos_emb_len = config.pos_emb_len
seed = 30
load_path = "/media/chengyuan/wpc/weather_prediction_correction/ckpt/unet3/plot_z500/model_10"


def GetModelAndOptNames():
    if len(sys.argv) < 2:
        print('USAGE: main.py [Model]\n')
        print('Model options: LCN, one2one, unet2, unet3, conv-deconv etc.')
        sys.exit(1)
    modelname = sys.argv[1]
    return modelname


def boxplot():
    modelname = GetModelAndOptNames()
    FLAGS = args.getFlag(modelname)

    file_comment = config.file_comment
    years_train = np.arange(1999, 2014)  # [2014, 2015, 2016]
    years_val = np.arange(2014, 2016) # [2017]
    years_test = np.arange(2016, 2018) # [2018]

    ipaths_train = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_train]
    ipaths_val = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_val]
    ipaths_test = [ (global_macros.TF_DATA_DIRECTORY + "/tf_" + str(i) + "_" + file_comment) for i in years_test]

    mean_path = global_macros.TF_DATA_DIRECTORY + "/mean_" + file_comment + ".npy"
    std_path = global_macros.TF_DATA_DIRECTORY + "/std_" + file_comment + ".npy"

    std = np.load(std_path)

    parsefcn = config.parsefcn

    tf.reset_default_graph()
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    # config_options = config.get_sess_options(mem_frac=FLAGS.mem_frac)

    with tf.Session() as sess:
        model = args.getModel(modelname, FLAGS, sess)
        with tf.name_scope('data'):
            # iter_train = TF2FLRD(ipaths_train, batchsize=FLAGS.batch_size, buffersize=FLAGS.batch_size * 3, parse=parsefcn)
            iter_test = TF2FLRD(ipaths_test, batchsize=FLAGS.batch_size, buffersize=FLAGS.batch_size * 3, shuffle=False, parse=parsefcn)

        loss, refloss, eval = model.get_loss_arrays(iter_data=iter_test, xchannels=X_SHAPE[1])

    types = config.getTypes()
    refloss = np.sqrt(refloss) * std[types[0], 0, 0, 0]
    loss = np.sqrt(loss) * std[types[0], 0, 0, 0]
    np.save("refloss", refloss)
    np.save("loss", loss)
    np.save("diff", loss-refloss)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.set_title('Reference loss')
    ax1.boxplot(refloss, notch=True)
    ax2.set_title('Prediction loss')
    ax2.boxplot(loss, notch=True)
    plt.show()


def plotLoss(fn):
    train_epochL, validate_epochL, train_lossL, validate_lossL = pickle.load(open(fn, 'rb'))
    fig, ax = plt.subplots()
    ax.plot(train_epochL, train_lossL, label='Train')
    ax.plot(validate_epochL, validate_lossL, label='Validate')
    plt.xlabel('Epoch')
    plt.ylabel('Loss((Normalized)')
    plt.title('Training and validation loss')
    ax.legend(loc='upper left', frameon=True)
    # plt.ylim((0, 1.5 * max(max(self.train_lossL), max(self.validate_lossL))))
    plt.show()


def heatmap(data, row_labels=None, col_labels=None, ax=None, cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    # ax.set_xticks(np.arange(data.shape[1]))
    # ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if col_labels is not None:
        ax.set_xticklabels(col_labels)
    if row_labels is not None:
        ax.set_yticklabels(row_labels)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def reorg(x):
    xlim = 720
    ylim = 360
    return x.reshape([ylim, xlim])


def rmse(x,y):
    return np.sqrt(np.mean(np.square(x-y)))


def getmap_overlay():
    Ni = 361
    Nj = 720

    INPUT_FILE = "land_sea_mask.grib"
    rf = open(INPUT_FILE)
    x = codes_count_in_file(rf)
    # print(x)

    gidm = codes_grib_new_from_file(rf)

    # iterid = codes_keys_iterator_new(gidm, 'ls')
    # while codes_keys_iterator_next(iterid):
    #     keyname = codes_keys_iterator_get_name(iterid)
    #     keyval = codes_get_string(gidm, keyname)
    #     print("%s = %s" % (keyname, keyval))
    values = codes_get_values(gidm)
    # print(values.shape)

    overlay = np.full([Ni, Nj], np.nan, dtype=np.float32)
    contour = np.full([Ni, Nj], np.nan, dtype=np.float32)
    overlay[:, :] = np.copy(values.reshape([Ni, Nj]))
    for i in range(Ni):
        for j in range(Nj):
            if i == Ni - 1 or j == Nj - 1:
                contour[i][j] = 0
            else:
                testsum = overlay[i][j] + overlay[i + 1][j] + overlay[i][j + 1] + overlay[i + 1][j + 1]
                if testsum <= 1 or testsum >= 3:
                    contour[i][j] = 0
                else:
                    contour[i][j] = 1

    # plot
    # cmap = 'bwr'
    # cmap = 'binary'
    # f, (ax1) = plt.subplots(1, 1)
    # im1 = ax1.imshow(contour, vmin=0, vmax=1, cmap=cmap)
    # # cbar1 = ax1.figure.colorbar(im1, ax=ax1)
    # # cbar1.ax.set_ylabel("", rotation=-90, va="bottom")
    # plt.show()
    return overlay[:360, :], contour[:360, :]


def generate_row_col_labels():
    row_labels = ["90N", "67.5N", "45N", "22.5N", "0N", "22.5S", "45S", "67.5S", "90S"]
    col_labels = ["0E", "45E", "90E", "135E", "180E", "135W", "90W", "45W"]
    return row_labels, col_labels


def stepquantize(array, steps):
    result = np.copy(array)
    maxval = steps[-1]
    minval = steps[0]
    result = np.clip(result, minval, maxval)
    for i in range (len(steps) - 1):
        np.where(result < steps[i+1], steps[i], result)
    return result


def plotheatmap():
    tempGeo = True
    if tempGeo:
        fn = "sample_temp"
        file_comment = "pl850slice48"
        steps = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    else:
        fn = "sample_geo"
        file_comment = "pl500slice48"
        steps = [-400, -320, -240, -160, -80, 0, 80, 160, 240, 320, 400]
    # std = 1
    std_path = global_macros.TF_DATA_DIRECTORY + "/std_" + file_comment + ".npy"
    mean_path = global_macros.TF_DATA_DIRECTORY + "/mean_" + file_comment + ".npy"
    if tempGeo:
        std = np.load(std_path).flatten()[0]  # temperature
        mean = np.load(mean_path).flatten()[0]
    else:
        std = np.load(std_path).flatten()[6]  # geo potential
        mean = np.load(mean_path).flatten()[6]
    print(std)
    # exit(0)

    vmin = -0.3 * std
    vmax = 0.3 * std
    # cmap = None
    # cmap = 'inferno'
    # cmap = 'seismic'
    cmap = 'bwr'

    x1, y, pred = pickle.load(open(fn, 'rb'))
    x1 = reorg(x1) * std
    y = reorg(y) * std
    pred = reorg(pred) * std

    overlay, contour = getmap_overlay()

    print(np.min(pred-y))
    print(np.max(pred-y))
    print(np.min(x1-y))
    print(np.max(x1-y))
    a = rmse(pred, y)
    b = rmse(x1, y)
    print(a)
    print(b)
    print((b-a)/b)
    #plot error
    f, (ax1, ax2) = plt.subplots(2, 1)

    row_labels, col_labels = generate_row_col_labels()

    # x0 = np.load("plslice_20150101_x0.npy").reshape(7, 361, 720)[0, :360, :]
    x48 = np.load("plslice850_20150103_x0.npy").reshape(7, 361, 720)[0, :360, :]
    # y48 = np.load("plslice_20150101_y48.npy").reshape(7, 361, 720)[0, :360, :]
    print(x48[100:110, 100] - (y+mean)[100:110, 100])
    # print(rmse(x48, y+mean))
    exit(0)

    if tempGeo:
        plt.title("2015.01.01 48hour lead time @ T850")
    else:
        plt.title("2015.01.01 48hour lead time @ Z500")
    im1 = ax1.imshow(y - pred, vmin=vmin, vmax=vmax, cmap=cmap)
    # im1 = ax1.imshow( stepquantize(pred - y, steps), vmin=steps[0], vmax=steps[-1], cmap=cmap)
    ct1 = ax1.imshow(contour, vmin=0, vmax=1, cmap='binary', alpha=0.5)
    im2 = ax2.imshow(y - x1, vmin=vmin, vmax=vmax, cmap=cmap)
    # im2 = ax2.imshow( stepquantize(x1 - y, steps), vmin=steps[0], vmax=steps[-1], cmap=cmap)
    ct2 = ax2.imshow(contour, vmin=0, vmax=1, cmap='binary', alpha=0.5)
    cbar1 = ax1.figure.colorbar(im1, ax=ax1)
    cbar1.ax.set_ylabel("", rotation=-90, va="bottom")
    cbar2 = ax2.figure.colorbar(im2, ax=ax2)
    cbar2.ax.set_ylabel("", rotation=-90, va="bottom")

    # ax1.set_xticklabels(col_labels)
    # ax1.set_yticklabels(row_labels)
    # ax2.set_xticklabels(col_labels)
    # ax2.set_yticklabels(row_labels)

    plt.show()
    # heatmap(y - pred)
    # plt.show()
    # heatmap(x1 - pred)
    # plt.show()
    return


if __name__ == "__main__":
    boxplot()
    # plotheatmap()