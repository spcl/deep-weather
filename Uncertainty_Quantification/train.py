#!/usr/bin/env python3

import tensorflow as tf
import parameters
import RESNET2D
import numpy as np
import os
from tqdm import tqdm, trange #progressbar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


tf.reset_default_graph()
def TFRparser(example, intime = parameters.NR_TIMESTEPS, InputLength=parameters.INPUT_SIZEL, InputWidth=parameters.INPUT_SIZEW,
              InputDepth=parameters.INPUT_DEPTH, OutputLength=parameters.OUTPUT_SIZEL,
              OutputWidth=parameters.OUTPUT_SIZEW, OutputDepth=parameters.OUTPUT_DEPTH, NrChannels=parameters.NR_CHANNELS):
    if parameters.DO_CRPS:
        features = {
                'Array': tf.FixedLenFeature([intime, NrChannels, InputDepth, InputWidth, InputLength], tf.float32),
                'ArrayC': tf.FixedLenFeature([1, OutputWidth, OutputLength], tf.float32)
        }
    else:
        features = {
                'Array': tf.FixedLenFeature([intime,NrChannels, InputDepth, InputWidth, InputLength], tf.float32),
                'ArrayC': tf.FixedLenFeature([2,7, OutputDepth, OutputWidth, OutputLength], tf.float32)
        }
    parsedf = tf.parse_single_example(example, features)
    return parsedf['Array'], parsedf['ArrayC']

# TensorFlow preprocessing when loading data
def TFRecord2FLRD(filenames, buffersize=210, batchsize=parameters.BATCH_SIZE): # shuffles 2 years' worth of entries when loading (1 year has 105 entries)
    train_dataset = tf.data.TFRecordDataset(filenames=filenames)
    train_dataset = train_dataset.map(TFRparser)
    train_dataset = train_dataset.shuffle(buffersize)
    train_dataset = train_dataset.prefetch(buffersize*5) #15 was too much
    train_dataset = train_dataset.batch(batchsize)
    return train_dataset.make_initializable_iterator()



ditert = TFRecord2FLRD(filenames=parameters.FILE_NAMES)
diterv = TFRecord2FLRD(filenames=parameters.FILE_NAMESV)
xdata, ydata = ditert.get_next()  # here xdata is the input, and ydata what it will be compared to
xdatav, ydatav = diterv.get_next()



rnet = RESNET2D.Model(should_pad=True)
init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(
    device_count={'GPU': 2})  # XLA_GPU is experimental, might get errors, only ~10% better performance on ResNet50

with tf.Session(config=config) as sess:
    if parameters.LOAD_MODEL:
        print('Trying to load saved model...')
        try:
            print('Loading from: ', parameters.SAVE_PATH + parameters.MODEL_NAME + '.meta')
            restorer = tf.train.Saver()
            restorer.restore(sess, tf.train.latest_checkpoint(parameters.SAVE_PATH))
            print("Model successfully restored")
        except IOError:
            sess.run(init)
            print("No previous model found, running default initialization")
    else:
        sess.run(init)
    TBwriter = tf.summary.FileWriter(parameters.LOGS_PATH, sess.graph)
    bestvalloss = float('inf')
    trloss = tf.summary.scalar('crps_training_loss', rnet.loss) if parameters.DO_CRPS else tf.summary.scalar('mse_training_loss', rnet.loss)
    ltrloss = tf.summary.scalar('lInf_training_loss', rnet.linf_loss)
    valoss = tf.summary.scalar('crps_validation_loss', rnet.loss) if parameters.DO_CRPS else tf.summary.scalar('mse_validation_loss', rnet.loss)
    lvaloss = tf.summary.scalar('lInf_validation_loss', rnet.linf_loss)
    v_loss = np.empty(parameters.EPOCHS)
    patience_c = 0
    for epoch_no in trange(parameters.EPOCHS, desc='Epochs', position=0):
        # loss values initialization
        train_loss = 0
        val_loss = 0
        tr_loss = 0
        va_loss = 0
        sess.run(ditert.initializer)
        itt = 0
        # Training
        try:
            with tqdm(desc='Batches', leave=False) as pbar:
                while True:
                    x, y = sess.run([xdata, ydata])
                    # Initialize iterator with training data
                    train_dict = {
                        rnet.training: True,
                        rnet.model_input: x,  # xdata,
                        rnet.model_cout: y  # ydata
                    }
                    if itt == 0:
                        _, loss, traloss, ltraloss = sess.run([rnet.train_op, rnet.loss, trloss, ltrloss], feed_dict=train_dict)
                        TBwriter.add_summary(traloss, (epoch_no+1))
                        TBwriter.add_summary(ltraloss, (epoch_no+1))
                    else:
                        _, loss = sess.run([rnet.train_op, rnet.loss], feed_dict=train_dict)
                    train_loss += loss
                    itt += 1
                    pbar.set_postfix(Loss=loss)
                    pbar.update()
        except tf.errors.OutOfRangeError:
            pass
        tr_loss = loss
        # Validation
        sess.run(diterv.initializer)
        itc = 0
        try:
            while True:
                xv, yv = sess.run([xdatav, ydatav])
                # Initialize iterator with validation data
                train_dict = {
                    rnet.training: False,
                    rnet.model_input: xv,  # xdatav,
                    rnet.model_cout: yv  # ydatav
                }
                if itc == 0:
                    loss, vall, lvall = sess.run([rnet.loss, valoss, lvaloss], feed_dict=train_dict)
                    TBwriter.add_summary(vall, (epoch_no+1))
                    TBwriter.add_summary(lvall, (epoch_no+1))
                else:
                    loss = sess.run(rnet.loss, feed_dict=train_dict)
                val_loss += loss
                itc += 1
        except tf.errors.OutOfRangeError:
            pass
        va_loss = loss
        # Calculate and output metrics
        tott_loss = train_loss / itt  # average training loss in 1 epoch
        totv_loss = val_loss / itc  # average validation loss in 1 epoch
        v_loss[epoch_no] = totv_loss  # average val loss saved for early stopping
        print('\nEpoch No: {}'.format(epoch_no + 1))
        print('Train loss = {:.8f}'.format(tott_loss))
        print('Val loss = {:.8f}'.format(totv_loss))

        # Early stopping on validation loss
        if bestvalloss > v_loss[epoch_no]:
            print('Saving model at epoch: ', (epoch_no + 1))
            saver.save(sess, parameters.SAVE_PATH + parameters.MODEL_NAME)
            bestvalloss = v_loss[epoch_no]
            patience_c = 0
        else:
            patience_c += 1
        if patience_c > parameters.PATIENCE:
            print("early stopping...")
            break
        if (epoch_no+1) % parameters.PSAVE == 0 and epoch_no > 0: # Save Model periodically
            print('Saving model at epoch: ', (epoch_no + 1))
            saver.save(sess, parameters.SAVE_PATH + parameters.MODEL_NAME, global_step=(epoch_no + 1))

    saver.save(sess, parameters.SAVE_PATH + parameters.MODEL_NAME, global_step=(
                epoch_no + 1)) # Final save

