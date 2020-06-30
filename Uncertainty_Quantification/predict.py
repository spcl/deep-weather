import tensorflow as tf
import parameters
import numpy as np
import RESNET2D
from tqdm import tqdm

def TFRparser(example, intime = parameters.NR_TIMESTEPS, InputLength=parameters.INPUT_SIZEL, InputWidth=parameters.INPUT_SIZEW,
              InputDepth=parameters.INPUT_DEPTH, OutputLength=parameters.OUTPUT_SIZEL,
              OutputWidth=parameters.OUTPUT_SIZEW, OutputDepth=parameters.OUTPUT_DEPTH, NrChannels=parameters.NR_CHANNELS):
    features = {
            'Array': tf.FixedLenFeature([intime,NrChannels, InputDepth, InputWidth, InputLength], tf.float32),
            'ArrayC': tf.FixedLenFeature([2,7, OutputDepth, OutputWidth, OutputLength], tf.float32)
    }
    parsedf = tf.parse_single_example(example, features)
    return parsedf['Array'], parsedf['ArrayC']


def TFRecord2FLRD(filenames, buffersize=210, batchsize=1):
    train_dataset = tf.data.TFRecordDataset(filenames=filenames)
    train_dataset = train_dataset.map(TFRparser)
    #train_dataset = tf.data.FixedLengthRecordDataset.from_tensors(train_dataset)
    #train_dataset = train_dataset.shuffle(buffersize)#buffersize #No shuffle for test data
    train_dataset = train_dataset.batch(batchsize)
    return train_dataset.make_initializable_iterator()

DATE = parameters.DATE_TO_PREDICT


ditert = TFRecord2FLRD(filenames=parameters.PATHTE+'DATA'+str(DATE))
xdata, ydata = ditert.get_next()

npx = np.empty([105, 1, 1, parameters.OUTPUT_SIZEW, parameters.OUTPUT_SIZEL])
rnet = RESNET2D.Model(should_pad=True)
init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto(
    device_count={'GPU': 1})  # XLA_GPU is experimental, might get errors, only ~10% better performance on ResNet50

with tf.Session(config=config) as sess:
    print('Loading saved model...')
    print('Loading from: ', parameters.SAVE_PATH + parameters.MODEL_NAME + '.meta')
    restorer = tf.train.Saver()
    restorer.restore(sess, tf.train.latest_checkpoint(parameters.SAVE_PATH))
    print("Model sucessfully loaded")
    sess.run(ditert.initializer)

    it = 0
    try:
        with tqdm(desc='Hours', leave=False) as pbar:
            while True:
                x, y = sess.run([xdata, ydata])
                # Initialize iterator with testing data
                train_dict = {
                    rnet.training: False,
                    rnet.model_input: x,  # xdata,
                    rnet.model_cout: y  # ydata
                }
                pred = sess.run(rnet.predictions, feed_dict=train_dict)
                npx[it,:,:,:,:] = pred
                it += 1
                pbar.update()
    except tf.errors.OutOfRangeError:
        pass
    npx = npx[:,0,0,:,:]
    np.save(parameters.NPY_DATA_DIRECTORY+'/predictions'+str(DATE), npx)

