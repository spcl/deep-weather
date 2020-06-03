import tensorflow as tf

def _parse_(example):
    """
    dummy function
    """
    SHAPEX = (12,7,30,30)
    SHAPEY = (6,7,30,30)
    features = {
            'X': tf.FixedLenFeature(SHAPEX, tf.float32), #FixedLenSequenceFeature, allow_missing=True
            'Y': tf.FixedLenFeature(SHAPEY, tf.float32)
        }
    data = tf.parse_single_example(example, features)
    return data['X'], data['Y']


def TF2FLRD(filenames, batchsize=10, buffersize=100, fetchbatch=2, shuffle=True, parse=_parse_, oneshot=False):
    fetchsize = batchsize*fetchbatch
    train_dataset = tf.data.TFRecordDataset(filenames=filenames)
    train_dataset = train_dataset.prefetch(fetchsize)
    train_dataset = train_dataset.map(parse)
    if shuffle:
        train_dataset = train_dataset.shuffle(buffersize)
    train_dataset = train_dataset.batch(batchsize)
    if oneshot:
        return train_dataset.make_one_shot_iterator()
    else:
        return train_dataset.make_initializable_iterator()
        # return tf.data.Dataset.range(2).interleave(train_dataset, cycle_length=4, num_parallel_calls=4).make_initializable_iterator()