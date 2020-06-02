import tensorflow as tf

# the only high level variable is here:
tempGeo = True  # if true, perform temperature at 850hPa; if false, perform geo at 500hPa
norm = False

if tempGeo:
    file_comment = "pl850slice48"
else:
    file_comment = "pl500slice48"
if norm:
    file_comment = file_comment + '_nor'
# file_comment = "pl"

pos_emb_len = 8*2
if file_comment == "pl_pos_24_small":
    pos_emb_len = 8
    pos_emb_SHAPE = (None, 8)
else:
    pos_emb_len = None
    pos_emb_SHAPE = None

if file_comment == "plslice" or "pl500slice48" in file_comment or "pl850slice48" in file_comment:
    X_SHAPE = (None, 14, 1, 361, 720)
    Y_SHAPE = (None, 7, 1, 361, 720)
elif file_comment == "pl":
    X_SHAPE = (None, 14, 11, 361, 720)
    Y_SHAPE = (None, 7, 11, 361, 720)


def parse_normal(example):
    """
    Normally parse X, Y data pair
    """
    features = {
            'Date': tf.FixedLenFeature((1, ), tf.int64),
            'X': tf.FixedLenFeature(X_SHAPE[1:], tf.float32),
            'Y': tf.FixedLenFeature(Y_SHAPE[1:], tf.float32)
        }
    data = tf.parse_single_example(example, features)
    return data


def parse_pos(example):
    """
    Include pos_emb in the parsing
    """
    features = {
        'Date': tf.FixedLenFeature((1,), tf.int64),
        'X': tf.FixedLenFeature(X_SHAPE[1:], tf.float32),
        'Y': tf.FixedLenFeature(Y_SHAPE[1:], tf.float32),
        'pos_emb': tf.FixedLenFeature(pos_emb_SHAPE[1:], tf.float32)
    }
    data = tf.parse_single_example(example, features)
    return data

parsefcn = parse_normal
if file_comment == "pl_pos_24" or file_comment == "pl_pos_24_small":
    parsefcn = parse_pos

def getTypes():
    if tempGeo:
        return [0, 1]
    else:
        return [6, 7]

def getHeights():
    singleHeight = True
    allHeight = False
    Is850or500 = tempGeo
    if singleHeight:
        return [0, 1]
    if allHeight:
        return [0, 11]
    elif Is850or500:
        return [8, 9]
    else:
        return [6, 7]