import tensorflow as tf
import global_macros
from tensorflow.python.platform import flags
import datetime
import config

ROOT_DIRECTORY = global_macros.ROOT_DIRECTORY
file_comment = config.file_comment

def getFlag(model_name):
    use_time = False
    if use_time:
        exp_name = datetime.datetime.now().strftime("%I:%M%p-%Y-%B-%d")
    else:
        exp_name = 'test'

    FLAGS = flags.FLAGS
    # Dataset Options:
    flags.DEFINE_integer('batch_size', 8, 'Size of a batch')

    # Base Model class Mandatory:
    flags.DEFINE_bool('train', True, 'whether to train or test')
    flags.DEFINE_bool('verbose', True, 'whether to show print information or not')
    flags.DEFINE_integer('epoch', 30, 'Number of Epochs to train on')
    flags.DEFINE_string('exp', exp_name, 'name of experiments')
    flags.DEFINE_integer('log_interval', 1, 'log outputs every so many epoch')
    flags.DEFINE_integer('val_interval', 3, 'validate every so many epoch')
    flags.DEFINE_integer('patience', 3, 'number of non-improving validation iterations before early stop')
    flags.DEFINE_integer('save_interval', 10,'save outputs every so many iterations')
    ## Saver load or options:
    flags.DEFINE_integer('max_to_keep', 10, 'maximum number of models to keep')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 3, 'check point intervals')
    flags.DEFINE_integer('resume_iter', -1,
    'iteration to resume training from, -1 means not resuming')
    flags.DEFINE_string('ckptdir', global_macros.CKPT_ROOT + "/" + model_name,
    'location where models will be stored')
    flags.DEFINE_string('logdir', global_macros.LOGGER_ROOT + "/" + model_name,
    'location where log of experiments will be stored')
    ## Plot option:
    flags.DEFINE_bool('plot', True, 'plot after training')
    flags.DEFINE_bool('crop', False, 'crop regions')
    flags.DEFINE_bool('crop_stack', True, 'crop stack/ random crop')

    # learning rate
    flags.DEFINE_bool('L1_loss', False, 'Use L1 or L2 loss')
    flags.DEFINE_bool('weight_decay', False, 'Turn on weight decay or not')
    flags.DEFINE_float('lr', 1e-4, 'Learning rate for training')
    flags.DEFINE_float('lr_decay_val', 10, 'Learning rate decay ratio')
    flags.DEFINE_bool('recompute', False, 'use recomputation')

    # Model specific:
    flags.DEFINE_bool('temp_only', False, 'only use temperature channel or not')
    flags.DEFINE_bool('ssim', False, 'use ssim loss or not')

    # Unet specific:
    flags.DEFINE_bool('is_pad', True, 'Use padding for convolution or not')
    flags.DEFINE_integer('nfilters', 8, 'The number of base filters for unet')
    flags.DEFINE_integer('unet_levels', 3, 'Levels of Unet')
    flags.DEFINE_bool('img_emb', False, 'Use image embedding or not')

    # LCN specific:
    flags.DEFINE_list('lcn_kernel', [1,3,3] , 'Kernel list for lcn model')
    flags.DEFINE_bool('regularize', False, 'Turn on regularizer for LCN')
    flags.DEFINE_float('alpha', 1e5, 'Regularizer value')

    # tile conv LCN
    flags.DEFINE_bool('use_LCN', False, 'use LCN as the last layer, tile conv LCN only')
    return FLAGS


def getModel(modelname, FLAGS, sess):
    from models.one2one import one2one
    from models.refmse import refmse
    from models.Unet3 import Unet3
    from models.Unet2 import Unet2
    from models.Unet2_l2 import Unet2_l2
    from models.conv_deconv import ConvDeconv
    from models.resnet import ResNet
    from models.LCN import LCN
    from models.simple_conv import LayerConv
    from models.tile_CNN import TileCNN
    from models.inception import Inception
    from models.LCN_unet import LCN_unet
    from models.model1 import model1
    from models.model2 import model2
    from models.dateModel import dateModel
    from models.Unet3_local import Unet3_local
    from models.Unet3_tile import Unet3_tile
    model_dic = {
        'one2one': one2one,
        'refmse': refmse,
        'unet3': Unet3,
        'unet2': Unet2,
        'unet2_l2': Unet2_l2,
        'conv_deconv': ConvDeconv,
        'ResNet': ResNet,
        'LCN': LCN,
        'LayerConv': LayerConv,
        'TileCNN': TileCNN,
        'inception': Inception,
        'LCN_unet': LCN_unet,
        'model1': model1,
        'model2': model2,
        'dateModel': dateModel,
        'unet3_local': Unet3_local,
        'unet3_tile': Unet3_tile
    }
    if modelname in model_dic:
        model = model_dic[modelname](sess, FLAGS, file_comment)
    else:
        raise NameError('Unrecognized model "%s"' % modelname)
    return model


def print_flag(FLAGS):
    for key in FLAGS.__flags.keys():
        print(' {:<20}: {}'.format(key, getattr(FLAGS, key)))


# def get_sess_options(mem_frac):
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
#     return tf.ConfigProto(gpu_options=gpu_options)
