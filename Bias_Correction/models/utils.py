import tensorflow as tf
import numpy as np
import os.path as osp
import os
import global_macros
import time
import matplotlib.pyplot as plt
import pickle
from models.memory_saving_gradients import gradients
from config import getTypes, getHeights


def getLossSlice(tensor):
    """
    Slice the tensor to match the loss
    """
    types = getTypes()
    heights = getHeights()
    return tensor[:, types[0]:types[1], heights[0]:heights[1], :, :]


def getLoss(y, pred, L1loss=False, single_layer=False, ssim=False):
    """
    Loss function is defined here
    """
    if ssim:
        return getSSIM(y, pred, single_layer)
    types = getTypes()
    heights = getHeights()
    # return tf.losses.mean_squared_error(y, pred)
    if L1loss:
        lossfcn = tf.losses.absolute_difference
    else:
        lossfcn = tf.losses.mean_squared_error
    if single_layer:
        return lossfcn(y[:, types[0]:types[1], heights[0]:heights[1], :, :], pred)
    else:
        return lossfcn(y[:, types[0]:types[1], heights[0]:heights[1], :, :],
                   pred[:, types[0]:types[1], heights[0]:heights[1], :, :])
    # return tf.losses.mean_squared_error(y[:,3,5,:,:], pred[:,0,0,:,:])
    # return tf.sqrt(tf.reduce_mean(tf.square( y[:,types[0]:types[1],:] - pred[:,types[0]:types[1],:] )))


def getSSIM(y, pred, single_layer=False):
    types = getTypes()
    heights = getHeights()
    assert(heights[1] - heights[0] == 1), "SSIM only support single layer output"
    yin = tf.transpose(y[:, types[0]:types[1], heights[0], :, :], perm=[0, 2, 3, 1]) # map to channel last
    if single_layer:
        predin = tf.transpose(pred[:, :, 0, :, :], perm = [0, 2, 3, 1])
    else:
        predin = tf.transpose(pred[:, types[0]:types[1], heights[0], :, :], perm=[0, 2, 3, 1])
    return -tf.reduce_mean(tf.image.ssim(yin, predin, max_val = 5))
        #power_factors=_MSSSIM_WEIGHTS,
        # filter_size=9, filter_sigma=1.5,
        # k1=0.01, k2=0.03)


def getCrop(y):
    types = getTypes()
    heights = getHeights()
    return y[:, types[0]:types[1], heights[0]:heights[1], :, :]


def getEvaluate(y, pred, file_comment, single_layer=False, height_seperate=False):
    """
    Get the RMS for the denormalized data
    """
    types = getTypes()
    heights = getHeights()
    std_path = global_macros.TF_DATA_DIRECTORY + "/std_" + file_comment + ".npy"
    std = tf.constant(np.expand_dims(np.load(std_path), axis=0), dtype=tf.float32)[:, types[0]:types[1],
          heights[0]:heights[1], :, :]
    if "_nor" in file_comment:
        std = std[:, :, :, :360, :]
    if single_layer:
        assert(heights[1] - heights[0] == 1), "Only single height level is supported"
        result = tf.square((y[:, types[0]:types[1], heights[0]:heights[1], :360, :] - pred[:, :, :, :360, :]) * std)
    else:
        result = tf.square((y[:, types[0]:types[1], heights[0]:heights[1], :360, :] - pred[:, types[0]:types[1], heights[0]:heights[1], :360, :]) * std)
    if height_seperate:
        result = tf.reduce_mean(result, axis=[0, 3, 4])  # type and height
        result = tf.sqrt(result)
        result = tf.reshape(result, [(types[1] - types[0]) * (heights[1] - heights[0])])
    else:
        result = tf.reduce_mean(result, axis=[0, 2, 3, 4])  # only type is left
        result = tf.sqrt(result)
        result = result
    return result


def variable_summaries(var, name, verbose=0):
    """
    Add summaries, by default only adds histogram,
    if verbose=1, add mean, stddev, min, max
    The default namespace is summary
    """
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        if verbose:
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def get_n_trainable_parameters():
    """
    Get the number of trainable parameters in the current graph
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


class base_model():
    def __init__(self, sess, FLAGS):
        """
        Default initializer
        """
        self.x = None  # input placeholder
        self.y = None  # output placeholder
        self.loss = None  # loss operator
        self.global_step = None  # global step counter
        self.train_op = None  # train operator
        self.evaluate_op = None  # evaluation fcn, which can be different from loss
        self.pred = None  # operator to get the prediction of the net

        self.summary_op = None  # summary operator
        self.latest_model = None  # string of the path of the latest model
        self.sess = sess  # session object
        self.FLAGS = FLAGS  # flag object
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self._buildnet()

        # define saver and logger objects
        self.saver = tf.train.Saver(max_to_keep=30, keep_checkpoint_every_n_hours=6)
        self.logger = tf.summary.FileWriter(osp.join(self.FLAGS.logdir, self.FLAGS.exp), self.sess.graph)
        self._check_graph()

        self.varlist = None # recomputation lists
        self.train_lossL = []
        self.train_epochL = []
        self.validate_lossL = []
        self.validate_epochL = []

    def _check_graph(self):
        """
        check if the all the mandatory operators are defined
        """
        assert (self.x is not None)
        assert (self.y is not None)
        assert (self.loss is not None)
        assert (self.pred is not None)
        assert (self.train_op is not None)

    def _buildnet(self):
        """
        building the network in the graph
        called inside of the class constructor
        strongly virtual function, requires child class implementation
        ALL graph definition must be inside this function
        """
        raise NotImplementedError()

    def get_train_dict(self, data):
        return {self.x: data['X'], self.y: data['Y']}

    def _train_average_loss(self, indata):
        """
        private function for training and calculating loss for each epoch / dataset
        """
        loss_list = []
        batch_len = []
        include_summary = self.summary_op is not None
        while True:
            try:
                data = self.sess.run(indata)
                train_dict = self.get_train_dict(data)
                # calculate loss
                if include_summary:
                    summary, _, steploss = self.sess.run([self.summary_op, self.train_op, self.loss],
                                                         feed_dict=train_dict)
                    # self.logger.add_summary(summary, self.global_step)
                else:
                    _, steploss = self.sess.run([self.train_op, self.loss], feed_dict=train_dict)
                loss_list.append(steploss)
                batch_len.append(data['Y'].shape[0])
            except tf.errors.OutOfRangeError:
                break
        loss = np.sum(np.array(loss_list) * np.array(batch_len)) / np.sum(np.array(batch_len))
        return loss

    def _test_average_loss(self, indata, include_pred=True):
        loss_list = []
        eval_list = []
        batch_len = []
        pred = []
        predflag = False
        include_eval = self.evaluate_op is not None
        while True:
            try:
                data = self.sess.run(indata)
                train_dict = self.get_train_dict(data)
                if include_eval:
                    seval, spred, steploss = self.sess.run([self.evaluate_op, self.pred, self.loss],
                                                           feed_dict=train_dict)
                else:
                    spred, steploss = self.sess.run([self.pred, self.loss], feed_dict=train_dict)
                    seval = np.array([-1])
                loss_list.append(steploss)
                eval_list.append(seval)
                batch_len.append(data['Y'].shape[0])

                if predflag is False:
                    predflag = include_pred
                    if predflag:
                        pred = spred
                else:
                    pred = np.concatenate([pred, spred], axis=0)
            except tf.errors.OutOfRangeError:
                break
        eval = np.sum(np.array(eval_list) * np.array(batch_len).reshape([-1, 1]), axis=0) / np.sum(np.array(batch_len))
        loss = np.sum(np.array(loss_list) * np.array(batch_len)) / np.sum(np.array(batch_len))
        return loss, pred, eval

    def myprint(self, string):
        """
        print function controlled by verbose flag
        Can be overloaded to print to file etc
        """
        if self.FLAGS.verbose == 1:
            print(string)

    def build_train_op(self):
        """
        function to build a default adam train operator with weight decay
        update self.train_op
        """
        with tf.name_scope('train'):
            if self.FLAGS.weight_decay:
                lr = tf.train.polynomial_decay(
                    learning_rate=self.FLAGS.lr,
                    global_step=self.global_step,
                    decay_steps=self.FLAGS.epoch,
                    end_learning_rate=self.FLAGS.lr / self.FLAGS.lr_decay_val,
                    power=1.0,
                    cycle=False,
                    name=None
                )
            else:
                lr = self.FLAGS.lr
            trainer = tf.train.AdamOptimizer(learning_rate=lr)
            extra_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS)  # Ensure correct ordering for batch-norm to work
            with tf.control_dependencies(extra_update_ops):
                if self.FLAGS.recompute and self.varlist is not None:
                    print("INFO: Enable recomputation")
                    vrs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    grads = gradients(self.loss, vrs, checkpoints=self.varlist)
                    self.train_op = trainer.apply_gradients(zip(grads, vrs))
                else:
                    print("INFO: Disable recomputation")
                    try:
                        self.train_op = trainer.minimize(self.loss)
                    except:
                        self.train_op = self.loss
            # self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

    def run(self, iter_data, iter_val=None, train=None, load=False):
        """
        run function serves as both training and testing
        controlled by FLAG.train boolean by default, but changeable by passing boolean into train argument
        if in the test mode, the function will return a numpy array of pred defined by _buildnet
        if in the train mode, the function will return the latest validation loss, if no validation -1 will be returned
        train (bool) : for training or test
        load (string or bool) : to load the ckpt file, if True, then load the latest model recorded
        """
        if train is None:
            train = self.FLAGS.train
        if load is True:
            self.saver.restore(self.sess, self.latest_model)
        elif load is not False:
            self.saver.restore(self.sess, load)

        data = iter_data.get_next()
        if iter_val is not None:
            val_data = iter_val.get_next()

        if train:
            if self.FLAGS.resume_iter != -1:
                self.myprint("\nResume Training mode: ")
                model_file = osp.join(self.FLAGS.ckptdir, self.FLAGS.exp, 'model_{}'.format(self.FLAGS.resume_iter))
                self.saver.restore(self.sess, model_file)
                print("Loaded from: ", model_file)
                self.global_step = tf.constant(self.FLAGS.resume_iter)
            else:
                self.myprint("\nInitiating Training mode: ")
                self.sess.run(tf.global_variables_initializer())
                total_parameters = get_n_trainable_parameters()
                self.myprint("\nNumber of total parameters is: {}\n".format(total_parameters))

            best_val = 1e100
            val_ct = 0
            latest_val_loss = -1

            for i in range(self.FLAGS.epoch):
                self.global_step = self.global_step + 1
                self.sess.run(iter_data.initializer)
                step = tf.train.global_step(self.sess, self.global_step)
                # get data
                start_time = time.time()
                loss = self._train_average_loss(data)
                self.train_epochL.append(step)
                self.train_lossL.append(loss)
                time_used = time.time() - start_time
                # save model
                if step % self.FLAGS.save_interval == 0:
                    savedir = osp.join(self.FLAGS.ckptdir, self.FLAGS.exp)
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    self.latest_model = osp.join(savedir, 'model_{}'.format(step))
                    self.saver.save(self.sess, self.latest_model)
                    print("model saved to: ", self.latest_model)
                # logging
                if step % self.FLAGS.log_interval == 0:
                    self.myprint("epoch " + str(step) + ": Training loss is: {:.6f}, time passed: {:.2f}s".format(loss,
                                                                                                                  time_used))
                    if iter_val is not None and step % self.FLAGS.val_interval == 0:
                        self.sess.run(iter_val.initializer)
                        start_time = time.time()
                        loss, _, eval = self._test_average_loss(val_data, include_pred=False)
                        if self.evaluate_op is not None:
                            latest_val_loss = eval
                        else:
                            latest_val_loss = loss
                        self.validate_epochL.append(step)
                        self.validate_lossL.append(loss)
                        time_used = time.time() - start_time
                        self.myprint("Validation loss is: {:.6f}, time passed: {:.2f}s".format(loss, time_used))
                        val_ct += 1
                        if loss < best_val:
                            best_val = loss
                            val_ct = 0
                        if val_ct == self.FLAGS.patience:
                            break
            info = [self.train_epochL, self.validate_epochL, self.train_lossL, self.validate_lossL]
            dumpdir = osp.join(self.FLAGS.logdir, self.FLAGS.exp)
            with open(osp.join(dumpdir, "info"), 'wb') as f:
                pickle.dump(info, f)
            if self.FLAGS.plot:
                fig, ax = plt.subplots()
                ax.plot(self.train_epochL, self.train_lossL, label='Train')
                ax.plot(self.validate_epochL, self.validate_lossL, label='Validate')
                plt.xlabel('Epoch')
                plt.ylabel('Loss((Normalized)')
                plt.title('Training and validation loss')
                ax.legend(loc='upper left', frameon=True)
                # plt.ylim((0, 1.5 * max(max(self.train_lossL), max(self.validate_lossL))))
                plt.show()
            return latest_val_loss

        else:
            self.myprint("\nInitiating Testing mode: ")
            self.sess.run(iter_data.initializer)
            loss, pred, eval = self._test_average_loss(data, include_pred=False)
            self.myprint("The test loss is: {:.6f}".format(loss))
            if self.evaluate_op is not None:
                self.myprint("The evaluation score is: " + str(eval))
                # if there is eval, report eval instead of loss
                return eval
            return loss

    def evaluate(self, iter_data, load=False):
        """
        It will print out pred and eval operators
        This method returns the prediction
        """
        if load is True:
            self.saver.restore(self.sess, self.latest_model)
        elif load is not False:
            self.saver.restore(self.sess, load)
        data = iter_data.get_next()
        self.myprint("\nInitiating Evaluation mode: ")
        self.sess.run(iter_data.initializer)
        loss, pred, eval = self._test_average_loss(data)
        self.myprint("The test loss is: {:.6f}".format(loss))
        if (self.evaluate_op is not None):
            self.myprint("The evaluation score is: " + str(eval))
        return pred

    def get_one_predict(self, iter_data, load=False):
        if load is True:
            self.saver.restore(self.sess, self.latest_model)
        elif load is not False:
            self.saver.restore(self.sess, load)
        elif self.FLAGS.resume_iter != -1:
            model_file = osp.join(self.FLAGS.ckptdir, self.FLAGS.exp, 'model_{}'.format(self.FLAGS.resume_iter))
            self.saver.restore(self.sess, model_file)
        else:
            assert(0), "cannot load the model"
        indata = iter_data.get_next()
        self.sess.run(iter_data.initializer)
        data = self.sess.run(indata)

        date = str(data['Date'][0][0])
        train_dict = self.get_train_dict(data)
        spred, steploss = self.sess.run([self.pred, self.loss], feed_dict=train_dict)
        xshape = train_dict[self.x].shape

        x1 = getCrop(train_dict[self.x][:, xshape[1] // 2: xshape[1], :, :, :].reshape(-1, xshape[1] // 2, xshape[2], xshape[4], xshape[3]))[0, :, :, :360, :]
        y = getCrop(train_dict[self.y].reshape(-1, xshape[1] // 2, xshape[2], xshape[4], xshape[3]))[0, :, :, :360, :]

        pred = spred[0, :]
        return x1, y, pred, date

    def get_predictions(self, iter_data, file_comment, load=False):
        """
        save predictions in numpy file format in directory: log/<exp>/*.npy
        :param iter_data: iterator of test data
        :param file_comment: prefix of the file
        :return None
        """
        if load is not False:
            self.saver.restore(self.sess, load)
        else:
            assert(self.FLAGS.resume_iter != -1), "Need to input a model for prediction"
            model_file = osp.join(self.FLAGS.ckptdir, self.FLAGS.exp, 'model_{}'.format(self.FLAGS.resume_iter))
            self.saver.restore(self.sess, model_file)
            print("Loaded from: ", model_file)
            self.global_step = tf.constant(self.FLAGS.resume_iter)

        types = getTypes()
        heights = getHeights()
        std_path = global_macros.TF_DATA_DIRECTORY + "/std_" + file_comment + ".npy"
        mean_path = global_macros.TF_DATA_DIRECTORY + "/mean_" + file_comment + ".npy"
        std = np.expand_dims(np.load(std_path), axis=0)[:, types[0]:types[1], heights[0]:heights[1], :, :]
        mean = np.expand_dims(np.load(mean_path), axis=0)[:, types[0]:types[1], heights[0]:heights[1], :, :]

        indata = iter_data.get_next()
        self.myprint("\nGetting predictions: ")
        self.sess.run(iter_data.initializer)

        # loss, pred, _ = self._test_average_loss(data,  include_pred=True)
        assert(self.FLAGS.batch_size==1) , "Only batch size one is supported for prediction dump"

        while True:
            try:
                data = self.sess.run(indata)
                train_dict = self.get_train_dict(data)
                x48 = data['Y'][:, types[0]:types[1], heights[0]:heights[1], :, :]
                y48 = data['X'][:, 7+types[0]:7+types[1], heights[0]:heights[1], :, :]
                spred = self.sess.run(self.pred, feed_dict=train_dict)
                date = data['Date'][0][0]

                y48 = (y48 * std + mean)[0,0,0,:,:]
                x48 = (x48 * std + mean)[0,0,0,:,:]
                pred = (spred * std + mean)[0,0,0,:,:]
                # concatenate the ens forecast
                pred = np.concatenate([pred, y48[360:,:]], axis=0)
                value = np.stack([pred, y48, x48], axis=0)

                datestr = str(date)
                dumpdir = osp.join(self.FLAGS.logdir, self.FLAGS.exp, datestr)
                np.save(dumpdir, value)
            except tf.errors.OutOfRangeError:
                break

        return


class general_model(base_model):
    def __init__(self, sess, FLAGS, file_comment=''):
        self.file_comment = file_comment
        base_model.__init__(self, sess, FLAGS)

    def _buildnet(self):
        self.define_net()

        self.loss = getLoss(self.y, self.pred, self.FLAGS.L1_loss)
        self.evaluate_op = getEvaluate(self.y, self.pred, self.file_comment)
        self.build_train_op()
        self.summary_op = tf.summary.merge_all()

    def define_net(self):
        """
        This layer is defined to be network specific
        """
        raise NotImplementedError()

    def _get_loss_array(self, indata, xchannels):
        loss_list = []
        ref_loss_list = []
        eval_list = []
        batch_len = []
        include_eval = self.evaluate_op is not None
        while True:
            try:
                data = self.sess.run(indata)
                train_dict = self.get_train_dict(data)
                ref_loss = self.sess.run([getLoss(self.x[:, xchannels//2: xchannels, :, :, :], self.y, self.FLAGS.L1_loss)],
                                         feed_dict=train_dict)
                if include_eval:
                    seval, steploss = self.sess.run([self.evaluate_op, self.loss], feed_dict=train_dict)
                else:
                    steploss = self.sess.run([self.loss], feed_dict=train_dict)
                    seval = np.array([-1])
                loss_list.append(steploss)
                eval_list.append(seval)
                ref_loss_list.append(ref_loss)
                batch_len.append(data['Y'].shape[0])
            except tf.errors.OutOfRangeError:
                break
        eval = np.array(eval_list)
        loss = np.array(loss_list)
        refloss = np.array(ref_loss_list)
        return loss, refloss, eval

    def get_loss_arrays(self, iter_data, load=False, xchannels=14):
        """
        The method returns two lists of rms pred error and rms forecast error
        """
        if load is not False:
            self.saver.restore(self.sess, load)
        else:
            assert(self.FLAGS.resume_iter != -1), "Need to input a model for prediction"
            model_file = osp.join(self.FLAGS.ckptdir, self.FLAGS.exp, 'model_{}'.format(self.FLAGS.resume_iter))
            self.saver.restore(self.sess, model_file)
            print("Loaded from: ", model_file)
            self.global_step = tf.constant(self.FLAGS.resume_iter)

        data = iter_data.get_next()
        self.myprint("\nGetting loss arrays: ")
        self.sess.run(iter_data.initializer)
        loss, refloss, eval = self._get_loss_array(data, xchannels)  # Hack warning
        return loss, refloss, eval

    def data_crop(self, div, crop=False, stack=True):
        """
        If crop is true: crop image randomly with the specific ratio
        If crop is false: round longitude and latitude to be a factor of 4
        """
        _, c, l, w, h = self.x.get_shape()
        _, c2, _, _, _ = self.y.get_shape()
        w = int(w)
        h = int(h)
        if crop and self.FLAGS.train:
            xshape = tf.shape(self.x)
            yshape = tf.shape(self.y)
            ws = int(w/div[0])
            hs = int(h/div[1])
            if stack:
                x_tensors = [[None for indj in range(div[1])] for jndi in range(div[0])]
                y_tensors = [[None for indj in range(div[1])] for jndi in range(div[0])]
                for i in range(div[0]):
                    for j in range(div[1]):
                        lx = i * w // div[0]
                        # hx = (rxy[0] + 1) * w // div[0]
                        ly = j * h // div[1]
                        # hy = (rxy[1] + 1) * h // div[1]
                        x_tensors[i][j] = self.x[:, :, :, lx:(lx+ws), ly:(ly+hs)]
                        y_tensors[i][j] = self.y[:, :, :, lx:(lx+ws), ly:(ly+hs)]
                for i in range(div[0]):
                    for j in range(div[1]):
                        # x = tf.cond(sel[i, j], tf.concat([x_tensors[i][j], x], axis=0), x)
                        # y = tf.cond(sel[i, j], tf.concat([y_tensors[i][j], y], axis=0), y)
                        x = tf.concat([x_tensors[i][j] for i in range(div[0]) for j in range(div[1])], axis=0)
                        y = tf.concat([y_tensors[i][j] for i in range(div[0]) for j in range(div[1])], axis=0)
                return x, y
            else:
                from tensorflow.python.ops import array_ops
                from tensorflow.python.framework import ops
                size = [xshape[0], c, l, ws, hs]
                sizey = [yshape[0], c2, l, ws, hs]
                size = ops.convert_to_tensor(size, dtype=tf.int32, name="size")
                sizey = ops.convert_to_tensor(sizey, dtype=tf.int32, name="sizey")
                woffset = tf.random_uniform(
                    (1,), dtype=tf.int32,
                    maxval=ws,
                    seed=None)
                hoffset = tf.random_uniform(
                    (1,), dtype=tf.int32,
                    maxval=hs,
                    seed=None)
                offset = ops.convert_to_tensor([0, 0, 0, woffset[0], hoffset[0]], dtype=tf.int32)
                x = array_ops.slice(self.x, offset, size)
                y = array_ops.slice(self.y, offset, sizey)
                return x, y
        else:
            x = self.x[:, :, :, :(w - w % 4), :(h - h % 4)]
            y = self.y[:, :, :, :(w - w % 4), :(h - h % 4)]
        return x, y
