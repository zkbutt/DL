import time
import warnings

import math
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import numpy as np


class FStoppingByLossVal(Callback):
    '''
    ’acc’,’val_acc’,’loss’,’val_loss’
    必须传入 ret用于记录
    '''

    def __init__(self, monitor='val_loss', value=0.00001, verbose=0, mode='auto'):
        super(Callback, self).__init__()
        # self.ret = ret  # 保存记录
        self.monitor = monitor  # 定义指标
        self.value = value
        self.verbose = verbose

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs=None):
        '''
        {'loss': 0.6839542388916016, 'binary_accuracy': 0.5} # 没有验证集时
        {'val_loss': 0.7, 'val_mean_squared_error': 0.70, 'loss': 2.3, 'mean_squared_error': 2.3}
        :param epoch:  反向次数
        :param logs:为训练完成传回的结果字典 区分训练集和验证集
        :return:
        '''
        # print(epoch,logs)
        if not logs:
            return
        # self.ret.append(logs)
        # print('on_epoch_end--------', epoch, logs)
        current = logs.get(self.monitor)  # 取值

        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if self.monitor_op(current, self.value):
            if self.verbose > 0:
                print('---feadre---EarlyStoppingByLossVal-----', current, self.value)
                print("---feadre---Epoch %05d: early stopping THR" % epoch)
            print('---feadre---模型满足而结果，当前epoch= %d ' % epoch)
            self.model.stop_training = True


def fit_keras(model, X_train, X_test, y_train, y_test, epochs=10, batch_size=32):
    best_score = 0
    val_acc_list = []  # 用 val_acc_list 保存最新的 10 个 val_acc
    step = 0
    num_fit = math.ceil(X_train.shape[0] / batch_size)

    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

    start_time = int(time.time())
    for i in range(num_fit):
        step += 1
        # x_train, y_train = dataset.next_train_batch()
        X_train = X_train[i * batch_size:(i + 1) * batch_size]
        y_train = y_train[i * batch_size:(i + 1) * batch_size]

        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            verbose=0,
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=[annealer], )
        print(str(step) + "/" + str(num_fit))
        val_acc = history.ret_train['val_accuracy'][0]
        print(
            "FlyAI_Log > train_acc:", history.ret_train['accuracy'][0],
            "- val_acc:", val_acc,
            "- train_loss:", history.ret_train['loss'][0],
            "- val_loss:", history.ret_train['val_loss'][0]
        )
        if len(val_acc_list) >= 10:
            val_acc_list.pop(0)
            val_acc_list.append(val_acc)
        else:
            val_acc_list.append(val_acc)
        # 每隔10步进行一次比较，用来保存最优结果
        if step % 10 == 0 and np.mean(val_acc_list) >= best_score:
            best_score = np.mean(val_acc_list)
            model.save_model('s_model.h5')
            # sqeue.save_weights('s_model_weights.h5')
            time_end = time.time()
            print("********  step %d, best accuracy %g，time %s" % (step, best_score, int(time_end - start_time)))
            start_time = time_end
    pass


def fit_data_batch(data1, data2, batch_size, ind_start=0):
    num_fit = np.math.ceil((data1.shape[0] - ind_start) / batch_size)  # 向上取整
    i = ind_start
    while True:
        ind_start = i * batch_size
        ind_end = (i + 1) * batch_size
        yield data1[ind_start:ind_end], data2[ind_start:ind_end], num_fit
        if ind_end >= data1.shape[0]:
            return
        i += 1
