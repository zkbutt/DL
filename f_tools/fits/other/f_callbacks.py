'''-----------------------------------学习率下降------------------------------------------'''

'''
阶层性下降
reduce_lr = ReduceLROnPlateau(min_delta=1e-4, monitor='val_loss', factor=0.5, patience=2, verbose=1)
    factor：在某一项指标不继续下降后学习率下降的比率
    patience：在某一项指标不继续下降几个时代后，学习率开始下降

线性下降
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
'''

'''
指数型下降
exponent_lr = ExponentDecayScheduler(learning_rate_base = learning_rate_base,
                                    global_epoch_init = init_epoch,
                                    decay_rate = 0.9,
                                    min_learn_rate = 1e-6
                                    )
'''

import tensorflow.keras
from tensorflow.keras import backend as K


def get_optimizer():
    # 优化器加学习率下降
    import tensorflow as tf
    def lr_schedule():
        return tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=STEPS_PER_EPOCH * 1000,
            decay_rate=1,
            staircase=False)

    return tf.keras.optimizers.Adam(lr_schedule)


def exponent(global_epoch, learning_rate_base, decay_rate, min_learn_rate=0):
    '''

    :param global_epoch:更新时才有显示
    :param learning_rate_base:基础学习率
    :param decay_rate:衰减系数
    :param min_learn_rate: 当前的学习率
    :return:
    '''
    learning_rate = learning_rate_base * pow(decay_rate, global_epoch)
    learning_rate = max(learning_rate, min_learn_rate)
    return learning_rate


class ExponentDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """

    def __init__(self,
                 learning_rate_base,
                 decay_rate,
                 global_epoch_init=0,
                 min_learn_rate=0,
                 verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 全局初始化epoch
        self.global_epoch = global_epoch_init

        self.decay_rate = decay_rate
        # 参数显示
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

    def on_epoch_end(self, epochs, logs=None):
        self.global_epoch = self.global_epoch + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    # 更新学习率
    def on_epoch_begin(self, batch, logs=None):
        lr = exponent(global_epoch=self.global_epoch,
                      learning_rate_base=self.learning_rate_base,
                      decay_rate=self.decay_rate,
                      min_learn_rate=self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            # 更新显示
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_epoch + 1, lr))


'''
余弦退火衰减： 学习率会先上升再下降，上升的时候使用线性上升，下降的时候模拟cos函数下降
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=1e-5,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=5,
                                            min_learn_rate = 1e-6
                                            )
'''

import numpy as np


def cosine_decay_with_warmup1(global_step,
                              learning_rate_base,
                              total_steps,
                              warmup_learning_rate=0.0,
                              warmup_steps=0,
                              hold_base_rate_steps=0,
                              min_learn_rate=0,
                              ):
    """
    参数：
            global_step: 上面定义的Tcur，记录当前执行的步数。
            learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
            total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
            warmup_learning_rate: 这是warm up阶段线性增长的初始值
            warmup_steps: warm_up总的需要持续的步数
            hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
    """
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    # 这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
                                                           (global_step - warmup_steps - hold_base_rate_steps) / float(
        total_steps - warmup_steps - hold_base_rate_steps)))
    # 如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        # 线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        # 只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)

    learning_rate = max(learning_rate, min_learn_rate)
    return learning_rate


class WarmUpCosineDecayScheduler1(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 min_learn_rate=0,
                 verbose=0):
        super(WarmUpCosineDecayScheduler1, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 总共的步数，训练完所有世代的步数epochs * sample_count / batch_size
        self.total_steps = total_steps
        # 全局初始化step
        self.global_step = global_step_init
        # 热调整参数
        self.warmup_learning_rate = warmup_learning_rate
        # 热调整步长，warmup_epoch * sample_count / batch_size
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        # 参数显示
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

    # 更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    # 更新学习率
    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup1(global_step=self.global_step,
                                       learning_rate_base=self.learning_rate_base,
                                       total_steps=self.total_steps,
                                       warmup_learning_rate=self.warmup_learning_rate,
                                       warmup_steps=self.warmup_steps,
                                       hold_base_rate_steps=self.hold_base_rate_steps,
                                       min_learn_rate=self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


'''
余弦退火多次上升下降

'''


def cosine_decay_with_warmup2(global_step,
                              learning_rate_base,
                              total_steps,
                              warmup_learning_rate=0.0,
                              warmup_steps=0,
                              hold_base_rate_steps=0,
                              min_learn_rate=0,
                              ):
    """
    参数：
            global_step: 上面定义的Tcur，记录当前执行的步数。
            learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
            total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
            warmup_learning_rate: 这是warm up阶段线性增长的初始值
            warmup_steps: warm_up总的需要持续的步数
            hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
    """
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    # 这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
                                                           (global_step - warmup_steps - hold_base_rate_steps) / float(
        total_steps - warmup_steps - hold_base_rate_steps)))
    # 如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        # 线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        # 只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)

    learning_rate = max(learning_rate, min_learn_rate)
    return learning_rate


class WarmUpCosineDecayScheduler2(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 min_learn_rate=0,
                 # interval_epoch代表余弦退火之间的最低点
                 interval_epoch=[0.05, 0.15, 0.30, 0.50],
                 verbose=0):
        super(WarmUpCosineDecayScheduler2, self).__init__()
        # 基础的学习率
        self.learning_rate_base = learning_rate_base
        # 热调整参数
        self.warmup_learning_rate = warmup_learning_rate
        # 参数显示
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = []

        self.interval_epoch = interval_epoch
        # 贯穿全局的步长
        self.global_step_for_interval = global_step_init
        # 用于上升的总步长
        self.warmup_steps_for_interval = warmup_steps
        # 保持最高峰的总步长
        self.hold_steps_for_interval = hold_base_rate_steps
        # 整个训练的总步长
        self.total_steps_for_interval = total_steps

        self.interval_index = 0
        # 计算出来两个最低点的间隔
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch) - 1):
            self.interval_reset.append(self.interval_epoch[i + 1] - self.interval_epoch[i])
        self.interval_reset.append(1 - self.interval_epoch[-1])

    # 更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        self.global_step_for_interval = self.global_step_for_interval + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    # 更新学习率
    def on_batch_begin(self, batch, logs=None):
        # 每到一次最低点就重新更新参数
        if self.global_step_for_interval in [0] + [int(i * self.total_steps_for_interval) for i in self.interval_epoch]:
            self.total_steps = self.total_steps_for_interval * self.interval_reset[self.interval_index]
            self.warmup_steps = self.warmup_steps_for_interval * self.interval_reset[self.interval_index]
            self.hold_base_rate_steps = self.hold_steps_for_interval * self.interval_reset[self.interval_index]
            self.global_step = 0
            self.interval_index += 1

        lr = cosine_decay_with_warmup2(global_step=self.global_step,
                                       learning_rate_base=self.learning_rate_base,
                                       total_steps=self.total_steps,
                                       warmup_learning_rate=self.warmup_learning_rate,
                                       warmup_steps=self.warmup_steps,
                                       hold_base_rate_steps=self.hold_base_rate_steps,
                                       min_learn_rate=self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


'''
-----------------------------------训练停止------------------------------------------
自定义停止
stopping1 = EarlyStopping(
    monitor='loss',  # 选择模式
    min_delta=0.0001,  # 增长减小率
    patience=100,  # 连续多少次
    verbose=1,  # 日志
    mode='min'
)
stopping2 = FStoppingByLossVal(
    monitor='loss',
    value=0.05,
    mode='min'  # 大于就over
)
'''

import warnings


class FStoppingByLossVal(keras.callbacks.Callback):
    '''
    ’acc’,’val_acc’,’loss’,’val_loss’
    必须传入 ret用于记录
    '''

    def __init__(self, monitor='val_loss', value=0.00001, verbose=0, mode='auto'):
        super(FStoppingByLossVal, self).__init__()
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
