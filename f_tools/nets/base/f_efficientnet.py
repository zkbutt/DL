# -------------------------------------------------------------#
#   EfficientNet的网络部分
# -------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from multiprocessing import Lock
from multiprocessing.dummy import Pool

import cv2
import math
import tensorflow as tf
import numpy as np
from keras import layers
from keras.datasets import mnist
from keras.models import Model
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing import image

# 用于下载模型的默认参数
from tensorflow.python.keras.applications.imagenet_utils import correct_pad
from tensorflow.python.keras.utils.vis_utils import plot_model

BASE_WEIGHTS_PATH = (
    'https://github.com/Callidior/keras-applications/'
    'releases/download/efficientnet/')
WEIGHTS_HASHES = {
    'b0': ('e9e877068bd0af75e0a36691e03c072c',
           '345255ed8048c2f22c793070a9c1a130'),
    'b1': ('8f83b9aecab222a9a2480219843049a1',
           'b20160ab7b79b7a92897fcb33d52cc61'),
    'b2': ('b6185fdcd190285d516936c09dceeaa4',
           'c6e46333e8cddfa702f4d8b8b6340d70'),
    'b3': ('b2db0f8aac7c553657abb2cb46dcbfbb',
           'e0cf8654fad9d3625190e30d70d0c17d'),
    'b4': ('ab314d28135fe552e2f9312b31da6926',
           'b46702e4754d2022d62897e0618edc7b'),
    'b5': ('8d60b903aff50b09c6acf8eaba098e09',
           '0a839ac36e46552a881f2975aaab442f'),
    'b6': ('a967457886eac4f5ab44139bdd827920',
           '375a35c17ef70d46f9c664b03b4437f2'),
    'b7': ('e964fd6e26e9a4c144bcb811f2a10f20',
           'd55674cc46b805f4382d18bc08ed43c1')
}

# 每个Blocks的参数
DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

# 两个Kernel的初始化器
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def block(inputs, activation_fn=tf.nn.swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):
    bn_axis = 3

    # 升多少维度
    filters = filters_in * expand_ratio

    # 利用Inverted residuals
    # part1 1x1升维度
    if expand_ratio != 1:
        x = layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=name + 'expand_conv')(inputs)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # padding
    if strides == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(x, kernel_size),
                                 name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'

    # part2 利用3x3卷积对每一个channel进行卷积
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=strides,
                               padding=conv_pad,
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation_fn, name=name + 'activation')(x)

    # 压缩后再放大,作为一个调整系数
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = layers.Conv2D(filters_se, 1,
                           padding='same',
                           activation=activation_fn,
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_reduce')(se)
        se = layers.Conv2D(filters, 1,
                           padding='same',
                           activation='sigmoid',
                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'se_expand')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')

    # part3 利用1x1对特征层进行压缩
    x = layers.Conv2D(filters_out, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)

    # 实现残差神经网络
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = layers.Dropout(drop_rate,
                               noise_shape=(None, 1, 1, 1),
                               name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation_fn=tf.nn.swish,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):
    '''

    :param width_coefficient:
    :param depth_coefficient:
    :param default_size:
    :param dropout_rate:
    :param drop_connect_rate:
    :param depth_divisor:
    :param activation_fn:
    :param blocks_args:
    :param weights:
    :param input_tensor:
    :param input_shape:
    :param pooling:
    :param classes:
    :param kwargs:
    :return:
    '''
    input_shape = [416, 416, 3]
    img_input = layers.Input(tensor=input_tensor, shape=input_shape)

    bn_axis = 3

    # 保证filter的大小可以被8整除
    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    # 重复次数，取顶
    def round_repeats(repeats):
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = img_input
    x = layers.ZeroPadding2D(padding=correct_pad(x, 3),
                             name='stem_conv_pad')(x)
    x = layers.Conv2D(round_filters(32), 3,
                      strides=2,
                      padding='valid',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation_fn, name='stem_activation')(x)

    # Build blocks
    from copy import deepcopy

    # 防止参数的改变
    blocks_args = deepcopy(blocks_args)

    b = 0
    # 计算总的block的数量
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, activation_fn, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1

    # 收尾工作
    x = layers.Conv2D(round_filters(1280), 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation_fn, name='top_activation')(x)

    # 利用GlobalAveragePooling2D代替全连接层
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='top_dropout')(x)

    x = layers.Dense(classes,
                     activation='softmax',
                     kernel_initializer=DENSE_KERNEL_INITIALIZER,
                     name='probs')(x)

    # 输入inputs
    inputs = img_input

    model = Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
        file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
        file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
        file_name = model_name + file_suff
        weights_path = get_file(file_name, BASE_WEIGHTS_PATH + file_name,
                                cache_subdir='models',
                                file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def EfficientNetB0(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.0, 1.0, 224, 0.2,
                        model_name='efficientnet-b0',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB1(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.0, 1.1, 240, 0.2,
                        model_name='efficientnet-b1',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB2(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.1, 1.2, 260, 0.3,
                        model_name='efficientnet-b2',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB3(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.2, 1.4, 300, 0.3,
                        model_name='efficientnet-b3',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB4(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.4, 1.8, 380, 0.4,
                        model_name='efficientnet-b4',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB5(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.6, 2.2, 456, 0.4,
                        model_name='efficientnet-b5',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB6(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.8, 2.6, 528, 0.5,
                        model_name='efficientnet-b6',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB7(weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(2.0, 3.1, 600, 0.5,
                        model_name='efficientnet-b7',
                        weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def 结果分析(history):
    import matplotlib.pyplot as plt
    # plot loss and accuracy image
    history_dict = history.history  # 拿到完成训练的结果
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]
    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


def _图像转换(x, ret, j):
    for i in range(x.shape[0]):
        img = x[i].copy()
        img = cv2.resize(img, (416, 416))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('', img)
        # cv2.waitKey()
        # lock.acquire()
        img = img.reshape(1, 416, 416, 3)
        ret[j] = np.append(ret[j], img, axis=0)
        # lock.release()
        # print(ret[j].shape)


def 多进程图像处理(data, num):
    b = data.shape[0] / num
    if b > (data.shape[0] // num):
        num += 2
    b = int(b)

    # b = X_train.shape[0] / num
    ret = [np.empty(shape=[0, 416, 416, 3]) for i in range(num)]
    # 图像转换(X_train[:10], 1)

    pp = Pool(num)
    for i in range(num):
        # 创建进程,放入进程池统一管理
        # print(i * b, (i + 1) * b)
        # print(X_train[int(i * b):int((i + 1) * b)])
        pp.apply_async(_图像转换, args=(data[int(i * b):int((i + 1) * b)], ret, i))
        # 在调用join之前必须先关掉进程池
        # 进程池一旦关闭  就不能再添加新的进程了
    pp.close()
    # 进程池对象调用join,会等待进程池中所有的子进程结束之后再结束父进程
    print('准备等待')
    pp.join()
    print("转换完成...")
    # print(ret)
    ret = np.concatenate(ret, axis=0)
    return ret


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # # 全局使用CPU配置
    model = EfficientNetB0(weights=None, classes=10)
    # net = EfficientNetB7(weights=None)
    # print(model.summary())
    # print(plot_model(net, to_file='f_efficientnet.png', show_shapes=True))

    # 获取训练集
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # b = 113
    X_train = 多进程图像处理(X_train, 10)
    X_test = 多进程图像处理(X_test, 10)
    # print(X_train)

    X_train = X_train / 255
    X_test = X_test / 255

    Y_train = np_utils.to_categorical(Y_train, num_classes=10)  # np转独热
    Y_test = np_utils.to_categorical(Y_test, num_classes=10)

    model.compile(
        loss='categorical_crossentropy',  # 交叉熵损失函数
        optimizer='adam',  # sgd = SGD(lr=0.2)
        metrics=['accuracy']
    )
    model.load_weights('f_efficientnet' + '_weights.h5')

    batch_size = 32
    epochs = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # # 全局使用CPU配置

    ret_train = model.fit(X_train, Y_train,
                          epochs=epochs, batch_size=batch_size,
                          validation_data=(X_test, Y_test),
                          # callbacks=[stopping1],
                          # callbacks=[stopping2],
                          verbose=1,  # 1-显示每批 2-显示每epochs
                          )
    结果分析(ret_train)

    model.save('f_efficientnet' + '.h5')  # 需安装HDF5 pip install h5py
    model.save_weights('f_efficientnet' + '_weights.h5')
