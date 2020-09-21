import numpy as np
from keras.utils import plot_model
from keras.layers import Input, Conv2D, MaxPool2D
from keras.layers import Activation, Add, Concatenate, Conv2D
from keras.layers import GlobalAveragePooling2D, Dense
from keras.layers import MaxPool2D, AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
from keras.models import Model
import keras.backend as K

import numpy as np


def channel_split(x, name=''):
    # 输入进来的通道数
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    # 对通道数进行分割
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c


def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    # 通道交换
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio, strides=2, stage=1, block=1):
    bn_axis = -1

    prefix = 'stage{}/block{}'.format(stage, block)

    # [116, 232, 464]
    bottleneck_channels = int(out_channels * bottleneck_ratio / 2)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    # [116, 232, 464]
    x = Conv2D(bottleneck_channels, kernel_size=(1, 1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(
        inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)

    # 深度可分离卷积
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)

    # [116, 232, 464]
    x = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    # 当strides等于2的时候，残差边需要添加卷积
    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)

        s2 = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1_conv_3'.format(prefix))(
            s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage - 1],
                     strides=2, bottleneck_ratio=bottleneck_ratio, stage=stage, block=1)

    for i in range(1, repeat + 1):
        x = shuffle_unit(x, out_channels=channel_map[stage - 1], strides=1,
                         bottleneck_ratio=bottleneck_ratio, stage=stage, block=(1 + i))

    return x


def ShuffleNetV2(input_tensor=None,
                 pooling='max',
                 input_shape=(224, 224, 3),
                 num_shuffle_units=[3, 7, 3],
                 scale_factor=1,
                 bottleneck_ratio=1,
                 classes=1000):
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))

    out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}

    out_channels_in_stage = np.array([1, 1, 2, 4])
    out_channels_in_stage *= out_dim_stage_two[scale_factor]  # calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage = out_channels_in_stage.astype(int)

    img_input = Input(shape=input_shape)

    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                  repeat=repeat,
                  bottleneck_ratio=bottleneck_ratio,
                  stage=stage + 2)

    if scale_factor != 2:
        x = Conv2D(1024, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)
    else:
        x = Conv2D(2048, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    x = Dense(classes, name='fc')(x)
    x = Activation('softmax', name='softmax')(x)

    inputs = img_input

    model = Model(inputs, x, name=name)

    return model


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model = ShuffleNetV2(input_shape=(224, 224, 3), scale_factor=1)
    model.summary()
