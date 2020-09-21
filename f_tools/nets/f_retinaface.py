import warnings
import numpy as np
from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras import backend as K


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu6(x):
    return K.relu(x, max_value=6)


def MobileNet(img_input, depth_multiplier=1):
    # 640,640,3 -> 320,320,8
    x = _conv_block(img_input, 8, strides=(2, 2))
    # 320,320,8 -> 320,320,16
    x = _depthwise_conv_block(x, 16, depth_multiplier, block_id=1)

    # 320,320,16 -> 160,160,32
    x = _depthwise_conv_block(x, 32, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 32, depth_multiplier, block_id=3)

    # 160,160,32 -> 80,80,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=5)
    feat1 = x

    # 80,80,64 -> 40,40,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=11)
    feat2 = x

    # 40,40,128 -> 20,20,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=13)
    feat3 = x

    return feat1, feat2, feat3


def ClassHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 2, kernel_size=1, strides=1)(inputs)
    return Activation("softmax")(Reshape([-1, 2])(outputs))


def BboxHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 4, kernel_size=1, strides=1)(inputs)
    return Reshape([-1, 4])(outputs)


def LandmarkHead(inputs, num_anchors=2):
    outputs = Conv2D(num_anchors * 10, kernel_size=1, strides=1)(inputs)
    return Reshape([-1, 10])(outputs)


def RetinaFace(cfg, backbone="mobilenet"):
    inputs = Input(shape=(None, None, 3))

    if backbone == "mobilenet":
        C3, C4, C5 = MobileNet(inputs)
    elif backbone == "resnet50":
        C3, C4, C5 = ResNet50(inputs)
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

    leaky = 0
    if (cfg['out_channel'] <= 64):
        leaky = 0.1
    P3 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=1, strides=1, padding='same', name='C3_reduced', leaky=leaky)(
        C3)
    P4 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=1, strides=1, padding='same', name='C4_reduced', leaky=leaky)(
        C4)
    P5 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=1, strides=1, padding='same', name='C5_reduced', leaky=leaky)(
        C5)

    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, P4])
    P4 = Add(name='P4_merged')([P5_upsampled, P4])
    P4 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=3, strides=1, padding='same', name='Conv_P4_merged',
                         leaky=leaky)(P4)

    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, P3])
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D_BN_Leaky(cfg['out_channel'], kernel_size=3, strides=1, padding='same', name='Conv_P3_merged',
                         leaky=leaky)(P3)


def SSH(inputs, out_channel, leaky=0.1):
    conv3X3 = Conv2D_BN(out_channel // 2, kernel_size=3, strides=1, padding='same')(inputs)

    conv5X5_1 = Conv2D_BN_Leaky(out_channel // 4, kernel_size=3, strides=1, padding='same', leaky=leaky)(inputs)
    conv5X5 = Conv2D_BN(out_channel // 4, kernel_size=3, strides=1, padding='same')(conv5X5_1)

    conv7X7_2 = Conv2D_BN_Leaky(out_channel // 4, kernel_size=3, strides=1, padding='same', leaky=leaky)(conv5X5_1)
    conv7X7 = Conv2D_BN(out_channel // 4, kernel_size=3, strides=1, padding='same')(conv7X7_2)

    out = Concatenate(axis=-1)([conv3X3, conv5X5, conv7X7])
    out = Activation("relu")(out)
    return out
