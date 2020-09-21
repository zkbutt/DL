from keras.preprocessing import image

from keras.models import Model
from keras import layers
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.utils.data_utils import get_file
from keras import backend
import numpy as np

BASE_WEIGTHS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/densenet/')
DENSENET121_WEIGHT_PATH = (
        BASE_WEIGTHS_PATH +
        'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH = (
        BASE_WEIGTHS_PATH +
        'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH = (
        BASE_WEIGTHS_PATH +
        'densenet201_weights_tf_dim_ordering_tf_kernels.h5')


def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def conv_block(x, growth_rate, name):
    bn_axis = 3
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def transition_block(x, reduction, name):
    bn_axis = 3
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def DenseNet(blocks,
             input_shape=None,
             classes=1000,
             **kwargs):
    img_input = layers.Input(shape=input_shape)

    bn_axis = 3

    # 224,224,3 -> 112,112,64
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)

    # 112,112,64 -> 56,56,64
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    # 56,56,64 -> 56,56,64+32*block[0]
    # Densenet121 56,56,64 -> 56,56,64+32*6 == 56,56,256
    x = dense_block(x, blocks[0], name='conv2')

    # 56,56,64+32*block[0] -> 28,28,32+16*block[0]
    # Densenet121 56,56,256 -> 28,28,32+16*6 == 28,28,128
    x = transition_block(x, 0.5, name='pool2')

    # 28,28,32+16*block[0] -> 28,28,32+16*block[0]+32*block[1]
    # Densenet121 28,28,128 -> 28,28,128+32*12 == 28,28,512
    x = dense_block(x, blocks[1], name='conv3')

    # Densenet121 28,28,512 -> 14,14,256
    x = transition_block(x, 0.5, name='pool3')

    # Densenet121 14,14,256 -> 14,14,256+32*block[2] == 14,14,1024
    x = dense_block(x, blocks[2], name='conv4')

    # Densenet121 14,14,1024 -> 7,7,512
    x = transition_block(x, 0.5, name='pool4')

    # Densenet121 7,7,512 -> 7,7,256+32*block[3] == 7,7,1024
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='fc1000')(x)

    inputs = img_input

    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='densenet201')
    else:
        model = Model(inputs, x, name='densenet')
    return model


def DenseNet121(input_shape=[224, 224, 3],
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 24, 16],
                    input_shape, classes,
                    **kwargs)


def DenseNet169(input_shape=[224, 224, 3],
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 32, 32],
                    input_shape, classes,
                    **kwargs)


def DenseNet201(input_shape=[224, 224, 3],
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 48, 32],
                    input_shape, classes,
                    **kwargs)


def preprocess_input(x):
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]
    return x


if __name__ == '__main__':
    # model = DenseNet121()
    # weights_path = get_file(
    # 'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
    # DENSENET121_WEIGHT_PATH,
    # cache_subdir='models',
    # file_hash='9d60b8095a5708f2dcce2bca79d332c7')

    model = DenseNet169()
    weights_path = get_file(
        'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
        DENSENET169_WEIGHT_PATH,
        cache_subdir='models',
        file_hash='d699b8f76981ab1b30698df4c175e90b')

    # model = DenseNet201()
    # weights_path = get_file(
    # 'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
    # DENSENET201_WEIGHT_PATH,
    # cache_subdir='models',
    # file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
    model.load_weights(weights_path)
    model.summary()
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds))
