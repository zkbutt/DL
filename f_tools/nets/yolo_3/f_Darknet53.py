from tensorflow.python.keras import layers
from tensorflow.python.keras.regularizers import l2

from yolo_3.utils.utils import compose


def Conv2D_BN_Leaky(out_channel, kernel_size=(3, 3), strides=(1, 1)):
    str_padding = 'valid' if strides == (2, 2) else 'same'
    return compose(
        layers.Conv2D(
            filters=out_channel, kernel_size=kernel_size,
            strides=strides, padding=str_padding, use_bias=False,
            kernel_regularizer=l2(5e-4),
        ),
        layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
        layers.LeakyReLU(alpha=0.1),
    )


def cre_resblock(x, num_filters, num_blocks):
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = Conv2D_BN_Leaky(num_filters, strides=(2, 2))(x)

    for i in range(num_blocks):
        y = Conv2D_BN_Leaky(num_filters // 2, (1, 1))(x)
        y = Conv2D_BN_Leaky(num_filters, (3, 3))(y)
        x = layers.Add()([x, y])
    return x


def Darknet53(x):
    x = Conv2D_BN_Leaky(32)(x)
    x = cre_resblock(x, 64, 1)
    x = cre_resblock(x, 128, 2)
    x = cre_resblock(x, 256, 8)
    feat1 = x
    x = cre_resblock(x, 512, 8)
    feat2 = x
    x = cre_resblock(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3
