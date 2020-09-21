from tensorflow.keras import layers, Model, Sequential
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU, Add
from tensorflow.python.keras.utils.vis_utils import plot_model


def lower_block(input, channel, state='other'):
    if state == 'first':  # 是第一块，
        s = 1
        dim_add = True
    elif state == 'dotted':
        s = 2
        dim_add = True

    elif state == 'other':
        s = 1
        dim_add = False
    else:
        assert '参数错误：%s 只能为 first、dotted、other' % state

    x = Conv2D(channel, kernel_size=3, padding="SAME", strides=s, use_bias=False)(input)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = ReLU()(x)

    x = Conv2D(channel, kernel_size=3, padding="SAME", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    if dim_add:  # 低层不升维
        y = Conv2D(channel, kernel_size=1, strides=s, padding="SAME", use_bias=False)(input)
        y = BatchNormalization(momentum=0.9, epsilon=1e-5)(y)
    else:
        y = input

    x = Add()([y, x])  # 叠加
    x = ReLU()(x)
    return x


def hight_block(input, channel, state='other'):
    '''
    state:
        first:是第一个块，的第一次
        dotted:不是第一块（虚线），的第一次
        other:其它块

    '''
    if state == 'first':  # 是第一块，
        s = 1
        dim_add = True
    elif state == 'dotted':
        s = 2
        dim_add = True

    elif state == 'other':
        s = 1
        dim_add = False
    else:
        assert '参数错误：%s 只能为 first、dotted、other' % state

    x = Conv2D(channel, kernel_size=1, strides=s, use_bias=False)(input)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = ReLU()(x)

    x = Conv2D(channel, kernel_size=3, padding="SAME", use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = ReLU()(x)

    x = Conv2D(channel * 4, kernel_size=1, use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)

    if dim_add:  # 高层要升维
        y = Conv2D(channel * 4, kernel_size=1, strides=s, padding="SAME", use_bias=False)(input)
        y = BatchNormalization(momentum=0.9, epsilon=1e-5)(y)
    else:
        y = input

    x = Add()([y, x])  # 叠加
    x = ReLU()(x)
    return x


def _make_block(input, channel, block_num, is_first=False, is_lower=False):
    # 块尺寸不变
    state = 'first' if is_first else 'dotted'
    if is_lower:
        x = lower_block(input, channel, state=state)
    else:
        x = hight_block(input, channel, state=state)

    for i in range(1, block_num):
        if is_lower:
            x = lower_block(x, channel, state='other')
        else:
            x = hight_block(x, channel, state='other')
    return x


def _resnet(blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True, is_lower=True):
    # tensorflow中的tensor通道排序是NHWC
    # (None, 224, 224, 3)
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")

    # same卷积，缩小一倍 w/s
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    # 重叠池化 w / s 与卷积一样的计算
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)  # 第一层通过池化降维
    # 通过 strides=2 缩减尺寸 输出56,56,64

    '''
    首先是否第一层：尺寸不变
    低层还是高低：低二卷维度（核个数，通道）不变，高三卷维度*4
    是否实线层：除第一层，后面每个块的第一层为虚线，实线strides=1  虚线第一个strides=2（减一半）
    '''
    x = _make_block(x, 64, blocks_num[0], is_lower=is_lower, is_first=True)
    x = _make_block(x, 128, blocks_num[1], is_lower=is_lower)
    x = _make_block(x, 256, blocks_num[2], is_lower=is_lower)
    x = _make_block(x, 512, blocks_num[3], is_lower=is_lower)

    if include_top:
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)

    return model


def resnet34(im_width=224, im_height=224, num_classes=1000):
    return _resnet([3, 4, 6, 3], im_width, im_height, num_classes, is_lower=True)


def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet([3, 4, 6, 3], im_width, im_height, num_classes, include_top, is_lower=False)


def resnet101(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet([3, 4, 23, 3], im_width, im_height, num_classes, include_top, is_lower=False)


if __name__ == '__main__':
    # resnet_ = resnet50()
    resnet_ = resnet34()
    print(resnet_.summary())
    plot_model(resnet_, to_file='f_resnet.png', show_shapes=True)  # 'TB'表示方向
