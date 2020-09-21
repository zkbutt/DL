from tensorflow.keras import layers, models, Model, Sequential


def AlexNet(im_height=224, im_width=224, class_num=1000):
    # tensorflow中的tensor通道排序是NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # output(None, 224, 224, 3)
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)  # output(None, 227, 227, 3)
    '''默认为 valid卷积，如果不补ZeroPadding2D那层，除不尽结果与论文不符 (w - k + 1) / s'''
    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)  # output(None, 55, 55, 48)
    # 使用 默认-- 核3 步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output(None, 27, 27, 48)
    # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output(None, 13, 13, 128)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)  # output(None, 6, 6, 128)

    # 全连接展平
    x = layers.Flatten()(x)  # output(None, 6*6*128)
    x = layers.Dropout(0.2)(x)
    # 这里原为4096 缩减一半
    x = layers.Dense(2048, activation="relu")(x)  # output(None, 2048)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation="relu")(x)  # output(None, 2048)
    x = layers.Dense(class_num)(x)  # output(None, 5)
    predict = layers.Softmax()(x)

    model = models.Model(inputs=input_image, outputs=predict)
    return model


if __name__ == '__main__':
    net_ = AlexNet()  # Total params: 16,630,440
    print(net_.summary())
