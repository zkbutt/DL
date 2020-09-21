from tensorflow.keras import layers, models, Model, Sequential

from f_tools.f_dl_tools import compose


def cre_layers(cfg):
    ls = []
    for v in cfg:
        if v == "M":
            ls.append(layers.MaxPool2D(pool_size=2))  # 默认 核2 步长2 尺寸减半 除不尽默认向下取整
        else:
            conv2d = layers.Conv2D(v, kernel_size=3, padding="same", activation="relu")
            ls.append(conv2d)
    return compose(*ls)


def VGG(cfg, im_height, im_width, class_num):
    '''

    '''
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    # tensorflow中的tensor通道排序是NHWC
    x = cre_layers(cfg)(input_image)

    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(class_num)(x)
    output = layers.Softmax()(x)

    model = models.Model(inputs=input_image, outputs=output)
    return model


cfgs = {
    'vgg_test': [64, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", im_height=224, im_width=224, class_num=1000):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(cfg, im_height, im_width, class_num)
    return model


if __name__ == '__main__':
    net = vgg(model_name="vgg_test")
    print(net.summary())
