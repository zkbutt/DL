from tensorflow.keras import layers
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils.vis_utils import plot_model

from f_tools.nets.yolo_3.f_Darknet53 import Darknet53, Conv2D_BN_Leaky
from yolo_3.utils.utils import compose


def cre_last_layers(x, num_filters, out_filters):
    # 五次卷积 保存尺寸不变
    x = Conv2D_BN_Leaky(num_filters, (1, 1))(x)  # 降维
    x = Conv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = Conv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = Conv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = Conv2D_BN_Leaky(num_filters, (1, 1))(x)

    # 将最后的通道数调整为outfilter
    y = Conv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    y = layers.Conv2D(filters=out_filters, kernel_size=(1, 1), )(y)
    return x, y


def yolo3_body(inputs, num_anchors, num_classes):
    feat1, feat2, feat3 = Darknet53(inputs)
    darknet53 = Model(inputs, feat3)

    x, y1 = cre_last_layers(darknet53.output, 512, num_anchors * (num_classes + 5))
    x = compose(
        Conv2D_BN_Leaky(256, (1, 1)),
        layers.UpSampling2D(2))(x)
    x = layers.Concatenate()([x, feat2])  # 堆叠

    x, y2 = cre_last_layers(x, 256, num_anchors * (num_classes + 5))
    x = compose(
        Conv2D_BN_Leaky(128, (1, 1)),
        layers.UpSampling2D(2))(x)
    x = layers.Concatenate()([x, feat1])  # 堆叠

    x, y3 = cre_last_layers(x, 128, num_anchors * (num_classes + 5))
    return Model(inputs, [y1, y2, y3])  # 返回最终模型


if __name__ == '__main__':
    inputs = layers.Input([416, 416, 3])
    model = yolo3_body(inputs, 3, 20)
    print(model.summary())
    plot_model(model, to_file='yolo3_body.png', show_shapes=True)  # 'TB'表示方向
    # 显示
    # plt.figure(figsize=(10, 10))
    # img = plt.imread('yolo3_body.png')
    # plt.imshow(img)

    # 保存模型
    model.save("./yolo3_body.hdf5")
    del model
    model = load_model("./model.hdf5")
