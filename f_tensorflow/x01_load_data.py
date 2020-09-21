import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

'''手写数字:  (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)图像是28x28 NumPy数组，像素值范围是0到255。 标签是整数数组，范围是0到9'''
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

'''服装图像:  (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)图像是28x28 NumPy数组，像素值范围是0到255。 标签是整数数组，范围是0到9'''
# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# # T恤/上衣,裤子,拉过来,连衣裙,涂层,凉鞋,衬衫,运动鞋,袋,脚踝靴,
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# x_train, y_train, x_test, y_test = train_images, train_labels, test_images, test_labels



import pandas as pd

dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()
dataset = dataset.dropna()
# 固定输出数据
x_train, y_train, x_test, y_test = [i for i in range(4)]


def show_pic(pic):
    plt.figure()
    plt.imshow(pic)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def show_pics(train_images, train_labels, class_names):
    plt.figure(figsize=(10, 10))
    for i, (train_image, train_label) in enumerate(zip(train_images, train_labels)):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # plt.imshow(train_image, cmap=plt.cm.binary) # 黑白显示
        plt.imshow(train_image)
        plt.xlabel(class_names[train_label])
    plt.show()


if __name__ == '__main__':
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    show_pic(x_train[0])
    # train_= x_train[0]
    # show_pic(train_.reshape((28, 28, 1)))

    # show_pics(x_train[:25], y_train[:25], class_names)
