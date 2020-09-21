import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation  ## 全连接层
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt


def 结果分析(history):
    # plot loss and accuracy image
    history_dict = history.history  # 拿到完成训练的结果
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]
    # figure 1
    plt.figure()
    plt.plot(range(epoch), train_loss, label='train_loss')
    plt.plot(range(epoch), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # figure 2
    plt.figure()
    plt.plot(range(epoch), train_accuracy, label='train_accuracy')
    plt.plot(range(epoch), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    epoch = 2
    batch_size = 32

    # 获取训练集
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # 首先进行标准化
    X_train = X_train.reshape(X_train.shape[0], -1) / 255
    X_test = X_test.reshape(X_test.shape[0], -1) / 255
    # 计算categorical_crossentropy需要对分类结果进行categorical
    # 即需要将标签转化为形如(nb_samples, nb_classes)的二值序列
    Y_train = np_utils.to_categorical(Y_train, num_classes=10)  # np转独热
    Y_test = np_utils.to_categorical(Y_test, num_classes=10)

    # 构建模型
    model = Sequential([
        Dense(32, input_dim=784),
        Activation("relu"),
        Dense(10),
        Activation("softmax")
    ]
    )

    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0)

    ## compile
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

    print("\ntraining")
    history = model.fit(
        X_train, Y_train, epochs=epoch, batch_size=batch_size,
        validation_data=(X_test, Y_test),
    )

    结果分析(history)

    print("\nTest")
    cost, accuracy = model.evaluate(X_test, Y_test)
    ## W,b = model.layers[0].get_weights()
    print("accuracy:", accuracy)
