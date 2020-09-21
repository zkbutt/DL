from f_tensorflow.x01_load_data import class_names
from f_tensorflow.x02_data_handler import y_test, x_test
from f_tensorflow.x05_model_predict import predictions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''----------------结果分析----------------'''


def f_show_plot_images(num_rows, num_cols, predictions, y_test, x_test):
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        _plot_image(i, predictions[i], y_test, x_test)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        _plot_value_array(i, predictions[i], y_test)
    plt.tight_layout()
    plt.show()


def f_show_plot_image(index, predictions, y_test, x_test):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    _plot_image(index, predictions[index], y_test, x_test)
    plt.subplot(1, 2, 2)
    _plot_value_array(index, predictions[index], y_test)
    plt.show()


def _plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    img = img.reshape(28, 28)
    plt.imshow(img, cmap=plt.cm.binary)  # 这个只支持28,28

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'  # 正确
    else:
        color = 'red'
    # 显示预测概率,及预测与真实的名称
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def _plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")  # 直方图查看各个类型的概率
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_history(history):
    '''
    显示回归mse  mae 的训练与测试损失对比图
    :param history:
    :return:
    '''
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],  # mae
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],  # mse
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


index = 0

# 单个预测
print(predictions[index])
print(np.argmax(predictions[index]))
print(y_test[index])

f_show_plot_image(index, predictions, y_test, x_test)
index = 12
f_show_plot_image(index, predictions, y_test, x_test)
num_rows = 5
num_cols = 3
f_show_plot_images(num_rows, num_cols, predictions, y_test, x_test)


# 显示最后几条训练的loss
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plot_history(history)