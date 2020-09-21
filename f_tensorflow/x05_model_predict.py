from f_tensorflow.x01_load_data import class_names
from f_tensorflow.x02_data_handler import x_train, y_train, x_test, y_test, train_ds, test_ds
from f_tensorflow.x03_model_def import model
from f_tensorflow.x04_model_fit import 简单的训练及验证

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def f预测01(x_train, y_train, x_test, y_test):
    model = 简单的训练及验证(EPOCHS, x_train, y_train)
    # model = f专家训练01()

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])  # 模型加一层预测
    predictions = probability_model.predict(x_test)
    return predictions


EPOCHS = 1

# 数据
# x_train = x_train[..., tf.newaxis]
# x_test = x_test[..., tf.newaxis]

predictions = f预测01(x_train, y_train, x_test, y_test)
