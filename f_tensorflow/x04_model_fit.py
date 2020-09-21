from tensorflow import keras
from tensorflow.keras import layers

from f_tensorflow.x02_data_handler import x_train, y_train, x_test, y_test, train_ds, test_ds
from f_tensorflow.x03_model_def import model, loss_object, train_loss, optimizer, train_accuracy, test_loss, \
    test_accuracy
import tensorflow as tf
import numpy as np


@tf.function


def train_step(images, labels, history, training=False):
    with tf.GradientTape() as tape:
        predictions = model(images, training=training)
        tf.debugging.assert_equal(predictions.shape, (32, 10))  # asserts 检测输出
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)  # 计算梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 反向传播更新参数
    loss2 = tf.keras.metrics.Mean(name='train_loss')(loss)
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')(labels, predictions)
    history.append([loss.numpy().mean(),loss2,accuracy])


@tf.function
def test_step(images, labels):
    predictions = model(images)  # 装载图片
    t_loss = loss_object(labels, predictions)  # 计算损失
    test_loss(t_loss)
    test_accuracy(labels, predictions)


def 简单的训练及验证(epochs, x_train, y_train):
    model.fit(x_train, y_train, epochs=epochs)
    # test_loss = model.evaluate(x_test, y_test, verbose=2)
    # 通常输出一个loss,当定义metrics时,会再计算一个指标
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(test_loss, test_acc)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # 自定义名称对应
    # tf.keras.losses.BinaryCrossentropy(
    #     from_logits=True, name='binary_crossentropy'),
    # 'accuracy'])
    # tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    board = tf.keras.callbacks.TensorBoard(logdir / name)
    history = model.fit(
        x_train, y_train,
        epochs=epochs, validation_split=0.2, verbose=0,
        callbacks=[early_stop, PrintDot()])
    return model


def f专家训练01(epochs):
    for epoch in range(epochs):
        # 在下一个epoch开始时，重置评估指标
        # 固定格式 针对损失器metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:  # 数据生成器
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):  # 自定义 callback
        if epoch % 100 == 0: print('')
        print('.', end='')
