import os

import tensorflow as tf
import matplotlib.pyplot as plt

from f_tools.GLOBAL_LOG import flog
from f_tools.datas.data_handler import spilt_voc2txt


def f数据加载():
    '''手写数字:  (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)图像是28x28 NumPy数组，像素值范围是0到255。 标签是整数数组，范围是0到9'''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


def f数据处理(x_train, y_train, x_test, y_test):
    # 归一化
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # 变成4维向量
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return x_train, y_train, x_test, y_test


def f模型定义_精细化():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # 图像展平,二维数组（28 x 28像素）转换为一维数组（28 * 28 = 784像素
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    return model, loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy,


# @tf.function
def _train_step(images, labels, model, history_train,
                optimizer, loss_object,
                train_loss=None, train_accuracy=None, training=False):
    with tf.GradientTape() as tape:
        predictions = model(images, training=training)
        tf.debugging.assert_equal(predictions.shape, (32, 10))  # asserts 检测输出
        loss = loss_object(labels, predictions)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)  # 计算层参数的梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 反向传播更新参数
    train_loss(loss) if train_loss else None
    train_accuracy(labels, predictions) if train_accuracy else None
    history_train.append(loss.numpy().mean())


# @tf.function
def _test_step(images, labels, model,
               loss_object, test_loss=None, test_accuracy=None, ):
    predictions = model(images)  # 装载图片
    t_loss = loss_object(labels, predictions)  # 计算损失
    test_loss(t_loss) if test_loss else None
    test_accuracy(labels, predictions) if test_accuracy else None


def f_fit_精细化(train_ds, test_ds, epochs, model,
              loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy):
    history = []
    history_train = []

    # 定义 TensorBoard     通过 tensorboard --logdir=./tensorboard 启动
    log_dir = './tensorboard'
    summary_writer = tf.summary.create_file_writer(log_dir)
    # pip install -U tensorboard-plugin-profile  -i https://pypi.tuna.tsinghua.edu.cn/simple
    tf.summary.trace_on(graph=True, profiler=True)  # 这个只需要一次, 开启Trace，可以记录图结构和profile信息

    for epoch in range(1, epochs + 1):
        # 在下一个epoch开始时，重置评估指标
        # 固定格式 针对损失器metrics 需要重置
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for images, labels in train_ds:  # 数据生成器
            _train_step(images, labels, model, history_train,
                        optimizer, loss_object,
                        train_loss=train_loss, train_accuracy=train_accuracy, training=True)

        for test_images, test_labels in test_ds:
            _test_step(test_images, test_labels, model,
                       loss_object, test_loss, test_accuracy)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        history.append([train_loss.result().numpy(),
                        train_accuracy.result().numpy() * 100,
                        test_loss.result().numpy(),
                        test_accuracy.result().numpy() * 100])

        with summary_writer.as_default():  # 希望使用的记录器
            tf.summary.scalar("loss", test_loss.result().numpy(), step=epoch)
            tf.summary.scalar("f_acc", test_accuracy.result().numpy(), step=epoch)  # 还可以添加其他自定义的变量
            # tf.summary.trace_export(name="model_trace", step=epoch, profiler_outdir=log_dir)  # 保存Trace信息到文件
    return history, history_train


def _grad(model, loss, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def f_tf_fit(train_dataset):
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 32
        for x, y in train_dataset:
            # 优化模型
            loss_value, grads = _grad(model, epoch_loss_avg, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 追踪进度
            epoch_loss_avg(loss_value)  # 添加当前的 batch loss
            # 比较预测标签与真实标签
            epoch_accuracy(y, model(x))

        # 循环结束
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
    return train_loss_results, train_accuracy_results


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = loss_object(labels, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, output)


# @tf.function
def _test_step(images, labels):
    output = model(images, training=False)
    t_loss = loss_object(labels, output)

    test_loss(t_loss)
    test_accuracy(labels, output)


def f_tf_fit01(train_data_gen, val_data_gen, epochs, total_train, total_val, batch_size):
    best_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        test_loss.reset_states()  # clear history info
        test_accuracy.reset_states()  # clear history info

        # train
        for step in range(total_train // batch_size):
            images, labels = next(train_data_gen)
            train_step(images, labels)

            # print train process
            rate = (step + 1) / (total_train // batch_size)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            acc = train_accuracy.result().numpy()
            print("\r[{}]train acc: {:^3.0f}%[{}->{}]{:.4f}".format(epoch, int(rate * 100), a, b, acc), end="")
        print()

        # validate
        for step in range(total_val // batch_size):
            test_images, test_labels = next(val_data_gen)
            _test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        if test_loss.result() < best_test_loss:
            best_test_loss = test_loss.result()
            model.save_weights("./save_weights/resMobileNetV2.ckpt", save_format="tf")


def show_loss(history_train):
    '''显示每 batch_size 的loss变化情况'''
    plt.plot(history_train)
    plt.xlabel('Batch #')
    plt.ylabel('Loss [entropy]')
    plt.show()


def show_loss2(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def f多GPU并行训练(dataset, num_epochs):
    batch_size_per_replica = 64
    '''
    无参数表示 识别所有GPU 可以指定设备间通讯的方式
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        tf.distribute.NcclAllReduce  # 默认
        tf.distribute.HierarchicalCopyAllReduce
    '''
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    # strategy = tf.distribute.OneDeviceStrategy("/cpu:0")

    print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    with strategy.scope():  # 多GPU
        model = tf.keras.applications.MobileNetV2(weights=None, classes=2)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.sparse_categorical_accuracy]
        )
    model.fit(dataset, epochs=num_epochs)


if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU  0开始

    from tensorflow.python.client import device_lib

    flog.debug('查看可用运算设备 %s', device_lib.list_local_devices())

    if tf.test.gpu_device_name():
        flog.info('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        flog.info("Please install GPU version of TF")

    if tf.test.is_gpu_available():
        flog.info("当前是GPU模式 , 应用的GPU列表: %s", tf.config.experimental.list_physical_devices("GPU")),
    else:
        flog.info("-------当前是CPU模式--------")

    '''------------------全局变量---------------------'''
    # 取当前文件名
    save_path = os.path.join(os.path.abspath('.'), os.path.basename(__file__) + '.h5')
    DATA_ROOT = 'M:\datas\m_VOC2007'

    '''------------------数据加载及处理---------------------'''
    path_datas = os.path.join(DATA_ROOT, 'trainval')
    file_name_train, file_name_val = spilt_voc2txt(path_datas)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_list, train_label_list))


    # '''------------------数据加载---------------------'''
    # x_train, y_train, x_test, y_test = f数据加载()
    #
    # '''------------------数据分析---------------------'''
    #
    # '''------------------数据处理---------------------'''
    # x_train, y_train, x_test, y_test = f数据处理(x_train, y_train, x_test, y_test)
    # # 生成数据生成器
    # batch_size = 32
    # train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    # test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    #
    # '''------------------模型定义---------------------'''
    #
    # model, loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy, = f模型定义_精细化()
    #
    # '''------------------模型训练---------------------'''
    # epochs = 3
    # # print(tf.executing_eagerly())
    # # f多GPU并行训练()
    #
    # history, history_train = f_fit_精细化(train_ds, test_ds, epochs, model,
    #                                    loss_object, optimizer, train_loss, train_accuracy, test_loss, test_accuracy)
    # print(history)
    #
    # '''------------------结果分析---------------------'''
    # show_loss(history_train)
    # show_loss2(history_train)
    #
    # '''------------------结果保存---------------------'''
    # path_model = "/tmp/file_model"
    # model.save(path_model)
    # restored_keras_model = tf.keras.models.load_model(path_model)  # 恢复 无需再次调用compile()
