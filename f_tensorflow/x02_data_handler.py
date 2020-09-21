import tensorflow as tf

from f_tensorflow.x01_load_data import x_train, y_train, x_test, y_test

# 归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# 变成4维向量
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 生成数据生成器
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)