import os
import numpy as np
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # # 全局使用CPU配置
b = np.random.random((2, 2))
a = tf.constant([[1, 2], [3, 4]])
l = [a]
print(a, b)
l[0] = np.concatenate([l[0], b[:]], axis=0)
print(type(l[0]))
print(l)
print(a.numpy())

b = tf.Variable([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
