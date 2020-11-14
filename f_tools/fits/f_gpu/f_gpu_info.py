import os
import torch
import time

from f_tools.GLOBAL_LOG import flog
'''f_multi_gpu'''

#
# def tf_time_matmul(x, number):
#     import tensorflow as tf
#     start = time.time()
#     # 10加矩阵相乘
#     for loop in range(number):
#         tf.matmul(x, x)
#     result = time.time() - start
#     print("10 loops: {:0.2f}ms".format(1000 * result))
#
#
# def tf_cpu_vs_gpu(shap=(1000, 1000), number=100):
#     # Force execution on CPU
#     print("On CPU:")
#     with tf.device("CPU:0"):
#         x = tf.random.uniform(shap)
#         assert x.device.endswith("CPU:0")
#         tf_time_matmul(x, number)
#
#     # Force execution on GPU #0 if available
#     if tf.config.experimental.list_physical_devices("GPU"):
#         print("On GPU:")
#         with tf.device("GPU:0"):  # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
#             x = tf.random.uniform(shap)
#             assert x.device.endswith("GPU:0")
#             tf_time_matmul(x, number)
#
#     x_gpu0 = x.gpu()
#     x_cpu = x.cpu()
#     _ = tf.matmul(x_cpu, x_cpu)  # Runs on CPU
#     _ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0
#
#
# # cpu_vs_gpu()
# def tensorflow():
#     from tensorflow.python.client import device_lib
#     print("查看可用的GPU列表: ", tf.config.experimental.list_physical_devices("GPU")),
#     print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
#
#     print("Is the Tensor on GPU #0:  "),
#     print(x.device.endswith('GPU:0'))
#     print('查看可用运算设备', device_lib.list_local_devices())
#     print(tf.test.is_built_with_cuda())
#
#     # 查看正在使用的GPU
#     print(tf.__version__)
#     if tf.test.gpu_device_name():
#         print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
#     else:
#         print("Please install GPU version of TF")
#
#     # 查看可用运算设备
#     print('查看可用运算设备', device_lib.list_local_devices())
#
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU  0开始
#     # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
#     # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # # 全局使用CPU配置
#     print('测试 tf 是不是运行在GPU', tf.test.is_gpu_available())


def pytorch():
    # 必须单独操作
    print('torch.__version__ 版本', torch.__version__)
    print('返回Ture GPU安装成功', torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('可用gpu数量', torch.cuda.device_count())
    print('输入索引，返回gpu名字', torch.cuda.get_device_name(0))
    print('返回当前设备索引', torch.cuda.current_device())
    print('返回当前空存使用量', torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

    # 切换gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.Tensor(2, 3).to(torch.device('cuda:0'))
    net = torch.nn.DataParallel(model, device_ids=[0])

    print('输入索引，返回gpu名字', torch.cuda.get_device_name(0))
    print('返回当前设备索引', torch.cuda.current_device())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    t = t.to(device)
    print(t.device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


def tf_set():
    # 配置GPU以实际内存消耗占用
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            exit(-1)


if __name__ == '__main__':
    flog.debug(' %s', os.cpu_count())  # 显示CPU进程数
