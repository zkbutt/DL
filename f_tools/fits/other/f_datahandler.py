import numpy as np


# ---------------------------------#
#   l1标准化
# ---------------------------------#
def l1_normalize(x, epsilon=1e-10):
    np_sum = np.sum(np.abs(x))
    maximum = np.maximum(np_sum, epsilon)
    output = x / maximum
    return output


# ---------------------------------#
#   l2标准化
# ---------------------------------#
def l2_normalize(x, axis=-1, epsilon=1e-10):
    square = np.square(x)
    np_sum = np.sum(square, axis=axis, keepdims=True)
    maximum = np.maximum(np_sum, epsilon)
    output = x / np.sqrt(maximum)
    return output


if __name__ == '__main__':
    a = np.arange(12).reshape([1, 3, 4])
    l1 = l1_normalize(a)
    l2 = l2_normalize(a, (1, 2))
    print(a, "\n")
    print(l1, "\n")
    print(l2, "\n")
