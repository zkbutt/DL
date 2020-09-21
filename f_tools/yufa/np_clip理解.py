import numpy as np
'''
a：输入矩阵；
a_min：被限定的最小值，所有比a_min小的数都会强制变为a_min；
a_max：被限定的最大值，所有比a_max大的数都会强制变为a_max；
out：可以指定输出矩阵的对象，shape与a相同
'''
# 一维矩阵
x = np.arange(12)
print(np.clip(x, 3, 8))
# 多维矩阵
y = np.arange(12).reshape(3, 4)
print(np.clip(y, 3, 8))
