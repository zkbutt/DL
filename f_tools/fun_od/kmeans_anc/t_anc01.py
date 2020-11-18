sizes = [
    [0.822, 0.85333333],
    [0.574, 0.426],
    [0.392, 0.736],

    [0.214, 0.512],
    [0.26, 0.25066667],
    [0.128, 0.10933333],

    [0.116, 0.304],
    [0.06, 0.16266667],
    [0.038, 0.06133333],
]

import numpy as np

array = np.array(sizes)
a = array[:, 0] + array[:, 1]
# indexs = np.argsort(a)  # 默认升序
indexs = np.argsort(-a)  # 降序
print((array * [416, 416])[indexs])
