import numpy as np


def 一维向量测试():
    x = np.arange(12)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]
    y = np.arange(12, 24)  # [12 13 14 15 16 17 18 19 20 21 22 23]
    # 一维向量测试
    print(x, '\n', y)
    # 0  1 。。。10 11 同等级比较，每个找最大的索引为 11
    print("np.argmax(x):", np.argmax(x))
    # 直接接在最后
    print("np.concatenate([x, y]):", np.concatenate([x, y]))


def 二维():
    x = np.arange(12)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]
    y = np.arange(12, 24)  # [12 13 14 15 16 17 18 19 20 21 22 23]
    x = x.reshape(3, 4)
    y = y.reshape(3, 4)
    # 二维向量测试
    # 0代表对行进行最大值选取，此时对每一列进行操作
    # -----x-----
    # [[ 0  1  2  3]
    #  [ 4  5  6  7]
    #  [ 8  9 10 11]]
    # -----y-----
    #  [[12 13 14 15]
    #  [16 17 18 19]
    #  [20 21 22 23]]
    print(x, '\n', y)
    # [ 0  1  2  3]
    # [ 4  5  6  7]
    # [ 8  9 10 11]
    # 找独立的元素对应比，输出 [ 2 2 2 2]
    print("np.argmax(x):", np.argmax(x, axis=0))
    #  0  1  2  3
    #  4  5  6  7
    #  8  9 10 11
    # 每个元素是一个整，输出
    # [3 3 3]
    print("np.argmax(x):", np.argmax(x, axis=1))
    #  [ 0  1  2  3]
    #  [ 4  5  6  7]
    #  [ 8  9 10 11]
    #  [12 13 14 15]
    #  [16 17 18 19]
    #  [20 21 22 23]
    print("np.concatenate([x, y]):", np.concatenate([x, y], axis=0))
    #  [ 0  1  2  3 12 13 14 15]
    #  [ 4  5  6  7 16 17 18 19]
    #  [ 8  9 10 11 20 21 22 23]
    print("np.concatenate([x, y]):", np.concatenate([x, y], axis=1))


def 三维():
    x = np.arange(12).reshape((2, 2, 3))
    x[1, 0, 2] = 1
    y = np.arange(12, 24).reshape((2, 2, 3))
    # -----x-----
    # [[[ 0  1  2]
    #   [ 3  4  5]]
    #  [[ 6  7  1]
    #   [ 9 10 11]]]
    # -----y-----
    #  [[[12 13 14]
    #   [15 16 17]]
    #  [[18 19 20]
    #   [21 22 23]]]
    print(x, x.shape, '\n', y, y.shape)
    # [[1 1 0]
    #  [1 1 1]]
    print("np.argmax(x):", np.argmax(x, axis=0))
    x[1, 0, 2] = 12
    # [[1,1,1][1,1,0]]
    print("np.argmax(x):", np.argmax(x, axis=1))
    x[1, 0, 1] = 99
    # [2,2][1,2]
    print("np.argmax(x):", np.argmax(x, axis=2))
    # [[[ 0  1  2]
    #   [ 3  4  5]]
    #  [[ 6  7  1]
    #   [ 9 10 11]]
    #  [[12 13 14]
    #   [15 16 17]]
    #  [[18 19 20]
    #   [21 22 23]]]
    concatenate = np.concatenate([x, y], axis=0)
    print("np.concatenate([x, y])   axis=0:", concatenate, concatenate.shape)

    # 拆括号找独立对象，对应
    # [[[ 0  1  2]
    #   [ 3  4  5]
    #   [12 13 14]
    #   [15 16 17]]
    #  [[ 6  7  1]
    #   [ 9 10 11]
    #   [18 19 20]
    #   [21 22 23]]]
    concatenate = np.concatenate([x, y], axis=1)
    print("np.concatenate([x, y])   axis=1:", concatenate, concatenate.shape)
    # [[[ 0  1  2  12 13 14]
    #   [ 3  4  5  15 16 17]]
    #  [[ 6  7  1  18 19 20]
    #   [ 9 10 11  21 22 23]]
    concatenate = np.concatenate([x, y], axis=2)
    print("np.concatenate([x, y])  axis=2 :", concatenate, concatenate.shape)

    def 遍历():
        print('\n-------------shape[2]-------------')
        for i in range(x.shape[2]):
            # 取两行两列
            print('--------------------------')
            print(x[:, :, i])
        print('\n-------------shape[1]-------------')
        for i in range(x.shape[1]):
            # 进二维，取每个类的第i行，三列
            print('--------------------------')
            print(x[:, i, :])
        print('\n------------shape[0]--------------')
        for i in range(x.shape[0]):
            # 进一维，一次取两行三列
            print('--------------------------')
            print(x[i, :, :])


def stack_():
    d1 = np.arange(9).reshape(3, 3)
    d2 = np.arange(3, 12).reshape(3, 3)
    print(d1.shape)
    # 数组中元素的维度 (3,3) 直接堆  -> [0,:,:]... (2,3,3)
    stack = np.stack([d1, d2])
    # 数组中元素的维度 (3,3) 则取[:,0,:],[:,1,:] 和 [:,2,:] -> (3,2,3)
    ''' 先判断出结果 shape
         1,2,3
         3,4,5
    
         5,6,7
         6,7,8
         '''
    stack = np.stack([d1, d2], axis=1)
    # 数组中元素的维度 (3,3) 则取[:,:,0],[:,:,1] 和 [:,:,2] -> (3,3,2)
    ''' 先判断出结果 shape
        1,3,
        4,6,
        7,9
        ...取三次
    
        '''
    stack = np.stack([d1, d2], axis=2)
    print(stack.shape)
    print(stack)


def 假升维():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    print(np.concatenate((a, b), axis=0))  # 假升维
    # array([[1, 2],
    #        [3, 4],
    #        [5, 6]])
    b = np.array([[5], [6]])
    print(np.concatenate((a, b), axis=1))  # 2,2   2,1 --> 2,3 [:,0]
    # array([[1, 2, 5],
    #        [3, 4, 6]])


if __name__ == '__main__':
    # 一维向量测试()

    # 二维()

    # 三维()

    x = np.arange(12).reshape((2, 2, 3))
    y = np.arange(12, 28).reshape((2, 2, 4))
    print(x, x.shape, '\n', y, y.shape)
    concatenate = np.concatenate([x, y], axis=2)
    print("np.concatenate([x, y])  axis=2 :", concatenate, concatenate.shape)

    # x = x.reshape((2, 3, 2))  # 12
    # y = y.reshape((2, 4, 2))  # 16
    # print(x, x.shape, '\n', y, y.shape)
    # concatenate = np.concatenate([x, y], axis=1)
    # print("np.concatenate([x, y])  axis=2 :", concatenate, concatenate.shape)

    # 遍历()

    # x = np.arange(12).reshape((2, 2, 3))
    # print(x)
    # print(x.shape)
    # x1 = np.arange(12, 24).reshape((2, 2, 3))
    # x2 = np.append(x, x1, axis=1)  # 将x1按维度向x添加
    # x2 = np.insert(x, obj=1, values=x1, axis=1)  # 指定位置添加
    # x2 = np.delete(x, obj=1, axis=1)
    # print(x2)
    # print(x2.shape)

    # stack_()

    # 假升维()
