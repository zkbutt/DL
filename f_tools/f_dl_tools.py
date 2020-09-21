def calc_oshape_pytorch(w, k, s=1, p=0, ):
    '''
    向下取整
    :param w: 输入尺寸
    :param k:核尺寸
    :param p:padding
    :param s:步长
    :return:
    '''
    return int((w - k + 2 * p) / s + 1)


def calc_oshape_tf(w, k, s=1, p='valid'):
    '''
    卷积向上取整，池化向下取整
    p='valid' 'same'
    :return:
    '''
    if p == 'valid':
        return (w - k + 1) / s
    else:
        return w / s


def calc_argnum(m, n, k=0, mode='CNN', bias=True, bn=False):
    '''
    假设卷积核的大小为 k*k, 输入channel为M， 输出channel为N
    bias为True时:k×k×M×N + N
    bias为False时：k×k×M×N
    当使用BN时，model.add(BatchNormalization())  # 输入维度*4
    model:FC   CNN
    :return:
    '''
    if mode == 'CNN':
        if bn:
            return k * k * m * n + 3 * n
        if bias:
            return k * k * m * n + n
        else:
            return k * k * m * n
    elif mode == 'FC':
        if bias:
            return m * n + n
        else:
            return m * n
    pass


def calc_complexity_cnn(h, w, c, k, s, ph, pw, h2, w2, o):
    '''
    一个样本所需的计算量
    '''
    print('参数个数为：%s' % calc_argnum(m=c, n=o, k=k))

    t1 = 2 * k * k - 1  # 一次卷积的计算量
    t2 = int((h - k + ph) / s + 1) * int((w - k + pw) / s + 1)  # 一个所需特征图所需卷积的次数
    # return b * o * (c - 1) * c * t2 * t1
    c_ = o * ((c - 1) * h2 * w2 + c * t2 * t1 * c)
    print('单个样本前向计算量为：%s' % c_)
    return c_


def calc_complexity_fc(m, n):
    '''
    一个样本所需的计算量
    '''
    print('参数个数为：%s' % calc_argnum(m=m, n=n, mode='FC'))

    c_ = 2 * m + n
    print('单个样本计算量为：%s' % c_)
    return c_


def compose(*funcs):
    from functools import reduce
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def calc_IOU(RecA, RecB):
    xA = max(RecA[0], RecB[0])
    yA = max(RecA[1], RecB[1])
    xB = min(RecA[2], RecB[2])
    yB = min(RecA[3], RecB[3])
    # 计算交集部分面积
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # 计算预测值和真实值的面积
    RecA_Area = (RecA[2] - RecA[0] + 1) * (RecA[3] - RecA[1] + 1)
    RecB_Area = (RecB[2] - RecB[0] + 1) * (RecB[3] - RecB[1] + 1)
    # 计算IOU
    iou = interArea / float(RecA_Area + RecB_Area - interArea)

    return iou


def __t_calc_IOU():
    import cv2
    import numpy as np
    img = np.zeros((512, 512, 3), np.uint8)
    img.fill(255)
    RecA = [50, 50, 300, 300]
    RecB = [60, 60, 320, 320]
    cv2.rectangle(img, (RecA[0], RecA[1]), (RecA[2], RecA[3]), (0, 255, 0), 10)
    cv2.rectangle(img, (RecB[0], RecB[1]), (RecB[2], RecB[3]), (255, 0, 0), 5)
    IOU = calc_IOU(RecA, RecB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "IOU = %.2f" % IOU, (130, 190), font, 0.8, (0, 0, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # print(calc_oshape_tf(w=606, k=7, s=2))
    # print(calc_oshape_tf(w=300, k=2, s=2, p='same'))
    # print(calc_oshape_pytorch(w=416, k=7, s=2, p=3))
    print(calc_oshape_pytorch(w=208, k=3, s=1, p=0))
    # print(calc_oshape_tf(224, 7, s=2, p='same'))
    # print(calc_oshape_tf(299, 3, s=2))
    # __t_calc_IOU()
