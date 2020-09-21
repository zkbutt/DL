import os
import random
import sys

import cv2
from keras.utils import np_utils
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory
import numpy as np


class SimilarityGenerator():

    def __init__(self, path, batch_size, img_hw, start_ind=0) -> None:
        '''

        :param path: 大类的路径
        :param batch_size: 每次生成的批量
        :param img_hw: 图片尺寸
        :param start_ind: 暂无作用
        '''
        super().__init__()

        # 大类文件夹名称
        names_type = [a for a in os.listdir(path) if a[0] != '.']  # get folder names
        # 随机打乱正例目录 和反例目录
        self.positive_names_type = names_type
        random.shuffle(self.positive_names_type)  # 直接操作内存不返回

        self.negative_names_type = self.positive_names_type.copy()
        random.shuffle(self.negative_names_type)

        # 需要偶数 除以2也必须是偶数，每个生成器数量为奇数时将无法匹配
        assert (batch_size % 2) == 0
        self.img_hw = img_hw
        self.size_ = batch_size
        self.path = path

        # 大类名称 每次取一个
        n_type = self.positive_names_type.pop()
        # 正例生成器，一次取出4个图片，大多数情况是正例 tensorflow.python.data.ops.iterator_ops.OwnedIterator
        self.train_gen1 = self.cre_datagen(img_hw, n_type, path)

        # 随机产生一个数字，注意是左闭右闭
        n_type = self.negative_names_type[random.randint(0, len(self.negative_names_type) - 1)]
        # 反倒生成器，取一半的图片，及少概率会重复
        self.train_gen2 = self.cre_datagen(img_hw, n_type, path, shuffle=True)

    def cre_datagen(self, img_size, n_type, path, shuffle=False):
        gen = image_dataset_from_directory(
            directory=os.path.join(path, n_type),
            labels='inferred',
            label_mode='int',  # int binary categorical
            batch_size=self.size_,  # 最后两个输入 实际输出只有一半，两个一半刚好 = batch_size
            shuffle=shuffle,
            image_size=img_size)
        # # 获取生成器文件有多少个
        # print('image_dataset_from_directory',
        #       len(gen._input_dataset._datasets[0]._input_dataset._tensors[0]))  # 这个文件夹有共有多少个文件，labels
        # print('image_dataset_from_directory',
        #       gen._input_dataset._datasets[1]._tensors[0].shape[0])
        return gen.__iter__()

    def gen_handler(self, X, labels, Y, ind=0):
        '''

        :param gen:
        :param X:
        :param Y:
        :param ind: 随机反例的y开始索引  修正
        :return:
        '''
        s = X.shape[0]
        # *号拆开数组 临时只有一半
        tx1 = np.empty([s // 2, *self.img_hw, 3])
        tx2 = np.empty([s // 2, *self.img_hw, 3])

        # 两两比较 一次处理两个
        for j in range(s // 2):
            si = j * 2

            # 标签判断
            if labels[si] == labels[(si + 1)]:
                Y[ind + j] = 1
            else:
                Y[ind + j] = 0

            tx1[j] = X[si]
            tx2[j] = X[si + 1]
        return tx1, tx2

    def __next__(self):
        '''一次返回batch_size 个样本[训练集1，训练集2]---list, [验证集]--nparry'''
        try:
            datas = self.train_gen1.__next__()
        except Exception as e:
            # 迭代器完成通过抛异常 try 才能捕获
            print('正例----出错拉', repr(e))
            # 正例没有了如果还有文件夹就重新生成生成器
            if len(self.positive_names_type):  # 有文件夹则取
                # 大类名称 每次取一个
                n_type = self.positive_names_type.pop()
                print('正例文件夹：', self.positive_names_type, n_type)
                # 正例生成器，一次取出4个图片，大多数情况是正例 tensorflow.python.data.ops.iterator_ops.OwnedIterator
                self.train_gen1 = self.cre_datagen(self.img_hw, n_type, self.path)
                datas = self.train_gen1.__next__()
                # 图片已完成 一个文件夹完成
            else:
                raise StopIteration()  # 表示至此停止迭代

        # 动态数量平衡确保批次一样长
        size1 = datas[0].shape[0]  # 数据取生成器出来的实际数量
        __x = datas[0]
        __y = datas[1]
        if size1 < self.size_:  # 没取足，再取一个批次
            if len(self.positive_names_type):  # 如果有文件夹则取
                n_type = self.positive_names_type.pop()
                self.train_gen1 = self.cre_datagen(self.img_hw, n_type, self.path)
                __ds = self.train_gen1.__next__()
                # 新数据添加
                __x = np.concatenate([__x, __ds[0][:]], axis=0)
                __y = np.concatenate([__y, __ds[1][:]], axis=0)
                size1 = __x.shape[0]  # 修正最新的size用于反例时，后面判断

        # 先开内存
        Y_train = np.empty([self.size_])  # 最后的标签数，保持不变

        tx1, tx2 = self.gen_handler(__x, __y, Y_train)
        # X_trains中的数据由于 有可能超界，故通过动态生成
        X_trains = [tx1, tx2]  # 形成 [训练集1，训练集2]

        # ---------------反例生成器----------------------
        try:
            datas = self.train_gen2.__next__()
        except Exception as e:
            # 迭代器完成通过抛异常 try 才能捕获
            print('反例----出错拉', repr(e))
            # 正例没有了如果还有文件夹就重新生成生成器
            n_type = self.negative_names_type[random.randint(0, len(self.negative_names_type) - 1)]
            print('反例文件夹：', n_type)
            # 正例生成器，一次取出4个图片，大多数情况是正例 tensorflow.python.data.ops.iterator_ops.OwnedIterator
            self.train_gen2 = self.cre_datagen(self.img_hw, n_type, self.path)
            # 图片已完成 一个文件夹完成
            datas = self.train_gen2.__next__()

        # 动态数量平衡确保批次一样长
        size2 = datas[0].shape[0]  # 数据取生成器出来的实际数量
        __x = datas[0]
        __y = datas[1]
        if size1 + size2 < self.size_ * 2:  # 整体不足，再取一个批次补齐 *2因为这里两个数据，处理后变成一个
            # 换一个文件夹，确保能超过 self.size_，如果还不够，就发生BUG，只要批次不要大于40，概率较小不想再修复
            n_type = self.negative_names_type[random.randint(0, len(self.negative_names_type) - 1)]
            self.train_gen2 = self.cre_datagen(self.img_hw, n_type, self.path)
            __ds = self.train_gen2.__next__()
            # 添加前面差的 = 取一次的 - 当前的已生成的尺寸
            __x = np.concatenate([__x, __ds[0][:self.size_ * 2 - size1 - size2]], axis=0)
            __y = np.concatenate([__y, __ds[1][:self.size_ * 2 - size1 - size2]], axis=0)
            if size1 + __x.shape[0] < self.size_ * 2:  # 取第二次一定能取满
                __ds = self.train_gen2.__next__()
                # 添加前面差的 = 取一次的 - 当前的已生成的尺寸
                __x = np.concatenate([__x, __ds[0][:self.size_ * 2 - size1 - __x.shape[0]]], axis=0)
                __y = np.concatenate([__y, __ds[1][:self.size_ * 2 - size1 - __y.shape[0]]], axis=0)

        elif size1 + size2 > self.size_ * 2:  # 前面多取了一个批次的正例，这里有可能多
            __x = __x[:self.size_ * 2 - size1]
            __y = __y[:self.size_ * 2 - size1]
            pass  # 不可能发生

        # 这里需对 y 的尺寸进行修正，因为y是提前分配的空间，这里的 data 数是个动态的 有可能不足

        # 这里有可能生成的没有 batch_size 的一半
        tx1, tx2 = self.gen_handler(__x, __y, Y_train, size1 // 2)

        # 这种很消耗算力和内存 应采用Y_train的方式先开内存
        X_trains[0] = np.concatenate([X_trains[0], tx1], axis=0)
        X_trains[1] = np.concatenate([X_trains[1], tx2], axis=0)

        # Y_train = np_utils.to_categorical(Y_train, num_classes=2)  # np转独热

        # 查看结果
        # from matplotlib import pyplot as plt
        # for i in range(min(self.size_, 8)):  # 取几张测试结果
        #     for j in range(2):
        #         plt.subplot(2, 1, j + 1)
        #         plt.imshow(X_trains[j][i], cmap='gray')
        #         plt.title(Y_train[i])
        #     plt.show()

        # for x in X_trains[0]:
        #     # img = X_trains[0][0]
        #     img = x
        #     # img = cv2.imread(img)
        #     cv2.imshow('123', img)
        #     cv2.waitKey()
        print("log---%s-%s---：" % (os.path.basename(__file__), sys._getframe().f_code.co_name),
              X_trains[0].shape,
              Y_train.shape)
        return X_trains, Y_train

    def __iter__(self):
        return self
