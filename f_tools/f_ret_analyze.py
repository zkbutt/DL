import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ConfusionMatrix:
    def __init__(self, num_class: int, labels: list) -> None:
        super().__init__()
        self.matrix = np.zeros((num_class, num_class))
        self.labels = labels
        self.num_class = num_class

    def update(self, preds, labels):
        # 这个标签是从0开始
        for p, t in zip(preds, labels):
            # p是混阵的行,t为列
            self.matrix[t, p] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_class):
            # 矩阵对角线之和
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("正确率 ", acc)

        # 对每个类别统计 精确 召回 特异
        df = pd.DataFrame(None, columns=['Precision', 'Recall', 'Specificity'])

        for i in range(self.num_class):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            _t = round(TP / (TP + FP), 3)
            precision = _t if not np.isnan(_t) else 0

            _t = round(TP / (TP + FN), 3)
            recall = _t if not np.isnan(_t) else 0

            _t = round(TP / (TP + FN), 3)
            specificity = _t if not np.isnan(_t) else 0
            df = df.append({'Precision': precision, 'Recall': recall, 'Specificity': specificity},
                           ignore_index=True)
        print(df)

    def polt(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_class), self.labels, rotation=45)
        plt.yticks(range(self.num_class), self.labels)
        plt.colorbar()

        thresh = matrix.max() / 2
        for x in range(self.num_class):
            for y in range(self.num_class):
                # 原点在左上角  y对应是行标
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color='white' if info > thresh else 'black'
                         )
        # plt.tight_layout()  # 显示紧凑??未发现作用
        plt.show()


if __name__ == '__main__':
    matrix = ConfusionMatrix(num_class=5, labels=['a', 'b', 'c', 'd', 'e'])
    # 在训练或验证时进行更新
    matrix.update([1, 2, 1, 2], [3, 2, 4, 3])  # 这个标签是从0开始
    matrix.update([1, 1, 1, 1], [2, 1, 1, 1])
    matrix.summary()
    matrix.polt()
