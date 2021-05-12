import torch
import torch.nn as nn
import time
import numpy as np

"""
这个脚本是centerNet的三个损失函数
分类损失 Focal loss
校正损失 L1 loss
回归损失 L1 loss
"""


class CenterNetLoss(nn.Module):
    # pred是网络输出结果，包含三个部分（分类信息，校正值和回归值）
    # target是数据集给定的结果，包含两个部分（bbox和分类信息）
    # candidate_num是候选点的个数，文中是100
    def __init__(self, pred=None, target=None, candidate_num=100):
        super(CenterNetLoss, self).__init__()
        # 先获取三个输入
        if (pred == None and target == None):
            self.cls_pred = None
            self.offset_pred = None
            self.size_pred = None
            self.gt_box = None
            self.gt_class = None
        else:
            # [batch, class_num, h, w]
            self.cls_pred = pred[0]
            # [batch, 2, h, w]
            self.offset_pred = pred[1]
            # [batch, 2, h, w]
            self.size_pred = pred[2]

            # 获取两个gt值
            # [batch, num, 4]
            self.gt_box = target[0]
            # [batch, num]
            self.gt_class = target[1]
        # 选出置信度最大的多少个点
        self.candidate = candidate_num
        self.batch_size = 0
        self.mask = None

    # 计算分类得分的gt，就是对应论文中使用高斯公式那部分
    # 我们假定方差都variance都是1
    # 返回cls_gt -> [batch, class_num, h, w]
    def get_cls_gt(self, variance=1.0):
        # [batch, class_num, h, w]
        cls_gt = torch.zeros_like(self.cls_pred)
        # print(cls_gt.shape)
        keypoints = []
        # 根据gt_box和分类标签计算keypoint
        for batch in range(self.batch_size):
            for num in range(self.gt_class.shape[1]):
                if (self.gt_class[batch][num] != -1):
                    # 计算gt_box的中心点坐标
                    center_x = (self.gt_box[batch][num][2] - self.gt_box[batch][num][0]) // 2
                    center_y = (self.gt_box[batch][num][3] - self.gt_box[batch][num][1]) // 2
                    # 进行四倍下采样
                    center_x = center_x // 4
                    center_y = center_y // 4
                    tmp = [batch, self.gt_class[batch][num], center_x, center_y]
                    # print(tmp)
                    keypoints.append(tmp)

        # 根据keypoints计算分类的gt
        for num in range(len(keypoints)):
            batch = keypoints[num][0]
            channel = keypoints[num][1]
            center_x = keypoints[num][2]
            center_y = keypoints[num][3]

            # print("(%d, %d)"%(center_x, center_y))
            cls_gt[batch][channel][center_x][center_y] = 1
            # 周围八个格子
            one_offset = torch.from_numpy(np.array(-1 / (2 * variance)))
            two_offset = torch.from_numpy(np.array(-2 / (2 * variance)))
            one_offset = torch.exp(one_offset)
            two_offset = torch.exp(two_offset)

            if (center_x - 1 >= 0 and center_y - 1 >= 0 and
                    center_x + 1 < cls_gt.shape[2] and center_y + 1 < cls_gt.shape[3]):
                cls_gt[batch][channel][center_x - 1][center_y - 1] = max(two_offset,
                                                                         cls_gt[batch][channel][center_x - 1][
                                                                             center_y - 1])
                cls_gt[batch][channel][center_x - 1][center_y] = max(one_offset,
                                                                     cls_gt[batch][channel][center_x - 1][center_y])
                cls_gt[batch][channel][center_x - 1][center_y + 1] = max(two_offset,
                                                                         cls_gt[batch][channel][center_x - 1][
                                                                             center_y + 1])
                cls_gt[batch][channel][center_x][center_y - 1] = max(one_offset,
                                                                     cls_gt[batch][channel][center_x][center_y - 1])
                cls_gt[batch][channel][center_x][center_y + 1] = max(one_offset,
                                                                     cls_gt[batch][channel][center_x][center_y + 1])
                cls_gt[batch][channel][center_x + 1][center_y - 1] = max(two_offset,
                                                                         cls_gt[batch][channel][center_x + 1][
                                                                             center_y - 1])
                cls_gt[batch][channel][center_x + 1][center_y] = max(one_offset,
                                                                     cls_gt[batch][channel][center_x + 1][center_y])
                cls_gt[batch][channel][center_x + 1][center_y + 1] = max(two_offset,
                                                                         cls_gt[batch][channel][center_x + 1][
                                                                             center_y + 1])

        # print("cls_gt.sum="+str((cls_gt[0]==1).sum()))
        return cls_gt

    # 计算分类损失
    ###############
    # 指数做了修改 #
    ###############

    def FocalLoss1(self, beita=4, alpha=2):
        # st1 = time.time()
        cls_gt = self.get_cls_gt()
        loss = 0
        # 防止梯度爆炸
        self.cls_pred = self.cls_pred.clamp(min=0.0001, max=0.9999)
        # print(self.cls_pred.shape)
        num_pos = self.batch_size * self.cls_pred.shape[2] * self.cls_pred.shape[3]
        for batch in range(self.batch_size):
            mask = (cls_gt[batch] == 1)
            # 当标签值取1的时候
            loss = loss + (((1 - self.cls_pred[batch][mask]) ** beita) * (torch.log(self.cls_pred[batch][mask]))).sum()

            # 当标签值不取1的时候
            loss = loss + (((1 - cls_gt[batch][~mask]) ** beita) * ((self.cls_pred[batch][~mask]) ** alpha) * (
                torch.log(1 - self.cls_pred[batch][~mask]))).sum()

        loss = loss / num_pos
        loss = -loss
        # ed1 = time.time()
        # print("Focal loss->"+str(ed1-st1))
        return loss

    # 选择候选点，选择当前与bbox距离最近的100个候选框
    def getTarget(self):
        target = torch.zeros(self.offset_pred.shape[0], self.offset_pred.shape[2], self.offset_pred.shape[3],
                             dtype=torch.int32)
        target_size = torch.zeros(self.offset_pred.shape[0], self.offset_pred.shape[2], self.offset_pred.shape[3], 2)
        target_offset = torch.zeros(self.offset_pred.shape[0], self.offset_pred.shape[2], self.offset_pred.shape[3], 2)
        for batch in range(self.batch_size):
            # 先计算出gt_box的keypoint
            keypoints = []
            for idx in range(len(self.gt_class[batch])):
                if (self.gt_class[batch][idx] == -1):
                    continue
                center_x = (self.gt_box[batch][idx][2] - self.gt_box[batch][idx][0]) // 2
                center_y = (self.gt_box[batch][idx][3] - self.gt_box[batch][idx][1]) // 2
                width = (self.gt_box[batch][idx][2] - self.gt_box[batch][idx][0]) // 4
                height = (self.gt_box[batch][idx][3] - self.gt_box[batch][idx][1]) // 4
                # print(center_x)
                # print(center_y)
                keypoints.append([self.gt_class[batch][idx], center_x // 4, center_y // 4, width, height, idx])

            for idx in range(len(keypoints)):
                channel = keypoints[idx][0]
                center_x = keypoints[idx][1]
                center_y = keypoints[idx][2]
                width = keypoints[idx][3]
                height = keypoints[idx][4]
                idx_point = keypoints[idx][5]
                # 选择候选框里面置信度最大的100个点进行回归
                min_ = 9999999
                # 指向最小值点的坐标
                min_idx = -1
                coords = []
                num_pos = 0
                radiu = 2
                left = max((center_x - width // 2) // radiu, 0)
                right = min((center_x + width // 2) // radiu, self.cls_pred.shape[2] // radiu)
                top = max((center_y - height // 2) // radiu, 0)
                bottom = min((center_y + height // 2) // radiu, self.cls_pred.shape[3] // radiu)
                for i in range(left, right):
                    for j in range(top, bottom):
                        # print(str(i)+"  "+str(j))
                        if (num_pos < self.candidate):
                            coords.append([i, j, idx_point])
                            if (self.cls_pred[batch][channel][i][j] < min_):
                                min_idx = len(coords) - 1
                                min_ = self.cls_pred[batch][channel][i][j]
                            num_pos = num_pos + 1
                        elif (self.cls_pred[batch][channel][i][j] < min_):
                            # print(len(coords))
                            coords[min_idx][0] = i
                            coords[min_idx][1] = j
                            coords[min_idx][2] = idx_point
                            min_ = self.cls_pred[batch][channel][i][j]
                            # 找到置信度最小的点
                            for k in range(len(coords)):
                                if (self.cls_pred[batch][channel][coords[k][0]][coords[k][1]] < min_):
                                    min_idx = k
                                    min_ = self.cls_pred[batch][channel][coords[k][0]][coords[k][1]]
            for k in range(len(coords)):
                target[batch][coords[k][0]][coords[k][1]] = 1
                # 取相应的box的坐标
                x0, y0, x1, y1 = self.gt_box[batch][coords[k][2]]
                center_x = (x1 - x0) / 2
                center_y = (y1 - y0) / 2
                center_x_ = center_x / 4
                center_y_ = center_y / 4
                _center_x = center_x // 4
                _center_y = center_y // 4

                target_size[batch][coords[k][0]][coords[k][1]][0] = x1 - x0
                target_size[batch][coords[k][0]][coords[k][1]][1] = y1 - y0
                target_offset[batch][coords[k][0]][coords[k][1]][0] = center_x_ - _center_x
                target_offset[batch][coords[k][0]][coords[k][1]][1] = center_y_ - _center_y
        return target, target_size, target_offset

    # 计算校正损失
    def offset_size_Loss(self):
        mask, target_size, target_offset = self.getTarget()
        mask = mask.long()
        self.mask = mask
        num_pos = mask.sum()
        # print(num_pos)
        pred_offset = self.offset_pred.permute(0, 2, 3, 1)
        pred_size = self.size_pred.permute(0, 2, 3, 1)
        offset_loss = 0.0
        size_loss = 0.0
        for batch in range(self.batch_size):
            for i in range(mask.shape[1]):
                for j in range(mask.shape[2]):
                    if (mask[batch][i][j] == 1):
                        offset_loss = offset_loss + abs(pred_offset[batch][i][j][0] - target_offset[batch][i][j][0])
                        offset_loss = offset_loss + abs(pred_offset[batch][i][j][1] - target_offset[batch][i][j][1])
                        size_loss = size_loss + abs(pred_size[batch][i][j][0] - target_size[batch][i][j][0])
                        size_loss = size_loss + abs(pred_size[batch][i][j][1] - target_size[batch][i][j][1])
        offset_loss = offset_loss / num_pos
        size_loss = size_loss / num_pos
        return offset_loss, size_loss

    def forward(self, pred, target, nameda_cls=1, nameda_size=0.1, nameda_offset=1):
        # [batch, class_num, h, w]
        self.cls_pred = pred[0]  # torch.Size([1, 20, 128, 128])
        # [batch, 2, h, w]
        self.offset_pred = pred[1]  # torch.Size([1, 2, 128, 128])
        # [batch, 2, h, w]
        self.size_pred = pred[2]  # torch.Size([1, 2, 128, 128])
        # 获取两个gt值
        # [batch, num, 4]
        self.gt_box = target[0]
        # [batch, num]
        self.gt_class = target[1]
        self.batch_size = self.cls_pred.shape[0]
        st = time.time()

        offset_loss, size_loss = self.offset_size_Loss()
        # size_loss有时候会非常大，必须做一下限制
        # size_loss = size_loss.clamp(max=5000)
        ed = time.time()
        # print("l1 loss->%.4f s" % (ed - st))
        cls_loss = self.FocalLoss1()
        ed = time.time()
        # print("focal loss->%.4f s" % (ed - st))

        # print("offset loss: %.4f size loss: %.4f cls loss:%.4f"%(offset_loss, size_loss, cls_loss))

        loss = nameda_cls * cls_loss + nameda_size * size_loss + nameda_offset * offset_loss
        # loss = loss/self.batch_size
        end = time.time()
        cost = end - st
        # print("cost time:%.4f s"%(cost))
        return loss


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = dataload.VOCDetection(readInfo=False, mode="train")
    batch = 4
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True,
                                               collate_fn=dataset.collate_fn, pin_memory=True, num_workers=8)
    image = None
    gt_box = None
    gt_class = None
    for data in train_loader:
        image = data[0].to(device)
        gt_box = data[1]
        gt_class = data[2]

        target = [gt_box, gt_class]
        centerNet = network.CenterNet().to(device)

        cls_pred, offset_pred, size_pred = centerNet(image)

        print("image->" + str(image.shape))
        print("box->" + str(gt_box.shape))
        print("class->" + str(gt_class.shape))

        pred = [cls_pred, offset_pred, size_pred]
        loss = CenterNetLoss()
        st = time.time()
        print(loss(pred, target))
        ed = time.time()
        print("cost time:%.4f s" % (ed - st))
