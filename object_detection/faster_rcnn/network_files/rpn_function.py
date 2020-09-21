import torch
import torchvision
from torch.nn import functional as F
from torch import nn
from object_detection.faster_rcnn.network_files import boxes as box_ops
from object_detection.faster_rcnn.network_files import det_utils
from torch.jit.annotations import List, Optional, Dict, Tuple
from torch import Tensor
from object_detection.faster_rcnn.network_files.image_list import ImageList


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # type: (Tensor, int) -> Tuple[int, int]
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    # TODO : remove cast to IntTensor/num_anchors.dtype when
    #        ONNX Runtime version is updated with ReduceMin int64 support
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0).to(torch.int32)).to(num_anchors.dtype)

    return num_anchors, pre_nms_top_n


class AnchorsGenerator(nn.Module):
    """
    anchors生成器
    """
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        '''
        resnet50 值
        :param sizes: ((32,), (64,), (128,), (256,), (512,))
        :param aspect_ratios:  已进行copy处理 ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        '''
        super(AnchorsGenerator, self).__init__()
        if not isinstance(sizes[0], (list, tuple)):
            # TODO 加一层,统一格式 当用于与多个特征图输入对应
            # ((32,), (64,), (128,), (256,), (512,))
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)  # 自动与前面的括号匹配
        assert len(sizes) == len(aspect_ratios)  # 确保一样长(list两层), 且只有一个元素
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}  # 用于缓存 anchors

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device="cpu"):
        '''
        计算 anchors 的尺寸 一次处理一个
        :param scales: (32,)
        :param aspect_ratios: aspect_ratios:(0.5, 1.0, 2.0)
        :param dtype: float32
        :param device: cpu/gpu
        :return:返回 3维(多个特图对应)或2维数组
        '''
        # type: (List[int], List[float], int, Device)
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        # 这里是标准算法
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios
        # 生成x个anchors的中点 多维交叉相乘 3个扩2维 1个扩2维 拉平 np高级
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # 这里调整以（0, 0）为中心的形成  left-top, right-bottom 两个点
        # 宽高合并2维调为4维 将数组中每一个值按 1 维 进行组合(列组合)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()  # 四舍五入

    def set_cell_anchors(self, dtype, device):
        '''
        新产生的数据保持变量的一致性

        :param dtype:
        :param device:
        :return:
            self.cell_anchors 用于保存一套 anchors
        '''
        # type: (int, Device) -> None
        # 已有直接返回
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        # sizes ((32,), (64,), (128,), (256,), (512,))
        # aspect_ratios ((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
        # anchors模板都是以(0, 0)为中心的anchor
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors  # 暂时感觉可以用普通变量,可能后面会用到, 通过类变量保存 每一个特图对应的 anchors 模块[torch.Size([3, 4]),]

    def num_anchors_per_location(self):
        # 计算每个特图对应的 anchors 数量 返回list
        num_anchors = [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]
        return num_anchors

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes, strides):
        '''
        生成对应原图的所有 anchors 坐标 左上右下
        :param grid_sizes: 每一个特图的尺寸
        :param strides: 对应的步距
        :return:
        '''
        # type: (List[List[int]], List[List[Tensor]])
        anchors = []
        cell_anchors = self.cell_anchors  # 每个特图对应的一套
        assert cell_anchors is not None

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            # shape: [grid_width] 对应原图上的列坐标 np高级
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # shape: [grid_height] 对应原图上的行坐标
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 计算预测特征矩阵上每个点对应原图上的坐标(anchors模板的坐标偏移量)
            # torch.meshgrid 函数分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            # shape: [grid_height, grid_width]
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
            # shape: [grid_width * grid_height, 4]
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # shifts 矩阵的每一个点(每一行偏移)需要叠加9个 anchors...形成 (base anchor, output anchor) ,
            # offset each zero-centered base anchor by the center of the output anchor.
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))  # 拉平形成 (grid_width * grid_height * len(anchor),4)

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        '''
        缓存生成的 anchors
        :param grid_sizes: 每一个特图的尺寸
        :param strides: 对应的步距
        :return: 输出 List[Tensor] 每一个特图对应的所有 anchors torch.Size([182400, 4]) torch.Size([741, 4])
            特图的越大 anchors 越多
        '''
        # type: (List[List[int]], List[List[Tensor]])
        """将计算得到的所有anchors信息进行缓存"""
        key = str(grid_sizes) + str(strides)  # 用于缓存 下批有相同的 grid_sizes 和 strides 时复用
        # self._cache是字典类型
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        # self._cache[key] = anchors  # 缓存这个缓存不要
        return anchors

    def forward(self, image_list, feature_maps):
        '''
        np高级 维度运算有点难
        :param image_list: 预处理后的图片对象(最大批) 属性是tensors 有几张图就有几个 和预处理层统一尺寸前的尺寸
        :param feature_maps: 特图s : 特图数-图片数
        :return:
            每一个图对应的所有anchors (每一个图 * 每一图的特图,4)  List[Tensor]-torch.Size([369303, 4])
        '''
        # type: (ImageList, List[Tensor])
        # 取每一个特图 H,W
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # 每一个图像图片tensor(最大批) H,W 一批图片大小是一样的
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device  # 取类型和设备

        # 这一批图片与每一个特图的步长  , 计算每一个特图与最大批的步长 最大批尺寸 / 多个特图的尺寸 自动扩展
        strides = [[torch.tensor(image_size[0] / g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] / g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        # 根据提供的 sizes 和 aspect_ratios 生成anchors模板
        self.set_cell_anchors(dtype, device)  # 存到 self.cell_anchors 中

        # -------应用到原图------
        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # List[Tensor] 每一个特图对应的所有 anchors 这里是真实位置
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        # anchors_over_all_feature_maps 是每一个特图上所有框的真实尺寸(不含每一个图) 通常特图越大框越多
        # 下面组装结构 形成 , 将每一个图 - 每一图的特图 - 对应anchors
        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        # List[List[Tensor]]
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []  # 每一个图对应的所有特征图的 anchors
            # 遍历List[Tensor] 每一个特图对应的所有 anchors
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # 将每一个图 - 每一图的特图 - 对应anchors  拉平 (每一个图 * 每一图的特图,4)
        # 形成anchors是个list，是每个一张图像的所有anchors信息 拉平
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.
        self._cache.clear()  # 这里要优化 才创建就删除不合理,应该按容量进行操作
        return anchors


class RPNHead(nn.Module):
    '''
    输出rpn的预测结果
    '''

    def __init__(self, in_channels, num_anchors):
        '''
        特征图预测
        :param in_channels: backbone的输出维度
        :param num_anchors: 预测k个 使用交叉熵值 h*w*9
        '''
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口 不同的模型输入不一样 some卷积
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 输出anchors的类型 计算预测的目标概率（这里的目标只是指前景或者背景） 这里只生成k个是采用交叉熵损失 生成0或1
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 输出anchors的偏移 计算预测的目标bbox regression参数  中心坐标xy 和h w
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        # 参数初始化
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        '''
        特图已从 有序字典转换为list(n,特图的3维) n表示图片
        :param x: 特图 List[Tensor]s  每个4维(有可能一个图片有多个特图) torch.Size([2, 256, 200, 304])
        :return: 网络输出 改变的是 C
            特图的每一个点对应的 anchors模块个 前后景分数 torch.Size([2, 3*1, 200, 304])
            特图的每一个点对应的 anchors模块个 预测偏差 torch.Size([2, 3*4, 200, 304])
        '''
        # type: (List[Tensor])
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):  # 遍历每一层(多个特征层) - 对应一批原图
            t = F.relu(self.conv(feature))  # 激活后分两路 1号取特图-对应一批图 torch.Size([2, 256, 272, 272])
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    '''

    :param layer: 预测特征层上预测的目标概率或bboxes regression参数
    :param N: batch_size
    :param A: anchors_num_per_position
    :param C: classes_num or 4(bbox coordinate)
    :param H: height
    :param W: width
    :return:
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]
    '''
    # type: (Tensor, int, int, int, int, int)
    # view和reshape功能是一样的，先展平所有元素在按照给定shape排列
    # view 函数只能用于内存中连续存储的tensor，permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
    # reshape则不需要依赖目标tensor是否在内存中是连续的
    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer = layer.view(N, -1, C, H, W)
    # 调换tensor维度
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    '''
    这里np高级 有点难  --- 就是将预测结果拉平
    每一个 anchors 的得分 和 每一个 anchors 类别 list(torch.Size([2, 3, 50, 76]),)
    对box_cla和box_regression两个list中的每个预测特征层的预测信息
    的tensor排列顺序以及shape进行调整 -> [N, -1, C]

    :param box_cls:  每一个特图list(batch,anc数,特图尺寸)4维 的分数 只区分前后背只需一维参数
    :param box_regression:  每一个特图list(batch,anc数*4,特图尺寸) anc框的偏差需要4个
    :return: 返回拉平成2维的结果
        与anchors统一  torch.Size([2, 3*1, 200, 304]) -> [N, H, W, -1, C] -> (N, -1, C)
    '''
    # type: (List[Tensor], List[Tensor])
    box_cls_flattened = []  # 目标分数调整后 list([N, -1, C])
    box_regression_flattened = []  # anc调整后

    # 遍历每个预测特征层
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # 遍历每一个特图  [batch_size, anchors_num_per_position * classes_num, height, width]
        # 注意，当计算RPN中的proposal时，classes_num=1,只区分目标和背景
        N, AxC, H, W = box_cls_per_level.shape  # rpn只有2个类 1个参数
        # # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]  # 框 要4个参
        # anchors_num_per_position
        A = Ax4 // 4
        # classes_num
        C = AxC // A

        # [N, -1, C] 把分数移到后面一维
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N, -1, C] 把框移到最后一维
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    # list([N, -1, C]) -> [N, -1(特图叠加), C]  list打开 在总anc个数维连接 相关于把特图的所有anc拉平
    cat = torch.cat(box_cls_flattened, dim=1)
    box_cls = cat.flatten(0, -2)  # start_dim, end_dim 实际就是0-1维合并 合并后超多 (nn,1)
    # 同样的进行拉平
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    '''
    这是个主干类 调度其它
    '''
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # use during training
        # 计算anchors与真实bbox的iou
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )

        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        '''
        计算每个anchors最匹配的gt，并划分为正样本，背景以及废弃的样本
        :param anchors:
        :param targets: 每一个图对应的 目标类
        :return:
            labels（1, 0, -1分别对应正样本，背景，废弃的样本）
            matched_gt_boxes：与anchors匹配的gt
        '''
        # type: (List[Tensor], List[Dict[str, Tensor]])
        labels = []
        matched_gt_boxes = []
        # 遍历一个图
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:  # 返回所有元素数目 如果没有目标 则 类别和回归全为0 不计损失
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # 计算anchors与真实bbox的iou信息
                # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)  # 返回(n,m)
                # 计算每个anchors与gt匹配iou最大的索引 （如果iou<0.3索引置为-1， 0.3<iou<0.7索引为-2）
                # 返回 det_utils.Matcher 这个图的 anchors 类别
                matched_idxs = self.proposal_matcher(match_quality_matrix)  # 输入每个gt对应所有 anchors 标记类别分值
                # ---通过分数和 正负样本阀值 确定index  剔除 -1 和-2 GT框扩展用于算回归
                _clamp = matched_idxs.clamp(min=0)
                matched_gt_boxes_per_image = gt_boxes[_clamp]  # torch.Size([5, 4]) 选很多次 torch.Size([242991, 4])

                labels_per_image = matched_idxs >= 0  # >0的 GT框扩展用于算回归
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = torch.tensor(0.0, device=labels_per_image.device)

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = torch.tensor(-1.0, device=labels_per_image.device)

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        '''
        根据前景概率 选出 top参超
        :param objectness: anchors 每个图片对应的 anchors 前景分
        :param num_anchors_per_level: 每个特图的 anchors 数量
        :return: 每个图片对应的 anchors 筛选后
        '''
        # type: (Tensor, List[int])
        r = []  # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0
        # 遍历每个预测特征层上的预测目标概率信息
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():  # 训练时是不会满足的 训练和预测进行不同的处理
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]  # 预测特征层上的预测的anchors个数
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)  # self.pre_nms_top_n=1000

            # Returns the k largest elements of the given input tensor along a given dimension
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)  # values 及 indices topk
            r.append(top_n_idx + offset)  # 这里的top需要加上前面 的特图的偏移 特图尺寸不一样
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        '''
        boxes赛选策略
        :param proposals: torch.Size([2, 242991, 4]) 每一个图对应的 所有 proposals = anchors + 修正
        :param objectness: 每个 anchors 对应的类型分数
        :param image_shapes: 预处理后的图片尺寸
        :param num_anchors_per_level: 每一个特图对应的 anchors 个数
        :return:
        '''
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int])
        num_images = proposals.shape[0]
        device = proposals.device

        # 去掉原有的梯度信息
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # anchors已拉平 levels 负责记录分隔不同特图对应的 anchors 索引信息
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)  # 保存每一个 anchors (个数不一样) 对应的特图索引 [0,0...0,1,1..1,...]

        # 通过 对应的特图索引 组合 anchors 分数 区别anchors是属于哪个分量
        levels = levels.reshape(1, -1).expand_as(objectness)

        # 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]

        # 根据预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        # 预测概率排前pre_nms_top_n的anchors索引值获取相应bbox坐标信息
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            '''
            遍历每一张图片
            proposals: torch.Size([2, 8741, 4]) 拉平的建议框
            objectness: torch.Size([2, 8741])  拉平的分数
            levels: torch.Size([2, 8741]) 对应特图的索引
            image_shapes: 每个输入的预处理图的尺寸  <class 'list'>: [(800, 1066), (800, 1201)]
            '''
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            # 返回boxes满足宽，高都大于min_size的索引
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        '''
        计算RPN损失，包括类别损失（前景与背景），bbox regression损失
        :param objectness:  预测的前景概率
        :param pred_bbox_deltas:  预测的bbox regression
        :param labels:  真实的标签 1, 0, -1（batch中每一张图片的labels对应List的一个元素中）
        :param regression_targets: 真实的bbox regression
        :return:
            objectness_loss (Tensor) : 类别损失
            box_loss (Tensor)：边界框回归损失
        '''
        # type: (Tensor, Tensor, List[Tensor], List[Tensor])
        # 按照给定的batch_size_per_image, positive_fraction选择正负样本
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 将一个batch中的所有正负样本List(Tensor)分别拼接在一起，并获取非零位置的索引
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        # 将所有正负样本索引拼接在一起
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算边界框回归损失
        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # 计算目标预测概率损失
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self, images, features, targets=None):
        '''
        ---------------RPN都在这里调度--------------------
        :param images: 预处理后的图片对象(最大批)
        :param features: 经有序字典包装的
        :param targets: 预处理后的目标
        :return:
        '''
        # type: (ImageList, Dict[str, Tensor], Optional[List[Dict[str, Tensor]]])
        features = list(features.values())  # 有序字典转换

        # 计算每个预测特征层上的预测目标概率和bboxes regression参数
        # objectness 每一个特图list(batch,anc数,特图尺寸)4维 的分数 只区分前后背只需一维参数
        # pred_bbox_deltas 每一个特图list(batch,anc数*4,特图尺寸) anc框的偏差需要4个
        objectness, pred_bbox_deltas = self.head(features)

        # AnchorsGenerator 生成一个(batch,特图数*特图尺寸*anc个,4) anchors 信息 返回图对应的 所有 anchors 拉平
        anchors = self.anchor_generator(images, features)

        '''-----------将预测出的回归参数 与类别调整成与 anchors 对齐'''
        # 取batch_size
        num_images = len(anchors)

        # 每个预测特征层上list(batch,anc数,特图尺寸)4维的对应的anchors数量 = 特图尺寸 anc数 * h * w(特图尺寸)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]  # 取每一个特图的c h w
        # 拉平objectness的后三维 [291840, 72960, 18240, 4560, 1140]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # 调整 head 输出的每一个 anchors 的得分 和 每一个 anchors 类别
        # 与anchors统一  torch.Size([2, 3*1, 200, 304]) -> [N, H, W, -1, C] -> (N, -1, C)
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        # 将预测的bbox regression参数应用到anchors上得到最终预测bbox坐标
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)  # 最终形成(n,特图*anc,4) 所有anchors 的框

        # 筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            # 计算每个图anchors最匹配的gt，并将anchors进行分类，前景，背景以及废弃的anchors
            # anchors 每一个图对应的anchors; 返回labels每一个anchors对应的类别 ,
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 结合anchors以及对应的gt，计算regression参数
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        return boxes, losses


if __name__ == '__main__':
    size = [[16, 32], [64, 128], [256, 512]]
    ratios = [8, 16, 32]
    generator = AnchorsGenerator(
        sizes=size,
        aspect_ratios=ratios
    )
    anchors = generator.generate_anchors(size, ratios)
    print(anchors)
