import torch
from torch import nn
from collections import OrderedDict
from object_detection.faster_rcnn.network_files.rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
from object_detection.faster_rcnn.network_files.roi_head import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor
import torch.nn.functional as F
import warnings
from object_detection.faster_rcnn.network_files.transform import GeneralizedRCNNTransform


class FasterRCNNBase(nn.Module):
    '''
    这是网络的整体组装结构
    '''

    def __init__(self, backbone, rpn, roi_heads, transform):
        '''

        :param backbone:
        :param rpn:
        :param roi_heads:
        :param transform: 预处理方法
        '''
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        '''
        ---这是有空行啊模型核心类---
        :param images: List[Tensor]
        :param targets:训练时必须 List[Dict[str, Tensor]]]
        :return:
        '''
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        # 训练的必要的判断--- targets 验证
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:  # 训练模式targets检查
            assert targets is not None
            for target in targets:  # 进一步判断传入的target的boxes参数是否符合规定
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # 创建一个指定类型的变量 存按min max预处理后的尺寸 一个批量有多个 用于后面批量缩放后恢复
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]  # c,h,w
            assert len(val) == 2  # 防止输入的是个一维向量
            original_image_sizes.append((val[0], val[1]))
        # original_image_sizes = [img.shape[-2:] for img in images]

        # 通过 GeneralizedRCNNTransform 对图像进行归一化 , 并统一缩放到指定尺寸 再按批确定尺寸
        # 返回 images是个对象 包含 image_sizes原图尺寸数组, 4维 tensors ;   targets 结构不变
        images, targets = self.transform(images, targets)
        # 将统一大小的4维图像输入backbone,  得到4个特征图 4维向量 对应4个图片的特图 (n,c,h,w)
        # 输出4个特图 对应有序字典,序是特图号 尺寸由大->小,每个特图有4张图片批量个图 4维向量 (n,c,h,w)
        features = self.backbone(images.tensors)
        # ---有些模型返回的不是有序字典 例如是单层 保持结构的统一---
        if isinstance(features, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
            features = OrderedDict([('0', features)])  # 若在多层特征层上预测，传入的就是一个有序字典

        # 将特征层以及标注target信息传入rpn中  ---最难的---RegionProposalNetwork
        proposals, proposal_losses = self.rpn(images, features, targets)  # 输出建议框和损失

        # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 对网络的预测结果进行后处理（将bboxes 还原到原图像尺度上）
        # original_image_sizes是未处理尺寸
        # images.image_sizes 是批量resize前的尺寸
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():  # 这里是生产环境部署
            # 不用依赖python环境 并优化编译成 静态
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

        # if self.training:
        #     return losses
        #
        # return detections


class FasterRCNN(FasterRCNNBase):
    '''
    实现类 定义一系列参数 默认参数 支持resnet50
    '''

    def __init__(self, backbone, num_classes=None,  # 这里训练用的91
                 # transform parameter
                 min_size=800, max_size=1344,  # 预处理resize时限制的最小尺寸与最大尺寸
                 image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差

                 # RPN parameters 初选框  rpn_anchor_generator  自动生成resnet50
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,  # rpn中在nms处理前保留的proposal数(根据score)
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
                 rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
                 # RPN 损失计算
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn计算损失时，正负例阈值
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时采样的样本数，正样本比例

                 # rpn后面的三层
                 box_roi_pool=None,  # 统一尺寸
                 box_head=None,  # 全连接
                 box_predictor=None,  # 出结果

                 # 终选参数
                 box_score_thresh=0.05,  # 移除低目标分数
                 box_nms_thresh=0.5,  # fast rcnn中进行nms处理的阈值
                 box_detections_per_img=100,  # 对预测结果根据score排序取前100个目标
                 # 终选损失
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,  # 计算损失时，正负样本设置的阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # 计算误差时采样的样本数，以及正样本占所有样本的比例
                 bbox_reg_weights=None,
                 ):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels"
            )

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # 预测特征层的channels
        out_channels = backbone.out_channels

        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        if rpn_anchor_generator is None:  # resnet50进入
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)  # 特别预处理 让比例COPY5份对应size
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        # 生成RPN通过滑动窗口预测网络部分
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels,  # backbone 输出
                rpn_anchor_generator.num_anchors_per_location()[0]  # 计算每个特图对应的 anchors 数量 返回list
            )

        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        #  创建ROI层
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                output_size=[7, 7],
                sampling_ratio=2)

        # fast RCNN中roi pooling后的展平处理两个全连接层部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认等于7
            representation_size = 1024
            # 网络最终预测
            box_head = TwoMLPHead(out_channels * resolution ** 2,
                                  representation_size
                                  )

        # 在box_head的输出上预测部分
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        # 将roi pooling, box_head以及box_predictor结合在一起
        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class TwoMLPHead(nn.Module):
    '''
    展平后接两个全连接输出
    '''

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


if __name__ == '__main__':
    pass
