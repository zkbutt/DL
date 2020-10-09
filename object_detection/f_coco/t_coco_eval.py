from f_tools.datas.data_factory import VOCDataSet
from object_detection.f_coco.coco_eval import CocoEvaluator
from object_detection.f_coco.convert_data.voc_dataset2coco_obj import voc2coco


def prepare_for_coco_detection(self, predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue


if __name__ == '__main__':
    VOC_root = 'M:/AI/datas/VOC2012/trainval'
    file_name = ['train.txt', 'val.txt']

    train_data_set = VOCDataSet(
        VOC_root,
        file_name[0],  # 正式训练要改这里
        bbox2one=True,
        isdebug=False
    )
    # 拿到 coco_gt
    coco_gt = voc2coco(train_data_set)
    coco_evaluator = CocoEvaluator(coco_gt, iou_types)


