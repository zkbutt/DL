import math

import os

import torch
from torch import optim

from f_tools.GLOBAL_LOG import flog
from f_tools.f_torch_tools import load_weight
from f_tools.fits.f_lossfun import LossYOLOv3
from f_tools.fits.f_lr_fun import f_lr_cos
from f_tools.fun_od.f_anc import FAnchors
from object_detection.f_yolov3.process_fun import init_model, data_loader, train_eval
from object_detection.f_yolov3.CONFIG_YOLO3 import CFG

'''
多尺度训练 multi-scale
 0:15:02

'''

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(CFG.PATH_SAVE_WEIGHT):
        try:
            os.makedirs(CFG.PATH_SAVE_WEIGHT)
        except Exception as e:
            flog.error(' %s %s', CFG.PATH_SAVE_WEIGHT, e)
    CFG.SAVE_FILE_NAME = os.path.basename(__file__)

    device = torch.device('cuda:%s' % 0 if torch.cuda.is_available() else "cpu")
    flog.info('模型当前设备 %s', device)

    if CFG.DEBUG:
        # device = torch.device("cpu")
        CFG.PRINT_FREQ = 1
        CFG.PATH_SAVE_WEIGHT = None
        CFG.BATCH_SIZE = 5
        CFG.DATA_NUM_WORKERS = 0
        pass
    else:
        torch.multiprocessing.set_sharing_strategy('file_system')  # 多进程开文件

    # CFG.FILE_FIT_WEIGHT = None

    # 预设尺寸必须是下采样倍数的整数倍 输入一般是正方形
    down_sample = CFG.FEATURE_MAP_STEPS[-1]  # 模型下采样倍数
    assert math.fmod(CFG.IMAGE_SIZE[0], down_sample) == 0, "尺寸 %s must be a %s 的倍数" % (CFG.IMAGE_SIZE, small_conf)
    # assert math.fmod(CFG.IMAGE_SIZE[1], down_sample) == 0, "尺寸 %s must be a %s 的倍数" % (CFG.IMAGE_SIZE, small_conf)

    if CFG.IS_MULTI_SCALE:
        # 动态输入尺寸选定 根据预设尺寸  0.667~1.5 之间 满足32的倍数
        imgsz_min = CFG.IMAGE_SIZE[0] // CFG.MULTI_SCALE_VAL[1]
        imgsz_max = CFG.IMAGE_SIZE[0] // CFG.MULTI_SCALE_VAL[0]
        # 将给定的最大，最小输入尺寸向下调整到32的整数倍
        grid_min, grid_max = imgsz_min // down_sample, imgsz_max // down_sample
        imgsz_min, imgsz_max = int(grid_min * down_sample), int(grid_max * down_sample)
        sizes_in = []
        for i in range(imgsz_min, imgsz_max + 1, down_sample):
            sizes_in.append(i)
        # imgsz_train = imgsz_max  # initialize with max size
        # img_size = random.randrange(grid_min, grid_max + 1) * gs
        flog.info("输入画像的尺寸范围为[{}, {}] 可选尺寸为{}".format(imgsz_min, imgsz_max, sizes_in))

    '''------------------模型定义---------------------'''
    model = init_model(CFG)

    model.to(device)  # 这个不需要加等号
    model.train()

    anc_obj = FAnchors(CFG.IMAGE_SIZE, CFG.ANC_SCALE, CFG.FEATURE_MAP_STEPS, CFG.ANCHORS_CLIP, device=device)
    # anchors = anchors.to(device)
    losser = LossYOLOv3(CFG.NUM_CLASSES, anc_obj.ancs)

    '''对模型进行冻结定义, 取出需要优化的的参数'''
    # if epoch < 5:
    #     # 主干网一般要冻结
    #     for param in model.body.parameters():
    #         param.requires_grad = False
    # else:
    #     # 解冻后训练
    #     for param in model.body.parameters():
    #         param.requires_grad = True
    pg = model.parameters()
    # pg = [p for p in model.parameters() if p.requires_grad]

    # 最初学习率
    lr0 = 1e-3
    lrf = lr0 / 100
    optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.937, weight_decay=0.0005, nesterov=True)
    start_epoch = load_weight(CFG.FILE_FIT_WEIGHT, model, optimizer, None, device)
    lr_scheduler = f_lr_cos(optimizer, start_epoch, CFG.END_EPOCH, lrf_scale=0.01)

    # '''---------------数据加载及处理--------------'''
    loader_train, loader_val, _ = data_loader(CFG, is_mgpu=False)

    flog.debug('---训练开始---epoch %s', start_epoch + 1)
    # 有些参数可通过模型来携带  model.nc = nc
    train_eval(cfg=CFG, start_epoch=start_epoch, model=model, anc_obj=anc_obj,
               losser=losser, optimizer=optimizer, lr_scheduler=lr_scheduler,
               loader_train=loader_train, loader_val=loader_val, device=device,
               )

    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
