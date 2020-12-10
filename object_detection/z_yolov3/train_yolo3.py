import math
import os
import sys

'''用户命令行启动'''
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from object_detection.z_yolov3.process_fun import init_model, train_eval, data_loader

from f_tools.fits.f_fit_fun import init_od, base_set
from object_detection.z_yolov3.CONFIG_YOLO3 import CFG

from f_tools.GLOBAL_LOG import flog

'''
多尺度训练 multi-scale
 0:15:02
python /home/win10_sys/tmp/DL/object_detection/f_yolov3/train_yolo3.py
'''

if __name__ == '__main__':
    '''------------------系统配置---------------------'''
    # -----------通用系统配置----------------
    init_od()
    device, cfg = base_set(CFG)

    if cfg.IS_MOSAIC:
        cfg.IMAGE_SIZE = [512, 512]

    if hasattr(cfg, 'FEATURE_MAP_STEPS'):
        # 预设尺寸必须是下采样倍数的整数倍 输入一般是正方形
        down_sample = cfg.FEATURE_MAP_STEPS[-1]  # 模型下采样倍数
        assert math.fmod(cfg.IMAGE_SIZE[0], down_sample) == 0, "尺寸 %s must be a %s 的倍数" % (cfg.IMAGE_SIZE, down_sample)
        # assert math.fmod(cfg.IMAGE_SIZE[1], down_sample) == 0, "尺寸 %s must be a %s 的倍数" % (cfg.IMAGE_SIZE, small_conf)

        '''-----多尺度训练-----'''
        if CFG.IS_MULTI_SCALE:
            # 动态输入尺寸选定 根据预设尺寸  0.667~1.5 之间 满足32的倍数
            imgsz_min = cfg.IMAGE_SIZE[0] // CFG.MULTI_SCALE_VAL[1]
            imgsz_max = cfg.IMAGE_SIZE[0] // CFG.MULTI_SCALE_VAL[0]
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
    model, optimizer, lr_scheduler, start_epoch = init_model(cfg, device, id_gpu=None)

    # '''---------------数据加载及处理--------------'''
    loader_train, loader_val_fmap, loader_val_coco, train_sampler, eval_sampler = data_loader(model.cfg, is_mgpu=False)

    flog.debug('---训练开始---epoch %s', start_epoch + 1)
    # 有些参数可通过模型来携带  model.nc = nc
    train_eval(start_epoch=start_epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
               loader_train=loader_train, loader_val_fmap=loader_val_fmap, loader_val_coco=loader_val_coco,
               device=device, train_sampler=None, eval_sampler=None, tb_writer=None,
               )
    flog.info('---%s--main执行完成------ ', os.path.basename(__file__))
