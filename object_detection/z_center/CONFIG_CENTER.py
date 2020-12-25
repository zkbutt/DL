import socket


class CFG:
    DEBUG = False
    IS_FORCE_SAVE = False
    NUM_KEYPOINTS = 0  # 关键点数, 0为没有 不能和 IS_MOSAIC 一起用

    IS_LOCK_BACKBONE_WEIGHT = True  # 锁定 BACKBONE_WEIGHT keypoints 不能使用
    # BATCH_SIZE = 132  # batch过小需要设置连续前传
    BATCH_SIZE = 1  # batch过小需要设置连续前传
    # BATCH_SIZE = 36  # batch过小需要设置连续前传
    FORWARD_COUNT = 1  # 连续前传次数 accumulate = max(round(64 / CFG.BATCH_SIZE), 1)
    PRINT_FREQ = int(400 / BATCH_SIZE)  # 400张图打印
    END_EPOCH = 200

    IS_TRAIN = True
    IS_COCO_EVAL = False
    IS_FMAP_EVAL = False  # FMAP只运行生成一次

    # 调参区
    LOSS_WEIGHT = [1, 1., 1., 1.]  # l_box_p,l_conf_p,l_cls_p,l_conf_n
    LR = 1e-3
    # LR = 1e-1

    # import getpass
    # # 获取当前系统用户名
    # user_name = getpass.getuser()
    # # 获取当前系统用户目录
    # user_home = os.path.expanduser('~')
    host_name = socket.gethostname()
    if host_name == 'Feadre-NB':
        PATH_HOST = 'M:'
        # raise Exception('当前主机: %s 及主数据路径: %s ' % (host_name, cfg.PATH_HOST))
    elif host_name == 'e2680v2':
        PATH_HOST = ''
    # PATH_HOST = 'M:'  # 416
    # PATH_HOST = '/home/bak3t/bak299g'  # 需要尾部加/

    '''训练参数'''
    SYSNC_BN = False  # 不冻结时可使用多设备同步BN,速度减慢

    '''可视化'''
    IS_VISUAL = False
    IS_VISUAL_PRETREATMENT = False  # 图片预处理

    '''样本及预处理'''
    DATA_NUM_WORKERS = 8
    IS_KEEP_SCALE = True  # 数据处理保持长宽
    IS_MOSAIC = False
    IS_MOSAIC_KEEP_WH = False  # IS_MOSAIC 是主开关 直接拉伸
    IS_MOSAIC_FILL = True  # IS_MOSAIC 使用 是IS_MOSAIC_KEEP_WH 副形状
    # PATH_DATA_ROOT = PATH_ROOT + '/AI/datas/widerface/coco'
    PATH_DATA_ROOT = PATH_HOST + '/AI/datas/VOC2012'
    PATH_PROJECT_ROOT = PATH_HOST + '/AI/temp/tmp_pycharm/DL/object_detection/z_center'  # 这个要改

    PATH_TENSORBOARD = 'runs_voc'
    # 训练
    PATH_COCO_TARGET_TRAIN = PATH_DATA_ROOT + '/coco/annotations'
    PATH_IMG_TRAIN = PATH_DATA_ROOT + '/trainval/JPEGImages'
    # 验证
    PATH_COCO_TARGET_EVAL = PATH_DATA_ROOT + '/coco/annotations'
    PATH_IMG_EVAL = PATH_DATA_ROOT + '/test/JPEGImages'
    # fmap
    PATH_EVAL_IMGS = PATH_HOST + r'/AI/datas/VOC2012/test/JPEGImages'
    PATH_EVAL_INFO = PATH_HOST + r'/AI/datas/VOC2012/f_map'  # dt会自动创建

    IMAGE_SIZE = (512, 512)  # wh 预处理 统一尺寸
    # IMAGE_SIZE = (1024, 1024)  # wh 预处理 统一尺寸
    NUM_CLASSES = 20  # 模型分类数 人脸只有1 0 影响类别输出   -----这个要根据样本改----

    '''模型权重'''
    PATH_SAVE_WEIGHT = PATH_HOST + '/AI/weights/feadre'
    SAVE_FILE_NAME = 'train_center_'  # 预置一个 实际无用 根据文件名确定
    FILE_FIT_WEIGHT = PATH_SAVE_WEIGHT + '/train_center_m2_widerface-59_4.557.pth'
    FILE_FIT_WEIGHT = PATH_SAVE_WEIGHT + '/train_center_m2_raccoon200-46_0.988.pth'
    # FILE_FIT_WEIGHT = None

    '''Loss参数'''
    # IGNORE_THRESH = 0.225
    FEATURE_MAP_STEP = 4  # 特图的步距 下采倍数
    NUMS_ANC = [3, 3, 3]

    IS_MIXTURE_FIX = True  # 这个不要动 半精度训练


if __name__ == '__main__':
    import numpy as np

    print(1e-3)
    array = np.array(CFG.ANC_SCALE)
    print(array * 416)
