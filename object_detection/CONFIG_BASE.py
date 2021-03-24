import socket


class CfgBase:
    START_EVAL = 10
    END_EVAL = 50  # 结束每轮验证
    EVAL_INTERVAL = 3  # 间隙
    MAPS_VAL = [0.1, 0.1]
    LR0 = 1e-3
    USE_MGPU_EVAL = True

    '''----TB_WRITER设备-----'''
    NUMS_EVAL = {10: 5, 100: 3, 160: 2}
    TB_WRITER = True
    DEL_TB = False
    LOSS_EPOCH = True  # False表示iter

    MODE_COCO_TRAIN = 'bbox'  # bbox segm keypoints caption
    MODE_COCO_EVAL = 'bbox'  # bbox segm keypoints caption

    '''多尺度'''
    IS_MULTI_SCALE = True  # 多尺度训练
    MULTI_SCALE_VAL = [10, 19]  # 多尺寸的比例0.667~1.5 之间 满足32的倍数

    '''data_pretreatment'''
    KEEP_SIZE = False

    host_name = socket.gethostname()
    if host_name == 'Feadre-NB':
        PATH_HOST = 'M:'
        # raise Exception('当前主机: %s 及主数据路径: %s ' % (host_name, cfg.PATH_HOST))
    elif host_name == 'e2680v2':
        PATH_HOST = ''

    '''训练参数'''
    SYSNC_BN = True  # 不冻结时可使用多设备同步BN,速度减慢

    '''可视化'''
    IS_VISUAL = False
    IS_VISUAL_PRETREATMENT = False  # 图片预处理

    IS_MIXTURE_FIX = True  # 这个不要动 半精度训练
    EPOCH_WARMUP = 1  # 这个控制热身的 epoch 1是只热身一次
    FORWARD_COUNT = 1  # 多次迭代再反向

    tcfg_epoch = None  # 多尺寸训练时用
    MODE_TRAIN = 1  # 自定义多种训练方式  及损失函数 通过备注
    # import getpass
    # # 获取当前系统用户名
    # user_name = getpass.getuser()
    # # 获取当前系统用户目录
    # user_home = os.path.expanduser('~')

    # loss_args = {
    #     's_match': 'log_g',  # 'log' 'whoned' 'log_g'
    #     's_conf': 'ohem',  # 'mse' 'foc' 'ohem'
    #     's_cls': 'bce',  # 'bce'  'ce'
    # }
