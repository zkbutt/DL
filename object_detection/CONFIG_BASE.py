import socket


class CfgBase:
    START_EVAL = 10
    END_EVAL = 50  # 结束每轮验证
    PTOPK = 500
    EVAL_INTERVAL = 3  # 间隙
    MAPS_VAL = [0.1, 0.1]
    LR0 = 1e-3
    USE_MGPU_EVAL = True  # 只能用多GPU进行预测 这个必须是True 否则不能使用GPU 一个有一个没得会卡死

    '''----TB_WRITER设备-----'''
    NUMS_EVAL = {10: 5, 100: 3, 160: 2}
    TB_WRITER = True
    # DEL_TB = False 已弃用
    LOSS_EPOCH_TB = True  # False表示iter 这个用于控制训练时 TB_WRITER采用 显示iter次数 或还轮

    MODE_COCO_TRAIN = 'bbox'  # bbox segm keypoints caption
    MODE_COCO_EVAL = 'bbox'  # bbox segm keypoints caption

    '''多尺度'''
    IS_MULTI_SCALE = True  # 多尺度训练
    MULTI_SCALE_VAL = [10, 19]  # 多尺寸的比例0.667~1.5 之间 满足32的倍数

    '''data_pretreatment'''
    KEEP_SIZE = False
    USE_BASE4NP = False  # 使用最基本的预处理(无任务图形增强)

    host_name = socket.gethostname()
    if host_name == 'Feadre-NB':
        PATH_HOST = 'M:'
        # raise Exception('当前主机: %s 及主数据路径: %s ' % (host_name, cfg.PATH_HOST))
    elif host_name == 'e2680v2':
        PATH_HOST = ''

    '''训练参数'''
    SYSNC_BN = True  # 不冻结时可使用多设备同步BN,速度减慢
    FILE_NAME_WEIGHT = '123' + '.pth'  # 重新开始 这个自动默认一个

    '''可视化'''
    IS_VISUAL = False
    IS_VISUAL_PRETREATMENT = False  # 图片预处理

    IS_MIXTURE_FIX = True  # 这个不要动 半精度训练
    EPOCH_WARMUP = 1  # 这个控制热身的 epoch 1是只热身一次
    FORWARD_COUNT = 1  # 多次迭代再反向

    tcfg_epoch = None  # 多尺寸训练时用
    tcfg_show_pic = 0  # 显示图片次数
    NUM_EVAL_SHOW_PIC = 2  # 每次验证显示两张 与 tcfg_show_pic 配合

    MODE_TRAIN = 1  # 自定义多种训练方式  及损失函数 通过备注
    CUSTOM_EVEL = False  # 自定义验证方法
    IS_FMAP_EVAL = False  # 只运行生成一次 这个是另一种 map 几乎没用
    IS_KEEP_SCALE = False  # 计算map时恢复尺寸 几乎没用

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
