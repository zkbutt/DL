import datetime
import logging
import logging.config
import os
from logging import handlers

import colorlog

from f_tools.f_general import get_path_root

'''
args=(os.path.abspath(os.getcwd() + "/info.log"),"midnight", 1, 6,'utf-8')
每一天午夜12点将当天的日志转存到一份新的日志文件中，并且加上时间戳后缀，最多保存6个文件
'''


def Singleton(cls):
    # 单例
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


def get_logger(name='root'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    # formatter = logging.Formatter(fmt)
    # black, red, green, yellow, blue, purple, cyan and white {color}，fg_ {color}，bg_ {color}：前景色和背景色
    log_colors_config = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s] [%(filename)s:%(lineno)d] [%(module)s:%(funcName)s] [%(levelname)s]- %(message)s',
        log_colors=log_colors_config)  # 日志输出格式

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    '''
    实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
    '''
    path_root = os.path.join(get_path_root(), 'logs')
    if not os.path.exists(path_root):
        os.makedirs(path_root)
    th = handlers.TimedRotatingFileHandler(
        filename=os.path.join(path_root, datetime.datetime.now().strftime('%Y_%m_%d') + '.log'),
        when='D',
        backupCount=5, encoding='utf-8')
    th.setFormatter(formatter)
    logger.addHandler(th)

    # fh = RotatingFileHandler(
    #     filename=os.path.join(get_path_root(), 'logs', datetime.datetime.now().strftime('%Y_%m_%d') + '.log'),
    #     mode='a', maxBytes=1024 * 1024 * 5, backupCount=5,
    #     encoding='utf-8')  # 使用RotatingFileHandler类，滚动备份日志
    # fh.setLevel(logging.CRITICAL)
    # fh.setFormatter(formatter)
    # print(logger.handlers)
    # logger.removeHandler(ch)
    return logger


flog = get_logger(__name__)

# flog=Log()
# flog1=Log()
# print(id(flog))
# print(id(flog1))

if __name__ == '__main__':
    # flog.debug('一个连接只需一个 %s', get_path_root)
    # flog.debug('多个连接无需   %s%s', [1, 2, {123}], get_path_root())
    flog.critical(123)
    pass
