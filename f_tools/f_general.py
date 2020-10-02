import errno
import json
import os
import sys
import time

import numpy as np


def get_path_root():
    debug_vars = dict((a, b) for a, b in os.environ.items() if a.find('IPYTHONENABLE') >= 0)
    # 根据不同场景获取根目录
    if len(debug_vars) > 0:
        """当前为debug运行时"""
        path_root = sys.path[2]
    elif getattr(sys, 'frozen', False):
        """当前为exe运行时"""
        path_root = os.getcwd()
    else:
        """正常执行"""
        path_root = sys.path[1]
    path_root = path_root.replace("\\", "/")  # 替换斜杠
    return path_root


def is_float(str):
    from f_tools.GLOBAL_LOG import flog
    if str.count('.') == 1:  # 小数有且仅有一个小数点
        left = str.split('.')[0]  # 小数点左边（整数位，可为正或负）
        right = str.split('.')[1]  # 小数点右边（小数位，一定为正）
        lright = ''  # 取整数位的绝对值（排除掉负号）
        if str.count('-') == 1 and str[0] == '-':  # 如果整数位为负，则第一个元素一定是负号
            lright = left.split('-')[1]
        elif str.count('-') == 0:
            lright = left
        else:
            flog.debug('%s 不是小数' % str)
            return False
        if right.isdigit() and lright.isdigit():  # 判断整数位的绝对值和小数位是否全部为数字
            flog.debug('%s 是小数' % str)
            return True
        else:
            flog.debug('%s 不是小数' % str)
            return False

    else:
        flog.debug('%s 不是小数' % str)
        return False


def show_time(f, *arg):
    import time
    from f_tools.GLOBAL_LOG import flog

    start = time.time()
    flog.debug('show_time---开始---%s-------' % (f.__name__))
    ret = f(*arg)
    flog.debug('show_time---完成---%s---%s' % (f.__name__, time.time() - start))
    return ret


def show_progress(epoch, epochs, title, context):
    '''
    完成后在循环的外层需要补一个        print()
    '''
    rate = (epoch + 1) / epochs
    a = "*" * int(rate * 50)
    b = "." * int((1 - rate) * 50)
    time.sleep(0.2)
    print("\r{}: {:^3.0f}%[{}->{}] {}".format(title, int(rate * 100), a, b, context), end="")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def mkdir(path):
    # 多GPU
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


'''------------初始化对象------------'''

if __name__ == "__main__":
    pass
