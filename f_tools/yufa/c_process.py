from multiprocessing import Process, Pool, Lock
import time, os

'''多进程---CPU密集型：程序需要占用CPU进行大量的运算和数据处理；'''
'''多线程---I/O密集型：程序中需要频繁的进行I/O操作；例如网络中socket数据传输和读取等；'''


def func(str, s):
    # os.getpid  获取当前进程的进程号
    # os.getppid  获取当前进程的父进程
    i = 1
    while True:
        if str == 'stop' or i == 5:
            break
        print("this is process：%s--pid：%s--ppid：%s" % (str, os.getpid(), os.getppid()))
        time.sleep(s)
        i += 1


class CustomProcess(Process):
    def __init__(self, p_name, target=None):
        # step 1: call base __init__ function()
        super(CustomProcess, self).__init__(name=p_name, target=target, args=(p_name,))

    def run(self):
        # step 2:
        # time.sleep(0.1)
        print("Custom Process name: %s, pid: %s " % (self.name, os.getpid()))


def 普通子进程():
    global p1
    # 创建子进程
    # target 说明进程的任务
    p = Process(target=func, args=("1111111111111111111",))
    p1 = Process(target=func, args=("222",))

    p1.start()
    # p1 = Process(target=func, args=("python",))
    # 启动进程
    p.start()
    # 主进程中的
    # while True:
    #     print("this is a process 1--%s--%s" % (os.getpid(), os.getppid()))
    #     time.sleep(1)
    p.join()
    print("父结束")


def 类进程():
    global p1
    '''class 进度'''
    p1 = CustomProcess("process_1")
    p1.start()
    p1.join()
    print("subprocess pid: %s" % p1.pid)
    print("current process pid: %s" % os.getpid())


if __name__ == '__main__':
    print("父进程启动...--getpid:%s--getppid:%s" % (os.getpid(), os.getppid()))
    # 普通子进程()

    pp = Pool(2)
    for i in range(10):
        # 创建进程,放入进程池统一管理
        pp.apply_async(func, args=(i, i + 1))
        # 在调用join之前必须先关掉进程池
        # 进程池一旦关闭  就不能再添加新的进程了
    pp.close()
    # 进程池对象调用join,会等待进程池中所有的子进程结束之后再结束父进程
    print('准备等待')
    pp.join()
    print("父进程结束...")

    # 类进程()
