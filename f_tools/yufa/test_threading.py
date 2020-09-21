import threading
import time

from tools_my import FTools


def thread_fun(self):
    print("thread_fun %s", self)


def fun1(self):
    time.sleep(1)
    print("激活线程数:%s  当前线程名称:%s" % (threading.active_count(), threading.current_thread().getName()), )
    print("我是 fun1", self)


def testThread():
    print("激活线程数:%s  当前线程名称:%s" % (threading.active_count(), threading.current_thread().getName()), )
    threads = []
    thread_num = 9
    for i in range(thread_num):
        threads.append(threading.Thread(target=fun1, args=('线程' + str(i),)))
    for t in threads:
        t.start()

    for t in threads:
        print("主线程阻塞---------------%s----------" % (t))
        t.join()


if __name__ == '__main__':
    print("主线程完成---", FTools.funTime(testThread))
