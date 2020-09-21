import _thread  # 直接使用
import os
import threading
import time
import queue


class A:
    @staticmethod
    def f1(title, msg):
        print('开始---------f1', )
        import smtplib
        from email.mime.text import MIMEText
        try:
            mail_server = "smtp.qq.com"  # 邮箱服务器地址
            username = 'feadre@qq.com'  # 邮箱用户名
            password = 'nusirjexzgydbhdh'  # 邮箱密码：需要使用授权码
            # add_to = '27113970@qq.com'  # 收件人，多个收件人用逗号隔开
            add_to = ['591421342@qq.com', '27113970@qq.com']
            # add_from = "维戈乐思健康管理系统: <%s>" % username

            mailmsg = MIMEText(msg)
            mailmsg['Subject'] = '维戈乐思系统通知：' + title
            mailmsg['From'] = "维戈乐思健康管理系统: <%s>" % username  # 发件人
            mailmsg['To'] = ",".join(add_to)  # 收件人；[]里的三个是固定写法，别问为什么，我只是代码的搬运工
            # smtp = smtplib.SMTP(mailserver, port=25)  # 连接邮箱服务器，smtp的端口号是25
            smtp = smtplib.SMTP_SSL(mail_server, port=465)  # QQ邮箱的服务器和端口号
            smtp.login(username, password)  # 登录邮箱
            smtp.sendmail(username, add_to, mailmsg.as_string())  # 参数分别是发送者，接收者，第三个是把上面的发送邮件的内容变成字符串
            smtp.quit()  # 发送完毕后退出smtp
            print('success')
        except Exception as e:
            print(e)


def test_fun_thread():
    '''
    _thread.start_new_thread ( function, args[, kwargs] )
        function - 线程函数。
        args - 方法的参数,他必须是个tuple类型。
        kwargs - 可选参数。
    '''
    try:
        # 创建两个线程
        _thread.start_new_thread(print_time, ("----------Thread-1-------------", 2,))
        _thread.start_new_thread(print_time, ("Thread-2", 4,))
    except:
        print("Error: 无法启动线程")
    while 1:
        pass


# 为线程定义一个函数
def print_time(threadName, delay):
    count = 0
    while count < 3:
        time.sleep(delay)
        count += 1
        print("%s: %s" % (threadName, time.ctime(time.time())))


exitFlag = 0


def class_print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            threadName.exit()
        time.sleep(delay)
        print("%s: %s -------%s  %s" % (threadName, time.ctime(time.time()), threading.current_thread(), os.getpid()))
        counter -= 1


class MyThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter  # 计数

    def run(self):
        print("开始线程：" + self.name)
        # 获取锁，用于线程同步
        # threadLock.acquire()

        # class_print_time(self.name, self.counter, 5)
        # 释放锁，开启下一个线程
        # threadLock.release()
        print("退出线程：" + self.name)


if __name__ == '__main__':
    # test_fun_thread()

    '''
    run(): 用以表示线程活动的方法。
    start():启动线程活动。
    join([time]): 等待至线程中止。这阻塞调用线程直至线程的join() 方法被调用中止-正常退出或者抛出未处理的异常-或者是可选的超时发生。
    isAlive(): 返回线程是否活动的。
    getName(): 返回线程名。
    setName(): 设置线程名。
    '''
    threadLock = threading.Lock()
    threads = []
    thread1 = MyThread(1, "Thread-1", 1)
    thread2 = MyThread(2, "Thread-2", 2)

    # thread1.setDaemon(True)  # 主线程退出时结束
    # thread1.start()
    # thread2.start()

    # 快速使用
    thread_hello = threading.Thread(target=A.f1, args=('hello',))
    thread_hello.start()
    # threads.append(thread1)
    # threads.append(thread2)
    # for t in threads:
    #     t.join()

    # thread1.join()  # 阻塞主线程待1结束
    # thread2.join()
    # print(thread1.is_alive())
    # print(thread1.getName())
    print("退出主线程")
