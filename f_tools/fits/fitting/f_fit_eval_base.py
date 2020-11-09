import math
import sys
import time
import datetime
import torch
from collections import defaultdict, deque
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from f_tools.GLOBAL_LOG import flog
from f_tools.fits.f_gpu.gpu_mem_track import MemTracker


def f_train_one_epoch(data_loader, loss_process, optimizer, epoch, end_epoch,
                      print_freq=60, lr_scheduler=None,
                      ret_train_loss=None, ret_train_lr=None,
                      is_mixture_fix=True,
                      forward_count=1,
                      ):
    '''

    :param data_loader:
    :param loss_process: 前向和反向 loss过程函数
    :param optimizer:
    :param epoch: 当前次数
    :param print_freq:  每隔50批打印一次
    :param lr_scheduler:  每批训练lr更新器
    :param ret_train_loss:  train_loss[] 返回值
    :param ret_train_lr:  learning_rate[] 返回值
    :return:
    '''
    import inspect
    frame = inspect.currentframe()  # define a frame to track
    # gpu_tracker = MemTracker(frame)  # define a GPU tracker
    # gpu_tracker.track()

    metric_logger = MetricLogger(delimiter="  ")  # 日志记录器
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch + 1, end_epoch)

    # ---半精度训练1---
    scaler = GradScaler(enabled=True)
    for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if is_mixture_fix:
            # ---半精度训练2---
            with autocast():
                #  完成  数据组装完成   模型输入输出    构建展示字典及返回值
                loss_total, log_dict = loss_process(batch_data)

                if ret_train_loss:
                    ret_train_loss.append(loss_total.detach())

                if not math.isfinite(loss_total):  # 当计算的损失为无穷大时停止训练
                    flog.error("Loss is {}, stopping training".format(loss_total))
                    flog.error(log_dict)
                    sys.exit(1)
                # ---半精度训练2  完成---

            # gpu_tracker.track()  # run function between the code line where uses GPU
            # ---半精度训练3---
            scaler.scale(loss_total).backward()
            # gpu_tracker.track()

            # 每训练n批图片更新一次权重
            if i % forward_count == 0:
                scaler.step(optimizer)
                scaler.update()  # 查看是否要更新scaler
                optimizer.zero_grad()
            # scaler.step(optimizer)
            # scaler.update()
            # optimizer.zero_grad()
            # del batch_data
            # torch.cuda.empty_cache()
        else:
            '''-------------全精度--------------'''
            #  完成  数据组装完成   模型输入输出    构建展示字典及返回值
            loss_total, log_dict = loss_process(batch_data)

            if ret_train_loss:
                ret_train_loss.append(loss_total.detach())

            if not math.isfinite(loss_total):  # 当计算的损失为无穷大时停止训练
                flog.error("Loss is {}, stopping training 请使用torch.isnan(x).any() 检测".format(loss_total))
                flog.error(log_dict)
                sys.exit(1)

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()

        if lr_scheduler is not None:  # 每批训练lr更新器
            lr_scheduler.step()

        # 这里记录日志输出 直接用字典输入
        metric_logger.update(**log_dict)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        if isinstance(ret_train_lr, list):
            ret_train_lr.append(now_lr)  # 这个是返回值
    return metric_logger.meters['loss_total'].avg  # 默认这epoch使用平均值


@torch.no_grad()
def f_evaluate(data_loader, predict_handler, epoch, res_eval):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: "
    print_freq = max(int(len(data_loader) / 5), 1)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        _dataset = data_loader.dataset
        d = predict_handler.handler_map_dt_txt(batch_data,
                                               path_dt_info=_dataset.path_dt_info,
                                               idx_to_class=_dataset.idx_to_class,
                                               )

        end_time = time.time() - start_time
        metric_logger.update(eval_time=end_time)  # 这个填字典 添加的字段


class SmoothedValue(object):
    """
    记录一系列统计量
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # deque简单理解成加强版list
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):  # @property 是装饰器，这里可简单理解为增加median属性(只读)
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}',
                                           # '剩余时间: {r_time}',
                                           'max mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}',
                                           # '剩余时间: {r_time}',
                                           ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_second = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=eta_second))
                if torch.cuda.is_available():
                    flog.debug(log_msg.format(i + 1, len(iterable),
                                              eta=eta_string,
                                              meters=str(self),
                                              time=str(iter_time),  # 模型迭代时间(含数据加载) SmoothedValue对象
                                              data=str(data_time),  # 取数据时间 SmoothedValue对象
                                              # r_time=str(int(iter_time.value * (len(iterable) - i))),
                                              memory=torch.cuda.max_memory_allocated() / MB))  # 只能取第一个显卡
                else:
                    flog.debug(log_msg.format(i, len(iterable),
                                              eta=eta_string,
                                              meters=str(self),
                                              time=str(iter_time),
                                              # r_time=str(int(iter_time.value * (len(iterable) - i))),
                                              data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # 每批时间
        flog.debug('{} Total time: {} ({:.4f} s / it)'.format(header,
                                                              total_time_str,
                                                              total_time / len(iterable)))


if __name__ == '__main__':
    pass
