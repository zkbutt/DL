import math
import sys
import time
import datetime
import torch
from collections import defaultdict, deque
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from f_tools.GLOBAL_LOG import flog
from object_detection.coco_t.coco_api import coco_eval


def f_train_one_epoch(data_loader, loss_process, optimizer, epoch, end_epoch,
                      print_freq=60, lr_scheduler=None,
                      ret_train_loss=None, ret_train_lr=None):
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
    metric_logger = MetricLogger(delimiter="  ")  # 日志记录器
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch + 1, end_epoch)

    # ---半精度训练1---
    scaler = GradScaler()
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        # ---半精度训练2---
        with autocast():
            #  完成  数据组装完成   模型输入输出    构建展示字典及返回值
            loss_total, log_dict = loss_process(batch_data)

            if isinstance(ret_train_loss, list):
                ret_train_loss.append(loss_total.detach())

            if not math.isfinite(loss_total):  # 当计算的损失为无穷大时停止训练
                flog.error("Loss is {}, stopping training".format(loss_total))
                flog.error(log_dict)
                sys.exit(1)
        # ---半精度训练2  完成---

        optimizer.zero_grad()
        # ---半精度训练3---
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        # 半精度开了这两个要关
        # losses.backward()
        # optimizer.step()
        if lr_scheduler is not None:  # 每批训练lr更新器
            lr_scheduler.step()

        # 这里记录日志输出 直接用字典输入
        metric_logger.update(**log_dict)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        if isinstance(ret_train_lr, list):
            ret_train_lr.append(now_lr)  # 这个是返回值
    return metric_logger.meters['total_losses'].avg  # 默认这epoch使用平均值


@torch.no_grad()
def f_evaluate(data_loader, forecast_process, epoch, coco_res_bboxs, coco_res_keypoints=None):
    '''

    :param data_loader:
    :param forecast_process:
    :param epoch:
    :return:
    '''
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: "
    print_freq = max(int(len(data_loader) / 5), 1)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        evaluator_time = time.time() # 139911109878320
        forecast_process(batch_data, epoch, coco_res_bboxs, coco_res_keypoints=coco_res_keypoints)

        eval_time = time.time() - evaluator_time
        metric_logger.update(eval_time=eval_time, coco_size=len(coco_res_bboxs))  # 这个填字典 添加的字段

    # 总输出
    coco_eval(coco_res_bboxs, forecast_process.coco, epoch, 'bbox')
    if forecast_process.eval_mode == 'keypoints':
        coco_eval(coco_res_keypoints, forecast_process.coco, epoch, 'keypoints')


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
                                           '剩余时间: {r_time}',
                                           'max mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}',
                                           '剩余时间: {r_time}',
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
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),  # 模型迭代时间(含数据加载) SmoothedValue对象
                                         data=str(data_time),  # 取数据时间 SmoothedValue对象
                                         r_time=str(int(iter_time.value * (len(iterable) - i))),
                                         memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         r_time=str(int(iter_time.value * (len(iterable) - i))),
                                         data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # 每批时间
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
                                                         total_time_str,
                                                         total_time / len(iterable)))


#
# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()


# def reduce_dict(input_dict, average=True):
#     """
#     Args:
#         input_dict (dict): all the values will be reduced
#         average (bool): whether to do average or sum
#     Reduce the values in the dictionary from all processes so that all processes
#     have the averaged results. Returns a dict with the same fields as
#     input_dict, after reduction.
#     """
#     world_size = get_world_size()
#     if world_size < 2:  # 单GPU的情况
#         return input_dict
#     with torch.no_grad():  # 多GPU的情况
#         names = []
#         values = []
#         # sort the keys so that they are consistent across processes
#         for k in sorted(input_dict.keys()):
#             names.append(k)
#             values.append(input_dict[k])
#         values = torch.stack(values, dim=0)
#         dist.all_reduce(values)
#         if average:
#             values /= world_size
#
#         reduced_dict = {k: v for k, v in zip(names, values)}
#         return reduced_dict


# def _get_iou_types(model):
#     model_without_ddp = model
#     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         model_without_ddp = model.module
#     iou_types = ["bbox"]
#     return iou_types


# def all_gather(data):
#     """
#     Run all_gather on arbitrary picklable data (not necessarily tensors)
#     Args:
#         data: any picklable object
#     Returns:
#         list[data]: list of data gathered from each rank
#     """
#     world_size = get_world_size()
#     if world_size == 1:
#         return [data]
#
#     # serialized to a Tensor
#     buffer = pickle.dumps(data)
#     storage = torch.ByteStorage.from_buffer(buffer)
#     tensor = torch.ByteTensor(storage).to("cuda")
#
#     # obtain Tensor size of each rank
#     local_size = torch.tensor([tensor.numel()], device="cuda")
#     size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
#     dist.all_gather(size_list, local_size)
#     size_list = [int(size.item()) for size in size_list]
#     max_size = max(size_list)
#
#     # receiving Tensor from all ranks
#     # we pad the tensor because torch all_gather does not support
#     # gathering tensors of different shapes
#     tensor_list = []
#     for _ in size_list:
#         tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
#     if local_size != max_size:
#         padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
#         tensor = torch.cat((tensor, padding), dim=0)
#     dist.all_gather(tensor_list, tensor)
#
#     data_list = []
#     for size, tensor in zip(size_list, tensor_list):
#         buffer = tensor.cpu().numpy().tobytes()[:size]
#         data_list.append(pickle.loads(buffer))
#
#     return data_list


# def setup_for_distributed(is_master):
#     """
#     This function disables when not in master process
#     """
#     import builtins as __builtin__
#     builtin_print = __builtin__.print
#
#     def print(*args, **kwargs):
#         force = kwargs.pop('force', False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)
#
#     __builtin__.print = print


# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()


# def is_main_process():
#     return get_rank() == 0


# def save_on_master(*args, **kwargs):
#     if is_main_process():
#         torch.save(*args, **kwargs)


# def init_distributed_mode(args):
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     elif 'SLURM_PROCID' in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gpu = args.rank % torch.cuda.device_count()
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return
#
#     args.distributed = True
#
#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}'.format(
#         args.rank, args.dist_url), flush=True)
#     torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                          world_size=args.world_size, rank=args.rank)
#     torch.distributed.barrier()
#     setup_for_distributed(args.rank == 0)


# class CocoExport():
#
#     def __init__(self, dataset) -> None:
#         super().__init__()
#         _coco_ds = COCO()
#         _coco_ds.dataset = dataset
#         _coco_ds.createIndex()
#         iou_types = ["bbox"]
#         self.coco_evaluator = CocoEvaluator(_coco_ds, iou_types)
#
#     def __call__(self, data_outputs):
#         '''
#         构造coco测试集格式
#         :param data_outputs:  list(dict{'boxes':(x,4),'labels':[x],'scores':[x]})
#         :return:
#             res={0:info1,...x:infox}
#         '''
#         cpu_device = torch.device("cpu")
#         res = dict()
#         for index, (bboxes_out, labels_out, scores_out, hw) in enumerate(data_outputs):
#             info = {
#                 "boxes": bboxes_out.to(cpu_device),
#                 "labels": labels_out.to(cpu_device),
#                 "scores": scores_out.to(cpu_device),
#                 "height_width": hw
#             }
#             res.update({index, info})
#         return res


if __name__ == '__main__':
    from torchvision import models
    from torch import optim
    SmoothedValue