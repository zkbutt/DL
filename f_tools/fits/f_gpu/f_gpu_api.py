import argparse
import pickle
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from f_tools.GLOBAL_LOG import flog


def dict_all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    接收任意类型
    Args:
        data: any picklable object
    Returns:
        返回组内每一个GPU值的 list
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return input_dict
    with torch.no_grad():  # 多GPU的情况
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict


def setup_for_distributed(is_master):
    """
    This function disables when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def fis_mgpu():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not fis_mgpu():
        return 1
    return dist.get_world_size()


def get_rank():
    if not fis_mgpu():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def mgpu_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])  # 多机时：当前是第几的个机器  单机时：第几个进程
        args.world_size = int(os.environ['WORLD_SIZE'])  # 多机时：当前是有几台机器 单机时：总GPU个数
        args.gpu = int(os.environ['LOCAL_RANK'])  # 多机时：当前GPU序号
    else:
        raise Exception('环境变量有问题 %s' % os.environ)

    torch.cuda.set_device(args.local_rank)

    device = torch.device("cuda", args.local_rank)  # 获取显示device
    torch.distributed.init_process_group(backend="nccl",
                                         init_method="env://",
                                         world_size=args.world_size,
                                         rank=args.rank
                                         )
    torch.distributed.barrier()  # 等待所有GPU初始化 初始化完成 629M
    return args, device


# def mgpu_process0_init(args, cfg, loader_train, loader_val_coco, model, device):
#     '''支持 add_graph'''
#     # 主进程任务
#     flog.info('多GPU参数: %s' % args)
#     if not os.path.exists(cfg.PATH_SAVE_WEIGHT):
#         try:
#             os.makedirs(cfg.PATH_SAVE_WEIGHT)
#         except Exception as e:
#             flog.error(' %s %s', cfg.PATH_SAVE_WEIGHT, e)
#     # tensorboard --logdir=runs_widerface --host=192.168.0.199
#     # tensorboard --logdir=runs_voc --host=192.168.0.199
#     # print('"tensorboard --logdir=runs_raccoon200 --host=192.168.0.199", view at http://192.168.0.199:6006/'
#     #       % cfg.PATH_TENSORBOARD)
#
#     path = os.path.join(cfg.PATH_PROJECT_ROOT, cfg.PATH_TENSORBOARD)
#
#     img_ = None
#     if os.path.exists(path):
#         if cfg.DEL_TB and cfg.PATH_HOST == '':
#             # os.remove(path)  # 删除空文件夹 shutil.rmtree(path, ignore_errors=True)
#             os.system("rm -rf %s" % path)  # Linux下调用bash命令
#             img_ = torch.zeros((1, 3, *cfg.IMAGE_SIZE), device=device)  # 删除后需要
#             import time
#             while os.path.exists(path):
#                 time.sleep(1)
#             else:
#                 flog.error('tb_writer 删除成功: %s', path)
#             pass
#     else:
#         img_ = torch.zeros((1, 3, *cfg.IMAGE_SIZE), device=device)  # 不存在时需要
#
#     tb_writer = SummaryWriter(path)
#
#     # if img_ is not None:
#     #     tb_writer.add_graph(model, img_)
#
#     return tb_writer


def mgpu_process0_init(args, cfg, loader_train, loader_val_coco, model, device):
    '''支持 add_graph'''
    # 主进程任务
    flog.info('多GPU参数: %s' % args)
    if not os.path.exists(cfg.PATH_SAVE_WEIGHT):
        try:
            os.makedirs(cfg.PATH_SAVE_WEIGHT)
        except Exception as e:
            flog.error(' %s %s', cfg.PATH_SAVE_WEIGHT, e)
    # tensorboard --logdir=runs_widerface --host=192.168.0.199
    # tensorboard --logdir=runs_voc --host=192.168.0.199
    # print('"tensorboard --logdir=runs_raccoon200 --host=192.168.0.199", view at http://192.168.0.199:6006/'
    #       % cfg.PATH_TENSORBOARD)
    import time
    c_time = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
    path = os.path.join(cfg.PATH_PROJECT_ROOT, 'log', cfg.PATH_TENSORBOARD, c_time)
    os.makedirs(path, exist_ok=True)

    # img_ = None
    # if os.path.exists(path):
    #     if cfg.DEL_TB and cfg.PATH_HOST == '':
    #         # os.remove(path)  # 删除空文件夹 shutil.rmtree(path, ignore_errors=True)
    #         os.system("rm -rf %s" % path)  # Linux下调用bash命令
    #         img_ = torch.zeros((1, 3, *cfg.IMAGE_SIZE), device=device)  # 删除后需要
    #         import time
    #         while os.path.exists(path):
    #             time.sleep(1)
    #         else:
    #             flog.error('tb_writer 删除成功: %s', path)
    #         pass
    # else:
    #     img_ = torch.zeros((1, 3, *cfg.IMAGE_SIZE), device=device)  # 不存在时需要

    tb_writer = SummaryWriter(path)

    # if img_ is not None:
    #     tb_writer.add_graph(model, img_)

    return tb_writer


def model_device_init(model, device, id_gpu, cfg):
    if id_gpu is not None:
        # 多GPU初始化
        is_mgpu = True
        if cfg.SYSNC_BN:
            # 不冻结权重的情况下可, 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        else:
            model.to(device)  # 这个不需要加等号
        # 转为DDP模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[id_gpu], find_unused_parameters=True)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[id_gpu])
    else:
        model.to(device)  # 这个不需要加等号
        is_mgpu = False
    return model, is_mgpu


if __name__ == '__main__':
    # scatter 可以将信息从master节点传到所有的其他节点
    # gather 可以将信息从别的节点获取到master节点

    # 多GPU 同步测试
    args, device = mgpu_init()
    rank = get_rank()
    # group = dist.new_group([0, 1])

    var = torch.randn((2), device=device)

    # 返回组内每一个GPU值的 list 支持任务类型
    # var = 123
    # gather = dict_all_gather(var) 这个加 if is_main_process()

    print('原Rank ', rank, ' has data ', var)
    req = None
    dist.broadcast(var, src=0)  # 同步广播  if is_main_process(): 这个不能加主进程

    '''ProcessGroupNCCL does not support send'''
    # req = dist.isend(tensor=var, dst=1)# 异步 req.wait()
    # req.wait()
    # req = dist.irecv(tensor=var, src=0)# 异步 req.wait()
    # req.wait()
    # dist.send(tensor=var, dst=1)
    # dist.recv(tensor=var, src=0)

    # if get_rank() == 0:
    #     # req = dist.isend(tensor=var, dst=1)  # 异步 req.wait()
    #     # dist.send(tensor=var, dst=1)
    #     print('Rank 0 started sending')
    # else:
    #     # req = dist.irecv(tensor=var, src=0)  # 异步 req.wait()
    #     print('Rank 1 started receiving')

    # 获取所有GPU的数据并完成计算 并赋值 只支持tensor
    # async_op=False 为同步 dist.reduce_op.SUM  dist.reduce_op.PRODUCT  dist.reduce_op.MAX  dist.reduce_op.MIN
    # dist.all_reduce(var, op=dist.ReduceOp.PRODUCT, async_op=False)

    print('目标Rank ', rank, ' has data ', var)
