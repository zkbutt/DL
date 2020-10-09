import os
import psutil
import argparse
import torch
# import torch.distributed as dist

# from f_tools.GLOBAL_LOG import flog
import torchvision

if __name__ == '__main__':
    '''
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --master_port 29501  t_multi_gpu_DDP.py
    CUDA_VISIBLE_DEVICES=2,1,0 python -m torch.distributed.launch --nproc_per_node=3  /home/fast120/ai_code/DL/tmp/pycharm_project_243/f_pytorch/t_multi_gpu_DDP.py
    python -m torch.distributed.launch --nproc_per_node=3 t_multi_gpu_DDP.py
        torch.distributed.launch根据GPU数量触发 n个进程
            --nproc_per_node=2 launch要求必须传
    '''
    parser = argparse.ArgumentParser()
    # 这是torch.distributed.launch 自动传来的
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    # print('传入的所有参数', args) # 只有 local_rank=0 这一个参数
    # print('gpu编号 args.local_rank: ', args.local_rank) # gpu编号根据  CUDA_VISIBLE_DEVICES=0,1
    device = torch.device("cuda", args.local_rank)  # 设置单一显卡

    # WORLD_SIZE 由torch.distributed.launch.py产生 具体数值为 nproc_per_node*node(主机数，这里为1)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print('当前 GPU', torch.cuda.get_device_name(args.local_rank),
          args.local_rank,  # 进程变量 GPU号 进程号
          '总GPU个数', num_gpus,  # 这个变量是共享的
          "ppid:%s ...pid: %s" % (os.getppid(), os.getpid())
          )

    is_distributed = num_gpus > 1

    if is_distributed:
        torch.cuda.set_device(args.local_rank)  # 这里设定每一个进程使用的GPU是一定的
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    '''用torch.distributed.launch为每一个GPU启动一个进程，接收参数cpu个数，'''
    # model = torchvision.models.vgg16(pretrained=True)
    # model = torchvision.models.MobileNetV2()
    # model = torchvision.models.ShuffleNetV2()
    # model = torchvision.models.SqueezeNet()
    model = torchvision.models.resnet50(num_classes=10)
    model.to(device)
    # 将模型移至到DistributedDataParallel中，此时就可以进行训练了
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=True,
        )
    # rnn = model.cuda()
    # par = torch.nn.DataParallel(rnn)

    for i in range(10):
        print(torch.cuda.get_device_name(args.local_rank), args.local_rank, '----------', i)
        inp = torch.randn(32, 3, 300, 300).to(device)
    output = model(inp)
    loss_val = output.sum()
    loss_val.backward()

    print(torch.cuda.get_device_name(torch.cuda.current_device()), args.local_rank, '完成')
