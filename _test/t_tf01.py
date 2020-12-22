from torch.utils.tensorboard import SummaryWriter

tb_writer = SummaryWriter()
for i, v in zip(range(100), range(100, 200)):
    tb_writer.add_scalar('Loss/%s' % 'abc', v, i)
