import os
import sys


def f_tensorwatch(model, path, size=(1, 3, 416, 416)):
    '''

    :param model:
    :param path: 保存的绝对 路径
    :return:
    '''
    import tensorwatch as tw

    # img = tw.draw_model(model, [1, 3, 416, 416])
    # print(type(img))
    # img.save(r'pic_model.jpg')

    stats = tw.model_stats(model, size)  # 输入shape 含batch_size
    print(stats)
    # print(type(stats))
    stats.to_excel(path)
    print(os.path.basename(__file__), sys._getframe().f_code.co_name, '完成')


def f_summary(model):
    from torchsummary import summary

    summary(model, (3, 416, 416))  # 无需print 输入shape 不含batch_size
    print(os.path.basename(__file__), sys._getframe().f_code.co_name, '完成')


def f_make_dot(model):
    from torchviz import make_dot

    x = torch.rand(8, 3, 416, 416)  # 输入数据
    y = model(x)
    g = make_dot(y)
    # g = make_dot(y, params=dict(model.named_parameters()))
    # g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    # g.view() # 会生成一个 Digraph.gv.pdf 的PDF文件
    g.render('model', view=False)  # 会自动保存为一个 espnet.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开
    print(os.path.basename(__file__), sys._getframe().f_code.co_name, '完成')
