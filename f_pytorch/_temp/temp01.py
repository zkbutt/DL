def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


def create_tensorboard(net, trainloader):
    # ----------------------批量看数据----------------------------
    from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    # get some random training images

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # create grid of images
    img_grid = torchvision.utils.make_grid(images, nrow=2, padding=2)
    # write to tensorboard  用这个命令运行:  tensorboard --logdir=runs
    writer.add_image('four_fashion_mnist_images', img_grid)

    # ----------------------保存计算图----------------------------
    writer.add_graph(net, images)

    # ---------------------数据可视化  1维到3维----------------------------
    n = 100
    images = train_set.data[:n]  # n,28,28
    features = images.view(-1, 28 * 28)  # 图片拉成1维
    #  不加这里要报错,解决兼容性问题
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    writer.add_embedding(
        features,
        metadata=train_set.targets[:n],  # 类别
        label_img=images.unsqueeze(1),  # n,1,28,28 # 3D中显示图片
    )

    return writer


def f_训练Tensorboard():
    global epoch, data, labels, outputs, loss, running_loss
    writer = create_tensorboard(model, train_loader)
    for epoch in range(epochs):  # loop over the dataset multiple times

        for i, data in enumerate(train_loader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients 清零
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)  # 评估方法,损失计算
            loss.backward()  # 反向传播
            optimizer.step()  # 优化

            running_loss += loss.item()  # tensor 叠加后平均 省内存
            if i % 1000 == 999:  # every 1000 mini-batches...
                print(i)
                # Tensorboard 平均损失
                writer.add_scalar('training loss', running_loss / 1000, epoch * len(train_loader) + i)
                # Tensorboard 平均损失
                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(model, inputs, labels),
                                  global_step=epoch * len(train_loader) + i)
                running_loss = 0.0
    print('Finished Training')
    writer.close()


# helper functions

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
