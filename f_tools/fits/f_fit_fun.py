import os
import time

import torch
from torch.nn.utils import clip_grad_norm

from f_tools.GLOBAL_LOG import flog
from f_tools.f_general import show_progress
from torch.autograd import Variable
import matplotlib.pyplot as plt

from f_tools.pic.f_show import show_pic_np


def f_fit_basics(model, epochs, train_loader, validate_loader,
                 loss_function, optimizer, lr_scheduler=None, device='cpu',
                 save_path=os.path.basename(__file__) + '_temp.pth',
                 is_eval=True
                 ):
    running_loss = 0.0
    logs_train = {}
    key_loss = 'loss'
    key_loss_step = 'loss_step'
    key_test_acc = 'test_acc'
    logs_train[key_loss] = []
    logs_train[key_loss_step] = []
    logs_train[key_test_acc] = []
    epoch_start = 0

    best_acc = 0.0

    # 加载已训练的数据
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler']),
        logs_train = checkpoint['logs_train']
        epoch_start = checkpoint['epoch']
        if epoch_start == epochs - 1:
            flog.info('已训练完成{}{}{}'.format(epoch_start, '/', epochs))
            return running_loss, best_acc, logs_train

    for epoch in range(epoch_start, epochs):
        model.train()
        t1 = time.perf_counter()  # 时间计算
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            optimizer.zero_grad()  # 固定---优化清0
            loss.backward()  # 固定---求梯度
            # clip_grad_norm(model.parameters(), 0.5)  # 梯度裁剪 限制梯度上限
            optimizer.step()  # 固定---更新参数
            running_loss += loss.item()  # 取单个值

            # ---进度条---
            show_progress(epoch, epochs, 'training', loss)
        print()
        t2 = time.perf_counter() - t1
        print('训练用时: ', t2)

        if not lr_scheduler:
            lr_scheduler.step()

        if is_eval:
            model.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                for i, val_data in enumerate(validate_loader):
                    val_images, val_labels = val_data
                    outputs = model(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    # _, predicted = torch.max(outputs.data, 1)
                    _acc = (predict_y == val_labels.to(device)).sum().item()
                    acc += _acc
                    show_progress(i, len(validate_loader), 'verification', _acc)
                print()
                val_accurate = acc / len(validate_loader)

                # 保存最好的参数
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    # 训练保存
                    state = {'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'lr_scheduler': lr_scheduler.state_dict(),
                             'logs_train': logs_train,
                             'epoch': epoch}
                    torch.save(state, save_path)  # 按验证集最好进行保存

                loss_step = running_loss / step
                logs_train[key_loss_step].append(loss_step)
                logs_train[key_test_acc].append(val_accurate)
                print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f  验证用时: %s' %
                      (epoch + 1, loss_step, val_accurate, time.perf_counter() - t1))

        if (epoch + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

    return running_loss, best_acc, logs_train


def train_model(train_loader, model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    since = time.time()
    best_model_wts = model.state_dict()  # Returns a dictionary containing a whole state of the module.
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # set the mode of model
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()  # about lr and gamma
                model.train(True)  # set model to training mode
            else:
                model.train(False)  # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for data in train_loader:
                inputs, labels = data
                if device == 'gpu':
                    inputs = Variable(inputs.cuda_idx())
                    labels = Variable(labels.cuda_idx())
                else:
                    inputs = Variable(inputs)
                    lables = Variable(labels)
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                # backward
                if phase == 'train':
                    loss.backward()  # backward of gradient
                    optimizer.step()  # strategy to drop
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds.data == labels.data)

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = running_corrects / len(train_loader)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(validate_loader, class_names, model, num_images=6, device='cpu', ):
    '''

    :param validate_loader:
    :param class_names: {index : name}
    :param model:
    :param num_images:
    :param device:
    :return:
    '''
    images_so_far = 0

    for i, data in enumerate(validate_loader):
        inputs, labels = data
        if device == 'gpu':
            inputs, labels = Variable(inputs.cuda_idx()), Variable(labels.cuda_idx())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            show_pic_np(inputs.cpu().data[j], is_torch=True)

            if images_so_far == num_images:
                return


def fit_one_epoch(model, cls_loss, epoch, data_loader, epoch_end, anchors, device):
    total_r_loss = 0
    total_c_loss = 0
    total_landmark_loss = 0

    start_time = time.time()
    with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{epoch_end}', postfix=dict, mininterval=0.3) as pbar:
        for i, (images, targets) in enumerate(data_loader):
            # if i >= epoch_size:
            #     break
            with torch.no_grad():  # np -> tensor
                # torch.Size([8, 3, 640, 640])
                images = Variable(torch.from_numpy(images).type(torch.float)).to(device)
                # n,15(4+10+1)
                targets = [Variable(torch.from_numpy(target).type(torch.float)).to(device) for target in targets]
            optimizer.zero_grad()
            # forward
            out = model(images)  # tuple(torch.Size([8, 16800, 4]),torch.Size([8, 16800, 2]),torch.Size([8, 16800, 10]))
            r_loss, c_loss, landm_loss = cls_loss(out, anchors, targets, device)
            loss = 2 * r_loss + c_loss + landm_loss  # r_loss放大2倍

            loss.backward()
            optimizer.step()

            total_c_loss += c_loss.item()
            total_r_loss += r_loss.item()
            total_landmark_loss += landm_loss.item()
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'Conf Loss': total_c_loss / (i + 1),
                                'Regression Loss': total_r_loss / (i + 1),
                                'LandMark Loss': total_landmark_loss / (i + 1),
                                'lr': get_lr(optimizer),
                                's/step': waste_time})
            pbar.update(1)
            start_time = time.time()

    flog.info('Saving state, iter:', str(epoch + 1))
    sava_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch}
    torch.save(sava_dict, os.path.join(PATH_MODEL_WEIGHT, 'retinaface{}.pth'.format(epoch + 1)))
    return (total_c_loss + total_r_loss + total_landmark_loss) / (len(data_loader))
