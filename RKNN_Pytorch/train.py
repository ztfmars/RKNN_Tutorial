import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from utils import EarlyStopping, draw_result_pic

import numpy as np
from tqdm import tqdm
import os
import time
import argparse
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------------------------------
#	数据加载和增强
# -------------------------------------------------------------------------------------------
def data_process(batch_size=32, dataset='./data/data', input_shape=[224, 224]):
    # print("data process", input_shape, type(input_shape))
    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=max(input_shape[0], input_shape[1]), scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=[input_shape[0], input_shape[1]]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=[input_shape[0], input_shape[1]]),
            # transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])
    }

    train_directory = os.path.join(dataset, 'train')
    valid_directory = os.path.join(dataset, 'valid')

    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
    }

    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])

    train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)

    print("[INFO] Train data / Test data number: ", train_data_size, valid_data_size)
    return train_data, valid_data, train_data_size, valid_data_size


# -------------------------------------------------------------------------------------------
#	模型定义： 此处以mobilenet为例进行说明，自用的时候可以更换成自己想用的模型,resnet, shuffle，etc.
# -------------------------------------------------------------------------------------------
# mobilenet-v2模型
def model_define(ntypes=3, input_shape=[224, 224], train_all=True):
    """
    因为mobilenet 的官方实现中（和原始paper实现不一样），
    不是直接sqeeze成固定尺寸送入fc，而是将h,x利用adaptive_avg_pool2d成1x1
    所以维度变为[batch, channel, 1] -> resize 后变为[batch, channels],
    即变为固定值[batch, 1280], 所以能自适应输入的任意的 input shape
    此处的input_shape暂时无用.
    对于固话模型input shape, 全靠的是dataloader图片尺寸的resize
    """
    model = models.mobilenet_v2(pretrained=True)
    print("[INFO] Use mobilenet v2 as model")
    # print("model:", model)
    if train_all:
        # 前面的参数参与训练
        for param in model.parameters():
            param.requires_grad = True
    else:
        # 前面的参数保持不变
        for param in model.parameters():
            param.requires_grad = False

    print("##" * 20)
    fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, ntypes),
    )
    # 直接替换掉最后的类别num即可
    model.classifier = fc
    # print("model:", model)
    # print("[INFO] Model Layer:  ", summary(model, (3, input_shape[0], input_shape[1])))
    return model


# -------------------------------------------------------------------------------------------
#	模型训练
#  添加 early stop / drop  / lr_schedule/ save_best
# -------------------------------------------------------------------------------------------
def train_and_valid(model,
                    # loss/opti/lr设置
                    loss_function,
                    optimizer,
                    lr_scheduler,
                    earlystop_patience,
                    # train & valid 参数设置
                    train_data, train_data_size,
                    valid_data, valid_data_size,
                    train_all,
                    epochs):
    history = []
    lr_list = []
    best_acc = 0.0
    best_epoch = 0

    # 初始化 early_stopping 对象
    # patience = 21  # 当验证集损失在连续21次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    if earlystop_patience != 0:
        if train_all:
            early_stopping = EarlyStopping(earlystop_patience, verbose=True, path="ckpt/best_trainall.pt")
        else:
            early_stopping = EarlyStopping(earlystop_patience, verbose=True, path="ckpt/best_trainfc.pt")

    print("[INFO] Train process to begin")
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        epoch_start = time.time()
        model = model.to(device)

        print("--------------------->Train / Epoch :%d " % (epoch + 1))
        for i, (inputs, labels) in enumerate(tqdm(train_data)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            print("##" * 20)
            print("input size:", inputs.size())
            print("labels:", labels)
            print("##" * 20)

            # 因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()
            print("--------------------->Valid / Epoch :%d " % (epoch + 1))
            for j, (inputs, labels) in enumerate(tqdm(valid_data)):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        lr_list.append(optimizer.state_dict()["param_groups"][0]["lr"])
        #  valid loss 如果>10 step不变化的话，更新lr
        lr_scheduler.step(avg_valid_loss)

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            print("[INFO] Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        epoch_end = time.time()

        print(
            "[INFO]Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\t Validation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))

        print("[INFO] lr list, new lr :", lr_list, lr_list[-1])
        # 前三个epoch往往不稳定，跳过不计
        if (earlystop_patience != 0) and (epoch > 3):
            early_stopping(valid_loss, model)
            # 若满足 early stopping 要求
            if early_stopping.early_stop:
                print("[WARNING] Early stopping")
                # 结束模型训练
                break

    # torch.save(model, 'ckpt/model_' + str(epoch + b) + '.pt')
    return model, history


"""
    config_dict = read_config()
    a = config_dict["input_shape"]
    c = re.findall(r"\d+\.?\d*",a)
    d = [int(i) for i in c]
    print(c)
    print(d)
"""


def train_process(args):
    # 参数设定
    data_dir = args.data
    num_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr_patience = int(args.lr_patience)
    lr_ratio = float(args.lr_ratio)
    earlystop_patience = int(args.earlystop_patience)
    train_all = args.train_all
    ntypes = int(args.ntypes)

    tmp = re.findall(r"\d+\.?\d*", args.input_shape)
    input_shape = [int(i) for i in tmp]

    # 数据处理和增强
    # print("#############Data process#############")
    train_data, valid_data, \
    train_data_size, valid_data_size = data_process(batch_size=batch_size,
                                                    dataset=data_dir,
                                                    input_shape=input_shape)
    # 模型、损失函数、优化函数定义
    # print("#############Defined model#############")
    model = model_define(ntypes=ntypes, input_shape=input_shape, train_all=train_all)

    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters())
    lr_scheduler = ReduceLROnPlateau(optimizer=opt, mode='min', factor=lr_ratio, patience=lr_patience)

    # 模型训练
    # print("#############Model tran and validation #############")
    trained_model, history = train_and_valid(model=model,
                                             # loss/opti/lr设置
                                             loss_function=loss_func,
                                             optimizer=opt,
                                             lr_scheduler=lr_scheduler,
                                             earlystop_patience=earlystop_patience,
                                             # train & valid 参数设置
                                             train_data=train_data, train_data_size=train_data_size,
                                             valid_data=valid_data, valid_data_size=valid_data_size,
                                             train_all=train_all,
                                             epochs=num_epochs)

    # 保存最终模型和绘制图像
    if train_all:
        torch.save(trained_model, 'ckpt/final.pt')
        draw_result_pic(history, add_name="all")
    else:
        torch.save(trained_model, 'ckpt/only_toplayer.pt')
        draw_result_pic(history, add_name="toplayer")


def main(args):
    train_process(args)


if __name__ == '__main__':
    # mbv2 = mbnet(n_types=3)
    # torch.save(mbv2, "mbv2.pt")

    # 命令行参数
    parser = argparse.ArgumentParser()
    # [data]
    parser.add_argument("--data", help="数据所在位置：train / valid", default='./data/data')
    # [model]
    parser.add_argument("--ntypes", help="分类数目.", default=3)
    parser.add_argument("--input_shape", help="模型中的输入尺寸.(图片的resize尺寸)", default='[224,224]')
    # [train]
    parser.add_argument("--epochs", help="解封后训练迭代次数.", default=1)
    parser.add_argument("--batch_size", help="每个epoch中batch_size数目.", default=16)
    parser.add_argument("--lr_patience", help="epochs > lr_patience, \
                        当loss不再变化, 降低lr.", default=10)
    parser.add_argument("--lr_ratio", help="lr 减小的倍率.", default=0.3)
    parser.add_argument("--earlystop_patience", help="ealy stop 停止的次数；\
                        如果 == 0， 则不开启earlystop", default=30)
    parser.add_argument("--train_all", help="是解压全部进行训练，还是只训练最后的fc层.", default=True)

    args = parser.parse_args()
    main(args)
