# -*- coding: utf-8 -*-
import os
import gc
import json
from tkinter import Label

import time  # 导入 time 模块
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import logging
import numpy as np
from os.path import join
import SimpleITK as sitk
from datetime import datetime
#from torch.utils.data import DataLoader
from monai.data import DataLoader
from sympy import false


from models.get_baseline import get_model #获取不同的模型

from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib
import torchio as tio
from tqdm import tqdm
import nibabel as nib

from torch.optim.lr_scheduler import LambdaLR
from data.Dataloader import Dataloader,Dataloader1
from losses.Loss import cross_loss,WeightedCrossEntropyLoss,CombinedLoss,dice_loss

import warnings
import wandb

warnings.filterwarnings("ignore")

# Use <AverageMeter> to calculate the mean in the process
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def Dice_one_hot(label, output, class_index=1):
    eps = 1e-6
    label = torch.nn.functional.one_hot(label.squeeze(dim=1).long(), num_classes=2).permute(0, 4, 1, 2, 3)
    # 选择感兴趣的类别通道
    label = label[:, class_index, ...]  # 假设类别索引为1
    # print(label)
    output = output[:, class_index, ...]
    output = (output > 0.5).float()

    dims = (1, 2, 3)

    intersection = torch.sum(label * output, dims)
    cardinality = torch.sum(label  + output, dims)
    dice_score = (2. * intersection + eps) / (cardinality + eps)
    # print(f"dice: {dice_score}")
    print(f"dice: {dice_score.mean()}")
    return dice_score.mean()

    # # 应用阈值将概率转换为二值图像
    # output = (output > 0.5).float()
    # # print(output)
    #
    # # 将标签转换为布尔型，如果标签已经是二值的，这一步可能不需要
    # label = label.bool()
    # output = output.bool()
    #
    # # 计算交集
    # intersection = (label & output).sum()
    # print(f"Intersection: {intersection.item()}")
    #
    # # 计算label和output中值为1的元素的和
    # label_sum = label.sum()
    # print(f"Label sum: {label_sum.item()}")
    # output_sum = output.sum()
    # print(f"Output sum: {output_sum.item()}")
    #
    # # 计算Dice系数
    # dice = (2. * intersection + eps) / (label_sum + output_sum + eps)
    # print(f"dice: {dice}")
    return dice

def extract_data_from_patch_test(patch):
    volume = patch['data'][tio.DATA].float().cuda()
    # gt = patch['dense'][tio.DATA].float().cuda()
    images = volume

    emb_codes = torch.cat((
        patch[tio.LOCATION][:,:3],
        patch[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
    ), dim=1).float().cuda()#patch[tio.LOCATION][:,:3] 的前三列假设表示图像块的起始位置和后面的结束位置
    # return images, gt, emb_codes
    return images, emb_codes

# One epoch in training process
def train_epoch(model, loader, optimizer, criterion, epoch, n_epochs, logger):#优化器，用于更新模型的参数）、损失函数，用于计算预测和真实标签之间的差异
    losses = AverageMeter()
    losses1 = AverageMeter()
    dices = AverageMeter()
    model.train()
    # for batch_idx, (image, label) in enumerate(loader):#开始遍历数据加载器提供的每个数据批次。batch_idx是当前批次的索引，image和label分别代表图像数据和对应的标签

    for batch_idx, (image, label, gt_dist) in enumerate(loader):
    # for batch_idx, (image, label) in enumerate(loader):
        epoch_start_time = time.time()  # 记录开始时间
        # print(image.shape)
        # print(label.shape)
        # print(f"Batch {batch_idx}: {data}")  # 查看返回的数据结构
        # image, label = data  # 这行可能会抛出错误
        if torch.cuda.is_available():
            image, label,gt_dist = image.cuda(), label.cuda(),gt_dist.cuda()
        gt_count = torch.sum(label == 1, dim=list(range(1, label.ndim)))
        print(gt_count)

        optimizer.zero_grad()#在每次迭代开始时，将优化器和模型的梯度清零。这是必要的，因为PyTorch会累积梯度，如果不清零，新的梯度会加到旧的梯度上。
        model.zero_grad()
        # 前向传播
        
        output = model(image)  # 现在在此上下文中计算模型输出时不会计算梯度
        loss = criterion(output, label,gt_dist)
        # loss = criterion(output, label)

        losses.update(loss.data,label.size(0))  # 使用AverageMeter对象更新损失的平均值。loss.data是损失的值，label.size(0)是这一批次中标签的数量（即这一批次的样本数）
        # 然后反向传播分割损失
        loss.backward()  # 完成分割任务的反向传播
        optimizer.step()  # 更新分割网络参数
        # torch.cuda.empty_cache()
        # 定义权重，例如主输出的损失权重为1.0，中间输出的损失权重为0.5
        # weights = [1.0, 0.5, 0.5, 0.5,0.5]  # 可以根据输出数量调整权重
        #
        # outputs = model(image)
        # total_loss = 0
        # # 针对每个输出都计算损失
        # for i in range(outputs.shape[1]):
        #     output = outputs[:, i]  # 取出第i个输出，形状为 [B, C, H, W, D]
        #     # print(f"output:{output.shape}")
        #     loss = criterion(output, label)  # 每个输出都与同一个 label 计算损失
        #     total_loss += weights[i] * loss  # 加权累加损失
        #
        # # 反向传播
        # total_loss.backward()
        # optimizer.step()
        # losses.update(total_loss.data,label.size(0))  # 使用AverageMeter对象更新损失的平均值。loss.data是损失的值，label.size(0)是这一批次中标签的数量（即这一批次的样本数）

        output= F.softmax(output,dim=1)
        dice = Dice_one_hot(label, output)
        dices.update(dice.data, label.size(0))
        # dice = 1-loss
        # dices.update(dice, label.size(0))


        #loss = criterion(label, output, 0)#使用提供的损失函数criterion计算真实标签和模型输出之间的损失。
        # loss = criterion(label, output)  # 使用提供的损失函数criterion计算真实标签和模型输出之间的损失。
        # losses.update(loss.data, label_patches.size(0))#使用AverageMeter对象更新损失的平均值。loss.data是损失的值，label.size(0)是这一批次中标签的数量（即这一批次的样本数）。

        # 在每个 batch 的末尾插入
        # torch.cuda.empty_cache()  # 释放未使用的缓存内存
        epoch_end_time = time.time()  # 记录结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # 计算时间差

        # 格式化时间，例如转为秒
        epoch_duration_str = "{:.2f}s".format(epoch_duration)  # 保留两位小数的秒数
        res = "\t".join(
            [
                "Epoch: [%d/%d]" % (epoch , n_epochs),
                "Iter: [%d/%d]" % (batch_idx + 1, len(loader)),
                "Lr: [%f]" % (optimizer.param_groups[0]["lr"]),
                # "Loss1 %f" % (losses1.avg),
                "Loss %f" % (losses.avg),
                "Dice: %f" % (dices.avg),  # 加入 Dice 值
                "Time: %s" % epoch_duration_str  # 加入当前 epoch 花费的时间
            ]
        )
        # logger.info(res)#记录到日志里面logger.info(res)#记录到日志里面
        print(res)
    return losses.avg, dices.avg#返回这一周期的平均损失值和dice值


# Generate the log
def Get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# Train process
def Train_net(net,args):
    dice_mean, dice_m, dice_v, dice_a, dice_e, dice_max = 0, 0, 0, 0, 0, 0

    # Determine if trained parameters exist
    if not args.if_retrain and os.path.exists(
        os.path.join(args.Dir_Weights, args.model_name)
    ):#如果args.if_retrain为False，且args.Dir_Weights目录下存在一个与args.model_name同名的文件（即预训练模型权重文件）。
        net.load_state_dict(torch.load(os.path.join(args.Dir_Weights, args.model_name)))
        print(os.path.join(args.Dir_Weights, args.model_name))
    if torch.cuda.is_available():
        net = net.cuda()
    # gpus = [0] if torch.cuda.is_available() else []
    # net = net.to(gpus[0])

    # Load dataset

    train_dataset = Dataloader1(args)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers, pin_memory=True
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95))

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.8, patience=10)


    criterion = dice_loss()


    #日志
    training_log_filename = os.path.join(args.Dir_Log, "training_log.csv")
    training_log = list()
    training_log_header = ["epoch", "loss", "acc", "lr"]
    dt = datetime.today()
    log_name = (
        str(dt.date())
        + "_"
        + str(dt.time().hour)
        + ":"
        + str(dt.time().minute)
        + ":"
        + str(dt.time().second)
        + "_"
        + args.log_name
    )
    logger = Get_logger(args.Dir_Log + log_name)
    logger.info("start training!")

    train_losses = []
    train_losses1 = []
    train_acces = []
    # Main train process
    for epoch in range(args.start_train_epoch, args.n_epochs+1):
        loss, dice_e= train_epoch(
            net, train_dataloader, optimizer, criterion, epoch, args.n_epochs, logger
        )
        train_losses.append(loss.item())
        train_acces.append(dice_e.item())
        torch.save(net.state_dict(), os.path.join(args.Dir_Weights, args.model_name))
        # torch.save(net1.state_dict(), os.path.join(args.Dir_Weights, args.model_name1))

        #scheduler.step(loss)
        # lr_scheduler.step(dice_max)  # 更新学习率
        if epoch < args.start_verify_epoch:
            lr_scheduler.step(dice_e)
        else:
            lr_scheduler.step(dice_max)

        if epoch >= args.start_verify_epoch:
            net.load_state_dict(
                torch.load(os.path.join(args.Dir_Weights, args.model_name))  # 这个就是用的最后一个权值，因为每次都更新，没什么意义啊
            )

            predict(net, args.Image_Ts_txt, args.save_path, args)
            # Calculate the Dice
            dice = Dice(args.Label_Ts_txt, args.save_path, logger)
            dice_mean = np.mean(dice)
            if dice_mean > dice_max:
                dice_max = dice_mean
                torch.save(
                    net.state_dict(),
                    os.path.join(args.Dir_Weights, args.model_name_max),
                )

        logger.info(
            "Epoch:[{}/{}]  lr={:.6f}  loss={:.5f}  dice_e={:.4f} "
            "max_dice={:.4f}".format(
                epoch,
                args.n_epochs,
                optimizer.param_groups[0]["lr"],
                loss,
                dice_e,
                dice_max,
            )
        )
        # update the training log
        training_log.append([epoch, loss.item(), dice_e.item(), optimizer.param_groups[0]["lr"]])
        pd.DataFrame(training_log, columns=training_log_header).set_index("epoch").to_csv(training_log_filename)

        # 绘制训练损失和准确率
        matplotlib.use('Agg')
        plt.subplot(1, 1, 1)  # 1行2列的第1个
        # print("Figure size before resizing:", plt.gcf().get_size_inches())
        plt.plot(np.arange(len(train_losses)), train_losses, label="Train Loss")
        plt.plot(np.arange(len(train_acces)), train_acces, label="Train acc")
        plt.xlabel('Epochs')
        plt.ylabel("Loss")
        plt.title('Training accuracy&loss')
        plt.legend()
        # 保存图像
        plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
        plt.savefig(f'accuracy_loss_epoch.png')
        plt.show()  # 显示图像
        plt.close()  # 关闭图形，避免过多图形打开
        plt.clf()  # 清除图形
    logger.info("finish training!")


def read_file_from_txt(txt_path):  # 从txt里读取数据
    files = []
    for line in open(txt_path, "r"):
        files.append(line.strip())
    return files


def reshape_img(image, z, y, x):
    out = np.zeros([z, y, x], dtype=np.float32)
    out[0 : image.shape[0], 0 : image.shape[1], 0 : image.shape[2]] = image[
        0 : image.shape[0], 0 : image.shape[1], 0 : image.shape[2]
    ]
    return out

def prepare_map_kernel(shape):
    """Prepare the map kernel based on the given shape."""
    a = np.zeros(shape=shape)
    a = np.where(a == 0)
    map_kernel = 1 / (
        (a[0] - shape[0] // 2) ** 4
        + (a[1] - shape[1] // 2) ** 4
        + (a[2] - shape[2] // 2) ** 4
        + 1
    )
    return np.reshape(map_kernel, newshape=(1, 1,) + shape)

def predict(model, image_dir, save_path, args):
    print("Predict test data")
    model.eval()
    file = read_file_from_txt(image_dir)
    file_num = len(file)
    # # Prepare map_kernel once
    # map_kernel = prepare_map_kernel(args.ROI_shape)

    for t in range(file_num):
        image_path = file[t]
        print(image_path)

        image1 = sitk.ReadImage(image_path)
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        image = image.astype(np.float32)

        name = image_path[image_path.rfind("/") + 1 :]
        mean, std = np.load(args.root_dir + args.Te_Meanstd_name)
        image = (image - mean) / std##归一化图像：使用预先计算的均值和标准差归一化图像数据。
        z, y, x = image.shape
        z_old, y_old, x_old = z, y, x#记录原始尺寸：存储原始的图像尺寸用于最后的裁剪

        if args.ROI_shape[0] > z:#尺寸调整：确保图像的深度（z）、高度（y）和宽度（x）至少与 ROI 形状相匹配。
            z = args.ROI_shape[0]
            image = reshape_img(image, z, y, x)
        if args.ROI_shape[1] > y:
            y = args.ROI_shape[1]
            image = reshape_img(image, z, y, x)
        if args.ROI_shape[2] > x:
            x = args.ROI_shape[2]
            image = reshape_img(image, z, y, x)

        predict = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)#为预测结果和对应的权重映射初始化全零数组。
        n_map = np.zeros([1, args.n_classes, z, y, x], dtype=np.float32)

        """
        Our prediction is carried out using sliding patches,
        and for each patch a corresponding result is predicted,
        and for the part where the patches overlap,
        we use weight <map_kernel> balance,
        and we agree that the closer to the center of the patch, the higher the weight
        我们的预测是使用滑动补丁进行的，
        并且对于每个补丁预测相应的结果，
        并且对于贴片重叠的部分，
        我们使用权重<map_kernel>平衡，
        我们一致认为，离贴片中心越近，重量就越高
        """

        shape = args.ROI_shape
        a = np.zeros(shape=shape)#整合到外面去了
        a = np.where(a == 0)
        map_kernal = 1 / (
            (a[0] - shape[0] // 2) ** 4
            + (a[1] - shape[1] // 2) ** 4
            + (a[2] - shape[2] // 2) ** 4
            + 1
        )
        map_kernal = np.reshape(map_kernal, newshape=(1, 1,) + shape)

        # print(np.max(map_kernal))
        image = image[np.newaxis, np.newaxis, :, :, :]#添加维度：为图像添加额外的批处理和通道维度，以符合 PyTorch 模型的输入要求。

        stride_x = shape[0] // 2#计算步长：定义在各个维度上移动图像块的步长。
        stride_y = shape[1] // 2
        stride_z = shape[2] // 2
        for i in range(z // stride_x - 1):
            for j in range(y // stride_y - 1):
                for k in range(x // stride_z - 1):
                    image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                              k * stride_z:k * stride_z + shape[2]]
                    image_i = torch.from_numpy(image_i)
                    if torch.cuda.is_available():
                        image_i = image_i.cuda()
                    with torch.no_grad():  # 不计算梯度
                        output = model(image_i)
                    output = output.data.cpu().numpy()

                    predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                    k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                    n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                    k * stride_z:k * stride_z + shape[2]] += map_kernal

                image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                          x - shape[2]:x]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                with torch.no_grad():  # 不计算梯度
                    output = model(image_i)
                output = output.data.cpu().numpy()
                predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                x - shape[2]:x] += output * map_kernal

                n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                x - shape[2]:x] += map_kernal

            for k in range(x // stride_z - 1):
                image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                          k * stride_z:k * stride_z + shape[2]]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                with torch.no_grad():  # 不计算梯度
                    output = model(image_i)
                output = output.data.cpu().numpy()
                predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                k * stride_z:k * stride_z + shape[2]] += map_kernal

            image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            with torch.no_grad():  # 不计算梯度
                output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += output * map_kernal
            n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += map_kernal

        for j in range(y // stride_y - 1):
            for k in range((x - shape[2]) // stride_z):
                image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                          k * stride_z:k * stride_z + shape[2]]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                with torch.no_grad():  # 不计算梯度
                    output = model(image_i)
                output = output.data.cpu().numpy()

                predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += map_kernal

            image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                      x - shape[2]:x]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            with torch.no_grad():  # 不计算梯度
                output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += output * map_kernal

            n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += map_kernal

        for k in range(x // stride_z - 1):
            image_i = image[:, :, z - shape[0]:z, y - shape[1]:y,
                      k * stride_z:k * stride_z + shape[2]]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            with torch.no_grad():  # 不计算梯度
                output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, z - shape[0]:z, y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += output * map_kernal

            n_map[:, :, z - shape[0]:z, y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += map_kernal

        image_i = image[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x]
        image_i = torch.from_numpy(image_i)
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        with torch.no_grad():  # 不计算梯度
            output = model(image_i)
        output = output.data.cpu().numpy()

        predict[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += output * map_kernal
        n_map[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += map_kernal

        predict = predict / n_map
        predict = np.argmax(predict[0], axis=0)
        predict = predict.astype(np.uint16)
        out = predict[0:z_old, 0:y_old, 0:x_old]
        out = sitk.GetImageFromArray(out)
        out.SetOrigin(image1.GetOrigin())
        out.SetDirection(image1.GetDirection())
        out.SetSpacing(image1.GetSpacing())
        sitk.WriteImage(out, join(save_path, name))
    torch.cuda.empty_cache()
    gc.collect()
    print("finish!")
def Dice(label_dir, pred_dir,logger):
    # 获取image文件索引
    file = read_file_from_txt(label_dir)
    file_num = len(file)
    i = 0
    dice_score = np.zeros(shape=(file_num), dtype=np.float32)

    for t in range(file_num):
        image_path = file[t]
        name = image_path[image_path.rfind('/') + 1:]
        predict = sitk.ReadImage(join(pred_dir, name))
        predict = sitk.GetArrayFromImage(predict)

        groundtruth = sitk.ReadImage(image_path)
        groundtruth = sitk.GetArrayFromImage(groundtruth)

        # 将预测和标签二值化，只关注标签为1的部分
        predict_binary = np.where(predict == 1, 1, 0).flatten()
        groundtruth_binary = np.where(groundtruth == 1, 1, 0).flatten()

        # 计算Dice系数
        tmp = predict_binary + groundtruth_binary
        a = np.sum(np.where(tmp == 2, 1, 0))  # 预测和真实标签都为1的像素数
        b = np.sum(predict_binary)  # 预测为1的像素总数
        c = np.sum(groundtruth_binary)  # 真实标签为1的像素总数
        dice_score[i] = (2 * a) / (b + c) if (b + c) != 0 else 1  # 防止除以零的情况

        print(name, dice_score[i])
        logger.info(
            "{} dice={:.4f}".format(name, dice_score[i]
            )
        )
        i += 1

    return dice_score




def Create_files(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.save_path_max):
        os.mkdir(args.save_path_max)


def Predict_Network(net, args):
    dt = datetime.today()
    log_name = (
            str(dt.date())
            + "_"
            + str(dt.time().hour)
            + ":"
            + str(dt.time().minute)
            + ":"
            + str(dt.time().second)
            + "_"
            + args.log_name
    )
    logger = Get_logger(args.Dir_Log + log_name)
    logger.info("start Predict!")
    if torch.cuda.is_available():
        net = net.cuda()
    try:
        net.load_state_dict(
            torch.load(os.path.join(args.Dir_Weights, args.model_name_max))#导入最好的模型
        )
        print(os.path.join(args.Dir_Weights, args.model_name_max))
    except:
        print(
            "Warning 100: No parameters in weights_max, here use parameters in weights"
        )
        # net.load_state_dict(torch.load(os.path.join(args.Dir_Weights, args.model_name)))
        print(os.path.join(args.Dir_Weights, args.model_name))
    predict(net, args.Image_Te_txt, args.save_path_max, args)
    dice = Dice(args.Label_Te_txt, args.save_path_max,logger)

    # predict(net, args.Image_Ts_txt, args.save_path_max, args)
    # dice = Dice(args.Label_Ts_txt, args.save_path_max, logger)
    dice_mean = np.mean(dice)
    print(dice_mean)

    logger.info(
        "dice_mean={:.4f} ".format(
            dice_mean,
        )
    )
    logger.info("finish Predict!")


def Train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net = DSCNet(
    #     n_channels=args.n_channels,
    #     n_classes=args.n_classes,
    #     kernel_size=args.kernel_size,
    #     extend_scope=args.extend_scope,
    #     if_offset=args.if_offset,
    #     device=device,
    #     number=args.n_basic_layer,
    #     dim=args.dim,
    # )
    print(f"args.name:{args.name}")
    net = get_model(args.name, args.att, args.n_channels, args.n_classes)
    # 如果有多个 GPU，将模型包装为 DataParallel
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     net = torch.nn.DataParallel(net,device_ids = [0,1])
    Create_files(args)
    if not args.if_onlytest:
        Train_net(net ,args)
        Predict_Network(net,args)
    else:
        Predict_Network(net,args)



