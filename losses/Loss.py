# -*- coding: utf-8 -*-

import torch
import monai.losses

from torch import nn
from torch.nn.functional import max_pool3d
from monai.losses import DiceLoss

class crossentry(nn.Module):#实现交叉熵损失

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):#模型的前向传播
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth))#y_pred 应该是模型输出的概率值，但在交叉熵损失中，我们通常对 y_pred 使用 softmax 或 sigmoid 来确保其在 [0, 1] 之间。


class cross_loss(nn.Module):#现标准的二分类交叉熵损失。

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        smooth = 1e-6
        return -torch.mean(y_true * torch.log(y_pred + smooth) +
                           (1 - y_true) * torch.log(1 - y_pred + smooth))#它同时考虑了 y_true 为正和为负的情况。使用了 smooth 来确保数值稳定性。


class dice_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, label, output, class_index=1):
        eps = 1e-6
        # 选择感兴趣的类别通道
        label = label[:, class_index, ...]  # 假设类别索引为1
        # print(label)
        output = output[:, class_index, ...]
        # print(output)

        #将输入和标签展平为一维张量，以便于计算
        output = output.view(-1)
        label = label.view(-1)
        # 计算交集
        intersection = (output * label).sum()

        # 计算label和output中值为1的元素的和
        label_sum = label.sum()
        output_sum = output.sum()

        # 计算Dice系数
        dice = (2. * intersection + eps) / (label_sum + output_sum + eps)

        ## Compute Dice loss
        return 1-dice


'''
Another Loss Function proposed by us in IEEE transactions on Image Precessing:
Paper: https://ieeexplore.ieee.org/abstract/document/9611074
Code: https://github.com/YaoleiQi/Examinee-Examiner-Network
'''


class Dropoutput_Layer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, alpha=0.4):
        smooth = 1e-6
        w = torch.abs(y_true - y_pred)
        w = torch.round(w + alpha)
        loss_ce = (
            -((torch.sum(w * y_true * torch.log(y_pred + smooth)) /
               torch.sum(w * y_true + smooth)) +
              (torch.sum(w * (1 - y_true) * torch.log(1 - y_pred + smooth)) /
               torch.sum(w * (1 - y_true) + smooth))) / 2)
        return loss_ce


import torch
import torch.nn.functional as F

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self,weight):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.weight is not None:
            self.weight = self.weight.to(device)

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight)

# class CombinedLoss(torch.nn.Module):
#     def __init__(self,weight):
#         super(CombinedLoss, self).__init__()
#         self.weight = weight
#         self.weighted_ce = WeightedCrossEntropyLoss(weight)
#         self.dice_loss = dice_loss()
#
#     def forward(self, inputs, targets,n):
#         n=float(n*0.002)
#         print(f"n:,{n}")
#         ce_loss = self.weighted_ce(inputs, targets)
#         dice_loss = self.dice_loss(inputs, targets)
#         return n*ce_loss + (1-n)*dice_loss


class CombinedLoss(torch.nn.Module):
    def __init__(self,weight):
        super(CombinedLoss, self).__init__()
        self.weight = weight
        self.weighted_ce = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(include_background=True, sigmoid=False, softmax=False)

    def forward(self, inputs, targets):
        output = F.softmax(inputs, dim=1)
        dice_loss = self.dice_loss(output, targets)
        targets = torch.argmax(targets, dim=1)
        ce_loss = self.weighted_ce(inputs, targets)
        return ce_loss + dice_loss






