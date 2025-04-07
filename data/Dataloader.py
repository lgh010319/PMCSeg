# -*- coding: utf-8 -*-
import torch
import gc
from torch.utils import data
import numpy as np
import SimpleITK as sitk
from Data_Augumentation import transform_img_lab
import warnings
import monai

warnings.filterwarnings("ignore")


def read_file_from_txt(txt_path):
    files = []
    for line in open(txt_path, "r"):
        files.append(line.strip())
    return files


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape

    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n))
    categorical[y, np.arange(n)] = 1

    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)
    return categorical


def reshape_img(image, z, y, x):
    out = np.zeros([z, y, x], dtype=np.float32)
    out[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image[0:image.shape[0], 0:image.shape[1],
                                                                0:image.shape[2]]
    return out


def smart_cropping1(image, label, z, y, x, crop_size):  # 智能裁剪，这个代码的问题就是在这个裁剪上
    z, y, x = z, y, x
    min_class1 = int(np.sum(label == 1) * 0)  # 最少类别1数量
    # print(f"min_class1:,{min_class1}")
    tries = 3000  # 尝试次数，防止无限循环
    while tries > 0:
        center_z = np.random.randint(0, z - crop_size[0] + 1, 1, dtype=np.int16)[0]
        center_y = np.random.randint(0, y - crop_size[1] + 1, 1, dtype=np.int16)[0]
        center_x = np.random.randint(0, x - crop_size[2] + 1, 1, dtype=np.int16)[0]
        crop_img = image[center_z:center_z + crop_size[0], center_y:center_y + crop_size[1],
                   center_x:center_x + crop_size[2]]
        crop_lbl = label[center_z:center_z + crop_size[0], center_y:center_y + crop_size[1],
                   center_x:center_x + crop_size[2]]

        # 检查类别1的数量
        if np.sum(crop_lbl == 1) > min_class1:
            return crop_img, crop_lbl
        tries -= 1

    # 如果找不到合适的裁剪区域，返回原始尺寸（可替换为其他逻辑）
    return crop_img, crop_lbl


class Dataloader1(data.Dataset):
    def __init__(self, args):
        super(Dataloader1, self).__init__()
        self.image_file = read_file_from_txt(args.Image_Tr_txt)
        self.label_file = read_file_from_txt(args.Label_Tr_txt)
        self.shape = args.ROI_shape
        self.args = args

    def __getitem__(self, index):
        image_path = self.image_file[index]
        label_path = self.label_file[index]

        # print(f"image_path:{image_path}")
        # print(f"label_path:{label_path}")

        # Read images and labels
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)

        # #从.npy文件读取数据
        # image = np.load(image_path)  # 加载图像数据
        # label = np.load(label_path)  # 加载标签数据
        #
        # # 转换为Tensor
        # image = torch.tensor(image, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.long)  # 假设标签为长整型（分类问题），需根据实际情况调整

        z, y, x = image.shape
        image = image.astype(dtype=np.float32)
        label = label.astype(dtype=np.float32)
        # print(f"label0: {label.shape}")
        # print(f"label0: {np.sum(label == 1)}")
        # print(f"image: {image.shape}")
        # print(f"label: {label.shape}")

        # Normalization
        mean, std = np.load(self.args.root_dir + self.args.Tr_Meanstd_name)
        image = (image - mean) / std

        if self.shape[0] > z:
            z = self.shape[0]
            image = reshape_img(image, z, y, x)
            label = reshape_img(label, z, y, x)
        if self.shape[1] > y:
            y = self.shape[1]
            image = reshape_img(image, z, y, x)
            label = reshape_img(label, z, y, x)
        if self.shape[2] > x:
            x = self.shape[2]
            image = reshape_img(image, z, y, x)
            label = reshape_img(label, z, y, x)

        crop_size = [self.shape[0], self.shape[1], self.shape[2]]  # 裁剪尺寸
        # print(f"crop_size: {crop_size}")
        image, label = smart_cropping1(image, label, z, y, x, crop_size)
        # Random crop, (center_y, center_x) refers the left-up coordinate of the Random_Crop_Block随机裁剪
        # center_z = np.random.randint(0, z - self.shape[0] + 1, 1, dtype=np.int16)[0]
        # center_y = np.random.randint(0, y - self.shape[1] + 1, 1, dtype=np.int16)[0]
        # center_x = np.random.randint(0, x - self.shape[2] + 1, 1, dtype=np.int16)[0]
        # image = image[center_z:self.shape[0] +
        #                        center_z, center_y:self.shape[1] + center_y, center_x:self.shape[2] + center_x]
        # label = label[center_z:self.shape[0] +
        #                          center_z, center_y:self.shape[1] + center_y, center_x:self.shape[2] + center_x]
        # print(f"label0: {label.shape}")
        # print(f"label0: {np.sum(label == 1)}")

        image = image[np.newaxis, :, :, :]  # b c d w h
        label = label[np.newaxis, :, :, :]  # b c d w h
        # print(f"image: {image.shape}")
        # print(f"label: {label.shape}")
        # crop_transform = RandCropByPosNegLabeld(
        #     keys=['image', 'label'],  # 处理图像和标签
        #     label_key='label',
        #     spatial_size=crop_size,  # 裁剪尺寸
        #     pos=1,  # 100% 概率选择正样本区域
        #     neg=0,  # 0% 概率选择负样本区域
        #     num_samples=1  # 每次变换生成一个样本
        # )
        # # # crop_transform = RandSpatialCropd(keys=['image', 'label'], roi_size=crop_size, random_size=False, lazy=True)
        # # mode = ("trilinear", "nearest")
        # # crop_transform = ResizeD(keys=['image', 'label'],spatial_size=crop_size, mode=mode, lazy=True)
        # sample = {'image': image, 'label': label}
        # cropped_sample = crop_transform(sample)
        # # # # print(cropped_sample)
        # image = cropped_sample[0]['image']#列表里面有个字典
        # label = cropped_sample[0]['label']
        # image = cropped_sample['image']
        # label = cropped_sample['label']

        # image = image[np.newaxis, :, :, :]  # b c d w h
        # label = label[np.newaxis, :, :, :]  # b c d w h
        # print(f"image: {image.shape}")
        # print(f"label: {label.shape}")

        # Data Augmentation
        data_dict = transform_img_lab(image, label, self.args)
        image_trans = data_dict["image"]
        label_trans = data_dict["label"]
        if isinstance(image_trans, torch.Tensor):
            image_trans = image_trans.numpy()
        if isinstance(label_trans, torch.Tensor):
            label_trans = label_trans.numpy()

        # Only focus on vessels ...
        # label_trans = np.where(label_trans == 2, 0, label_trans)
        # label_trans = np.where(label_trans == 3, 2, label_trans)
        # label_trans = np.where(label_trans == 4, 0, label_trans)
        # label_trans = to_categorical(label_trans[0], 2)#这个是one-hot编码
        label_trans = to_categorical(label_trans[0], 2)  # 这个是one-hot编码
        # print(f"label_trans: {np.sum(label_trans == 1)}")
        # label_trans = to_categorical(label_trans[0], 3)#

        return image_trans, label_trans

    def __len__(self):
        return len(self.image_file)
