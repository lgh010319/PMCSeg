# -*- coding: utf-8 -*-
# from os import listdir
# from os.path import join
# import numpy as np
# import SimpleITK as sitk

"""
该代码的目的是计算图像的“均值”和“标准差”，

将在随后的规范化过程中使用



以以“nii.gz”结尾的图像为例（使用SimpleITK）
"""


# def Getmeanstd(args, image_path, meanstd_name):
#     """
#     ：param args：参数
#     ：param image_path：图像地址
#     ：param meanstd_name：保存“mean”和“std”的名称（使用“.npy”格式保存）
#     ：return：无
#     """
#     root_dir = args.root_dir
#     file_names = [x for x in listdir(join(image_path))]
#     mean, std, length = 0.0, 0.0, 0.0

#     for file_name in file_names:
#         image = sitk.ReadImage(image_path + file_name)
#         image = sitk.GetArrayFromImage(image).astype(np.float32)
#         length += image.size
#         mean += np.sum(image)
#         # print(mean, length)
#     mean = mean / length

#     for file_name in file_names:
#         image = sitk.ReadImage(image_path + file_name)
#         image = sitk.GetArrayFromImage(image).astype(np.float32)
#         std += np.sum(np.square((image - mean)))
#         # print(std)
#     std = np.sqrt(std / length)
#     print("1 Finish Getmeanstd: ", meanstd_name)
#     print("Mean and std are: ", mean, std)
#     np.save(root_dir + meanstd_name, [mean, std])
import os
import numpy as np
import SimpleITK as sitk
from os import listdir
from os.path import join
from concurrent.futures import ProcessPoolExecutor
import torch

def process_image(file_path):
    """Process individual images and return their arrays and sizes."""
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    # # 从 .npy 文件读取图像数据
    # image_array = np.load(file_path).astype(np.float32)  # 确保图像数据为 float32 类型

    num_pixels = np.prod(image_array.shape)
    sum_pixels = np.sum(image_array)
    sum_square_pixels = np.sum(np.square(image_array))
    return num_pixels, sum_pixels, sum_square_pixels

def Getmeanstd(args, image_path, meanstd_name):
    """
    Calculate mean and standard deviation of images in a directory using multiprocessing.
    :param args: parameters including 'root_dir'
    :param image_path: path to the directory containing images
    :param meanstd_name: file name to save mean and std (using .npy format)
    """
    root_dir = args.root_dir
    file_names = [join(image_path, x) for x in listdir(image_path)]
    total_pixels = 0
    sum_pixels = 0
    sum_square_pixels = 0

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_image, file_names)
        for num_pixels, sum_pix, sum_square_pix in results:
            total_pixels += num_pixels#计算总像素数量。
            sum_pixels += sum_pix#计算像素和
            sum_square_pixels += sum_square_pix

    mean = sum_pixels / total_pixels
    std = np.sqrt((sum_square_pixels - (sum_pixels**2 / total_pixels)) / total_pixels)
    print("1 Finish Getmeanstd: ", meanstd_name)
    print("Mean and std are: ", mean, std)
    np.save(join(root_dir, meanstd_name), [mean, std])

# Example usage:
# args = {'root_dir': '/path/to/save/'}
# Getmeanstd(args, '/path/to/images/', 'meanstd.npy')
