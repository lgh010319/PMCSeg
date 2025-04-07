import os
import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize
from scipy.ndimage import sobel, generic_gradient_magnitude
import matplotlib.pyplot as plt
import pywt



def read_nifti_file(file_path):
    # 使用 SimpleITK 读取 nii.gz 文件
    sitk_image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(sitk_image)
    return array


def process_images(image_dir, mask_dir, output_dir,output_dir1):
    images = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir))]
    masks = [os.path.join(mask_dir, x) for x in sorted(os.listdir(mask_dir))]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    #os.makedirs(output_dir1, exist_ok=True)

    for idx, (image_path, mask_path) in enumerate(zip(images, masks)):
        standard = sitk.ReadImage(image_path)
        image = read_nifti_file(image_path)
        mask = read_nifti_file(mask_path)
        filename = os.path.basename(image_path)

        print(filename)

        # 确保 image 和 mask 成功加载
        if image is None or mask is None:
            raise FileNotFoundError(f"Unable to load files. Image: {image_path}, Mask: {mask_path}")

        # 假设 data 是你的三维数据
        coeffs = pywt.dwtn(image, wavelet='db4', mode='periodization')

        # 增强所有三个方向的高频细节
        coeffs['ddd'] *= 2  # 增强细节

        # 重建图像
        reconstructed_data = pywt.idwtn(coeffs, wavelet='db4', mode='periodization')

        # 保存中心线和边缘线为 nii.gz

        save_nii(reconstructed_data, standard,output_dir, filename)
        #save_nii(edges, output_dir1, filename)


def save_nii(img_array, standard, output_dir, filename):
    img = sitk.GetImageFromArray(img_array.astype(np.int16))
    img.SetOrigin(standard.GetOrigin())
    img.SetDirection(standard.GetDirection())
    img.SetSpacing(standard.GetSpacing())
    file_path = os.path.join(output_dir, filename)
    print(f"Saving NIfTI file to {file_path}")  # 打印保存路径
    sitk.WriteImage(img, file_path)


if __name__ == "__main__":
    args = {
        "image_dir": "/home/robot/shiyan_code/DSCNet-main/DSCNet_3D_opensource/Data/upper1/train/image/",
        "mask_dir": "/home/robot/shiyan_code/DSCNet-main/DSCNet_3D_opensource/Data/upper1/train/label/",
        "output_dir": "/home/robot/shiyan_code/DSCNet-main/DSCNet_3D_opensource/Data/upper1/train/image1/",
        "output_dir1": "/home/robot/桌面/AI医疗/DSCNet-main/DSCNet_3D_opensource/Data/upper/train/edge/"
    }
    process_images(**args)
