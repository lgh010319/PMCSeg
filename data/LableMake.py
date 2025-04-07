import SimpleITK as sitk
import numpy as np
import os

def cropGenerate(volumePath,RightMapPath,Right_LowerPath,LeftMapPath,Left_LowerPath,cropMapPath,targetPath):
    lower = sitk.ReadImage(volumePath)
    lower_array = sitk.GetArrayFromImage(lower)

    # 读取四个标签映射
    labelMap1 = sitk.ReadImage(RightMapPath)
    labelMap2 = sitk.ReadImage(Right_LowerPath)
    labelMap3 = sitk.ReadImage(LeftMapPath)
    labelMap4 = sitk.ReadImage(Left_LowerPath)

    Right_array = sitk.GetArrayFromImage(labelMap1)
    count = np.sum(Right_array == 1)
    print(count)
    # count = np.sum(Right_array == 0)
    # print(count)
    # row,col,high=np.nonzero(Right_array)
    # print(row,col,high)
    # print(Right_array[row,col,high])
    Right_Lower_array = sitk.GetArrayFromImage(labelMap2)
    Left_array = sitk.GetArrayFromImage(labelMap3)
    Left_Lower_array = sitk.GetArrayFromImage(labelMap4)

    target_array = (Right_array + Right_Lower_array + Left_array + Left_Lower_array).astype(lower_array.dtype)
    #target_array = (Right_array + Left_array).astype(lower_array.dtype)
    count = np.sum(target_array == 2)
    print(count)
    target = sitk.GetImageFromArray(target_array)
    #print(target_array.dtype)

    target.SetOrigin(lower.GetOrigin())
    target.SetDirection(lower.GetDirection())
    target.SetSpacing(lower.GetSpacing())

    sitk.WriteImage(target, targetPath)  # 保存图像

if __name__=='__main__':
    begin = 76
    end = 76
    baseVolumePath = '/home/robot/shiyan data/data/NIIDATA/{:0>3d}/labelMap/skull.nii.gz'
    baseRightMapPath = '/home/robot/shiyan data/data/NIIDATA/{:0>3d}/labelMap/right.nii.gz'
    baseRight_LowerPath = '/home/robot/shiyan data/data/NIIDATA/{:0>3d}/labelMap/right_lower.nii.gz'
    baseLeftMapPath = '/home/robot/shiyan data/data/NIIDATA/{:0>3d}/labelMap/left.nii.gz'
    baseLeft_LowerPath = '/home/robot/shiyan data/data/NIIDATA/{:0>3d}/labelMap/left_lower.nii.gz'

    baseCropMapPath = '/home/robot/shiyan data/data/NIIDATA/{:0>3d}/labelMap/segmented_image.nii.gz'

    #baseTargetPath = '/home/robot/shiyan data/data/NIIDATA/{:0>3d}/label/lower_label.nii.gz'
    baseTargetPath = '/home/robot/shiyan data/data/NIIDATA/{:0>3d}/label/all_label1.nii.gz'
    for i in range(begin,end+1):
        volumePath = baseVolumePath.format(i)
        RightMapPath = baseRightMapPath.format(i)
        Right_LowerPath = baseRight_LowerPath.format(i)
        LeftMapPath = baseLeftMapPath.format(i)
        Left_LowerPath = baseLeft_LowerPath.format(i)

        cropMapPath = baseCropMapPath.format(i)
        targetPath = baseTargetPath.format(i)
        print(i,end=' ')
        if os.path.exists(targetPath):
            print("have been converted")
            cropGenerate(volumePath, RightMapPath, Right_LowerPath, LeftMapPath, Left_LowerPath, cropMapPath,targetPath)
            continue
        if os.path.exists(volumePath) and os.path.exists(RightMapPath) and os.path.exists(Right_LowerPath) and os.path.exists(LeftMapPath) and os.path.exists(Left_LowerPath):
            print("begin to convert")
            cropGenerate(volumePath,RightMapPath,Right_LowerPath,LeftMapPath,Left_LowerPath,cropMapPath,targetPath)
        else:
            print("No")
        # break

