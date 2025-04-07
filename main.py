# -*- coding: utf-8 -*-
import os
import argparse
import sys
sys.path.append('/home/robot/shiyan_code/DSCNet-main')
from data.Pre_Getmeanstd import Getmeanstd
from data.Pre_Generate_Txt import Generate_Txt
from Train_Process import Train

"""
This code contains all the "Parameters" for the entire project -- <DSCNet>
Code Introduction: (The easiest way to run a code!)
    !!! You just need to change lines with "# todo" to get straight to run
    !!! Our code is encapsulated, but it also provides some test interfaces for debugging
    !!! If you want to change the dataset, you can change "KIPA" to other task name
    
KIPA22 [1-4] challenge (including simulataneous segmentation of arteries and veins) is used as 
a public 3D dataset to further validate our method
Challenge: https://kipa22.grand-challenge.org/

[1] He, Y. et. al. 2021. Meta grayscale adaptive network for 3D integrated renal structures segmentation. 
Medical image analysis 71, 102055.
[2] He, Y. et. al. 2020. Dense biased networks with deep priori anatomy and hard region adaptation: 
Semisupervised learning for fine renal artery segmentation. Medical Image Analysis 63, 101722.
[3] Shao, P. et. al. 2011. Laparoscopic partial nephrectomy with segmental renal artery clamping: 
technique and clinical outcomes. European urology 59, 849–855.
[4] Shao, P. et. al. 2012. Precise segmental renal artery clamping under the guidance of dual-source computed 
tomography angiography during laparoscopic partial nephrectomy. European urology 62, 1001–1008.
"""


def Create_files(args):
    print("0 Start all process ...")
    if not os.path.exists(args.Dir_Txt):
        os.makedirs(args.Dir_Txt)
    if not os.path.exists(args.Dir_Log):
        os.makedirs(args.Dir_Log)
    if not os.path.exists(args.Dir_Save):
        os.makedirs(args.Dir_Save)
    if not os.path.exists(args.Dir_Weights):
        os.makedirs(args.Dir_Weights)


def Process(args):
    # step 0: Prepare all files in this projects
    Create_files(args)
    # #Step 1: Prepare image and calculate the "mean" and "std" for normalization
    # Getmeanstd(args, args.Tr_Image_dir, args.Tr_Meanstd_name)
    # Getmeanstd(args, args.Te_Image_dir, args.Te_Meanstd_name)
    Getmeanstd(args, args.Ts_Image_dir, args.Ts_Meanstd_name)
    # #
    # #
    # # # Step 2: Prepare ".txt" files for training and testing data
    # Generate_Txt(args.Tr_Image_dir, args.Image_Tr_txt)
    # Generate_Txt(args.Tr_Label_dir, args.Label_Tr_txt)
    # Generate_Txt(args.Te_Image_dir, args.Image_Te_txt)
    # Generate_Txt(args.Te_Label_dir, args.Label_Te_txt)
    Generate_Txt(args.Ts_Image_dir, args.Image_Ts_txt)
    Generate_Txt(args.Ts_Label_dir, args.Label_Ts_txt)


    # Step 3: Train the "Network"
    Train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # "root_dir" refers to the address of the outermost code, and "***" needs to be replaced
    root_dir = "/home/robot/shiyan_code/DSCNet-main/DSCNet_3D_opensource/"                              # todo
    data_dir = '/home/robot/shiyan_code/DSCNet-main/DSCNet_3D_opensource/Data/lower/'                                         # todo
    parser.add_argument(
        "--root_dir", default=root_dir, help="the address of the outermost code"
    )

    # information about the image and label
    parser.add_argument(
        "--Tr_Image_dir",
        default=data_dir + "train/image/",
        help="the address of the train image",
    )
    parser.add_argument(
        "--Te_Image_dir",
        default=data_dir + "open/image/",
        help="the address of the test image",
    )
    parser.add_argument(
        "--Tr_Label_dir",
        default=data_dir + "train/label/",
        help="the address of the train label",
    )
    parser.add_argument(
        "--Te_Label_dir",
        default=data_dir + "open/label/",
        help="the address of the test label",
    )
    parser.add_argument(
        "--Ts_Image_dir",
        default=data_dir + "test/image/",
        help="the address of the test image",
    )
    parser.add_argument(
        "--Ts_Label_dir",
        default=data_dir + "test/label/",
        help="the address of the train label",
    )
    parser.add_argument(
        "--Tr_Meanstd_name",
        default="KIPA_Tr_Meanstd.npy",
        help="Train image Mean and std for normalization",#训练图像平均值和标准差
    )
    parser.add_argument(
        "--Te_Meanstd_name",
        default="KIPA_Te_Meanstd.npy",
        help="Test image Mean and std for normalization",
    )
    parser.add_argument(
        "--Ts_Meanstd_name",
        default="KIPA_Ts_Meanstd.npy",
        help="Test image Mean and std for normalization",
    )

    # files that are needed to be used to store contents 定义了三个用于存储内容、日志、和保存权重的路径参数
    parser.add_argument(
        # "--Dir_Txt", default=root_dir + "Txt/Txt_upper1/", help="Txt path"
        "--Dir_Txt", default=root_dir + "Txt/Txt_lower/", help="Txt path"
    )
    parser.add_argument(#"--Dir_Log", default=root_dir + "Log/upper_wt/", help="Log path"
        "--Dir_Log", default=root_dir + "Log/upper_lower/", help="Log path"
    )
    parser.add_argument(
        # "--Dir_Save", default=root_dir + "Results/upper_wt/", help="Save path"
        "--Dir_Save", default=root_dir + "Results/upper_lower/", help="Save path"
    )#预测结果
    parser.add_argument(
        # "--Dir_Weights", default=root_dir + "Weights/upper_wt/", help="Weights path"
        "--Dir_Weights", default=root_dir + "Weights/upper_lower/", help="Weights path"
    )#模型权重

    # Folders, dataset, etc.
    parser.add_argument(
        "--Image_Tr_txt",
        default=root_dir + "Txt/Txt_upper1/Image_Tr.txt",
        help="train image txt path",
    )
    parser.add_argument(
        "--Image_Te_txt",
        default=root_dir + "Txt/Txt_upper1/Image_Te.txt",
        help="test image txt path",
    )
    parser.add_argument(
        "--Label_Tr_txt",
        default=root_dir + "Txt/Txt_upper1/Label_Tr.txt",
        help="train label txt path",
    )
    parser.add_argument(
        "--Label_Te_txt",
        default=root_dir + "Txt/Txt_upper1/Label_Te.txt",
        help="test label txt path",
    )
    parser.add_argument(
        "--Image_Ts_txt",
        default=root_dir + "Txt/Txt_upper1/Image_Ts.txt",
        help="test image txt path",
    )
    parser.add_argument(
        "--Label_Ts_txt",
        default=root_dir + "Txt/Txt_upper1/Label_Ts.txt",
        help="train label txt path",
    )

    # Detailed path for saving results
    """
    Breif description:
        由于细管状结构所占比例较小，
         模型的结果可能会带来巨大的波动。
         为了减少不确定因素对模型分析的影响，
         我们将 <best> 结果保存在 <max> 文件夹中的验证数据集上，
         并对所有比较方法采用相同的标准，以确保公平！
    """
    parser.add_argument(
        # "--save_path", default=root_dir + "Results/upper_wt/DSCNet/", help="Save dir"
        "--save_path", default=root_dir + "Results/upper_lower/DSCNet/", help="Save dir"
    )
    parser.add_argument(
        # "--save_path_max",
        # default=root_dir + "Results/upper_wt/DSCNet_max/",
        # help="Save max dir",
        "--save_path_max",
        default=root_dir + "Results/upper_lower/DSCNet_max/",
        help="Save max dir",
    )
    parser.add_argument("--model_name", default="DSCNet_KIPA", help="Weights name")
    parser.add_argument(
        "--model_name_max", default="DSCNet_KIPA_max", help="Max Weights name"
    )
    parser.add_argument("--model_name1", default="DSCNet_KIPA1", help="Weights name")
    parser.add_argument(
        "--model_name_max1", default="DSCNet_KIPA_max1", help="Max Weights name"
    )
    parser.add_argument("--log_name", default="DSCNet_KIPA.log", help="Log name")

    # Network options
    parser.add_argument("--name", default='dyn_cbam_wt1',  help="model_name")
    parser.add_argument("--att", default=[32, 64, 128, 256, 320,320], help="attention")
    parser.add_argument("--n_channels", default=1, type=int, help="input channels")
    parser.add_argument("--n_classes", default=2, type=int, help="output channels")#代表几个分类，原本是3代表三分类,label 是2通道，采用的是onehot编码
    parser.add_argument(
        "--kernel_size", default=3, type=int, help="kernel size"
    )  # 9 refers to 1*9/9*1 for DSConv
    parser.add_argument(
        "--extend_scope", default=1.0, type=float, help="extend scope"
    )  # This parameter is not used
    parser.add_argument(
        "--if_offset", default=True, type=bool, help="if offset"
    )  # Whether to use the deformation or not
    parser.add_argument(
        "--n_basic_layer", default=16, type=int, help="basic layer numbers"
    )#用于指定网络中基本层的通道数量。
    parser.add_argument("--dim", default=8, type=int, help="dim numbers")#用于指定某种维度（可能是特征图的维度或其他）的数量

    # Training options
    parser.add_argument("--GPU_id", default="0", help="GPU ID")                             # todo

    #dataset
    parser.add_argument("--pin_memory", action="store_true", default=False)
    """
    Reference: --ROI_shape: (128, 96, 96)  3090's memory occupancy is about 16653 MiB
    """
    parser.add_argument("--ROI_shape", default=(96, 96, 96), type=int, help="roi size")#高度深度宽度default=(128, 96, 96)(128, 160, 160)
    parser.add_argument("--samples_per_volume", default=2, type=int, help="samples_per_volume")  # 高度深度宽度default=(128, 96, 96)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size")#1
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument("--lr", default=0.001, type=int, help="learning rate")#1e-4
    parser.add_argument(
        "--start_train_epoch", default=1, type=int, help="Start training epoch"
    )
    parser.add_argument(
        "--start_verify_epoch", default=200, type=int, help="Start verifying epoch"
    )
    parser.add_argument("--n_epochs", default=300, type=int, help="Epoch Num")
    parser.add_argument("--if_retrain", default=True, type=bool, help="If Retrain")#True
    parser.add_argument("--if_onlytest", default=False, type=bool, help="If Only Test")#False

    args, unknown = parser.parse_known_args()
    Process(args)