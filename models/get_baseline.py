from monai.networks.nets import  SegResNet, VNet, SwinUNETR,UNet,ResNet,UNETR,DynUNet
import torch
import torch.nn as nn
from reg import Out, ConvBNReLU
# from reg import Out, ConvBNReLU


def get_model(name, att = None, in_channels=4, out_channels=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print( device)
    if name == "dynunet":
        model = DynUNet(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    #filters = [32, 64, 128, 256, 320, 320],
                    filters=[64, 96, 128, 192, 256, 384],
                    #filters =[64, 96, 128, 192, 256, 384, 512, 768, 1024][:len(["strides"])],
                    kernel_size= [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                    strides= [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                    upsample_kernel_size=[ [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                    res_block=True,
                    norm_name="instance",
                    deep_supervision=False,
                    deep_supr_num=1,).to(device)
        return model
    elif name == "vnet1":
        model = VNet(
            ).to(device)
        return model
    elif name == "unet":  # 效果
        model = UNet(
            spatial_dims=3,  # 选择3D U-Net
            in_channels=1,  # 输入通道数（例如，单通道CT或MRI数据）
            out_channels=2,  # 输出通道数（例如，二分类任务的类别数）
            #channels=(4, 8, 16, 32, 64),  # 每个级别的卷积核数量
            channels=(16, 32, 64, 128, 256),  # 每个级别的卷积核数量
            strides=(2, 2, 2, 2),  # 每个级别的步幅
            num_res_units=2,  # 残差单元数量
            norm='INSTANCE'  # 使用实例归一化（可选）
        ).to(device)
        return model
    elif name == "unetr":  # 效果
        model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(96,96,96),
            feature_size=32,
            norm_name='batch').to(device)
        return model
    elif name == "resnet":  # 构造ResNet模型
        model = ResNet(
            block="basic",  # 或者 ResNetBottleneck，根据需要选择
            layers=[2, 2, 2, 2],  # 每个block的层数（例如ResNet-18: 2, 2, 2, 2）
            block_inplanes=[64, 128, 256, 512],  # 每个block的输入通道数
            spatial_dims=3,  # 选择3D空间
            n_input_channels=1,  # 输入的通道数（例如RGB图像）
            conv1_t_size=7,  # 卷积核大小，7x7
            conv1_t_stride=2,  # 步幅为2
            no_max_pool=False,  # 是否禁用最大池化
            shortcut_type="B",  # 选择B类型的shortcut
            widen_factor=1.0,  # 宽度因子
            num_classes=2,  # 输出类别数（例如ImageNet的1000类）
            feed_forward=True,  # 是否使用前馈层
            bias_downsample=True  # 是否使用偏置进行下采样
        ).to(device)
        return model
    elif name == "segresnet":#效果不好
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=0.2,).to(device)
        return model
        
    elif name == "swinunet":
        model = SwinUNETR(
            img_size=(128,128,128),
            in_channels=in_channels,
            out_channels=out_channels,
            drop_rate=0.3,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,).to(device)
        return model

    elif name == "vnet":
        model = VNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=0,
            dropout_prob_up=(0, 0),
            ).to(device)
        return model
    else:
        model = None


def get_down(name, trained, cate=False):
    ## Init model
    model = get_model(name)
    ## Load n Frozen model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(trained ,map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    for param in model.parameters():
        param.requires_grad = False
    
    net = Out(320, cate).to(device)
    ## Get encoder part:
    if name == 'dynunet':
        down = torch.nn.Sequential(model.input_block, *model.downsamples[:],model.bottleneck)
        net = torch.nn.Sequential(ConvBNReLU(320,320,3,1),ConvBNReLU(320,320,3,1), net).to(device)
        return net, down
    elif name == 'segresnet':
        down = torch.nn.Sequential(model.act_mod, model.convInit, *model.down_layers)
        net = torch.nn.Sequential(ConvBNReLU(128,256,3,2),ConvBNReLU(256,320,3,2), net).to(device)
        return net, down
        # torch.nn.Conv3d(512,320,1,1)

if __name__ == "__main__":
    ## params
    name = "segresnet"
    trained = "temp/trained_segresnet.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## getmodel
    x = torch.zeros((2,4,128,128,128)).to(device)
    reg, encode = get_down(name, trained, cate=True)
    model = nn.Sequential()
    model.add_module("encode", encode)
    model.add_module("reg", reg)
    y = model(x)
    print(y)
