
import torch
from torch import nn
import torch.nn.functional as F
from segment_anything import sam_model_registry3D

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        # elif isinstance(module, nn.InstanceNorm3d):
        #     nn.init.constant_(module.weight, 1)
        #     nn.init.constant_(module.bias, 0)

class Res_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            )
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv(x)
        x += res
        x = self.relu(x)
        return x

class conv_block3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size= 3, stride = 1,padding = 1 ):
        super(conv_block3D, self).__init__()
        self.conv_block3D = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            #affine是否需要仿射变化
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_block3D(x)

class MultiScaleFeatureFusionConvBlock3d(nn.Module):
    '''
    多尺度特征融合模块+卷积模块
    '''

    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureFusionConvBlock3d, self).__init__()
        self.out_channels_split = out_channels // 4
        #Res block
        self.conv1_in = Res_conv_block(in_channels,out_channels)
        #MSFF block
        self.conv3_addition_2 = conv_block3D(self.out_channels_split, self.out_channels_split)
        self.conv3_addition_3 = conv_block3D(self.out_channels_split, self.out_channels_split)
        self.conv3_addition_4 = conv_block3D(self.out_channels_split, self.out_channels_split)
        self.conv1_out = nn.Conv3d(out_channels, out_channels, [1, 1, 1], stride=1, padding=0)
        self.conv3 = conv_block3D(out_channels, out_channels)

    def forward(self, x):
        # Res block
        f = self.conv1_in(x)
        # MSFF block
        f_1 = f[:, 0: self.out_channels_split, :, :, :]
        f_2 = self.conv3_addition_2(f[:, self.out_channels_split: 2 * self.out_channels_split, :, :, :])
        f_3 = self.conv3_addition_3(f[:, 2 * self.out_channels_split: 3 * self.out_channels_split, :, :, :] + f_2)
        f_4 = self.conv3_addition_4(f[:, 3 * self.out_channels_split: 4 * self.out_channels_split, :, :, :] + f_3)
        fusion = f_1 + f_2 + f_3 + f_4
        ####按照论文这里有个卷积Conv1（fusion）
        f_1 = f_1 + fusion
        f_2 = f_2 + fusion
        f_3 = f_3 + fusion
        f_4 = f_4 + fusion
        return self.conv3(self.conv1_out(torch.cat((f_1, f_2, f_3, f_4), dim=1)))

class SpatialAttentionBlock(nn.Module):
    '''
    空间注意模块(Spatial Attention Block, SA)
    1×1×1卷积 + BN + ReLU → 1×1×1卷积 → sigmoid
    【ATTENTION】这里用的是instance norm而不是batch normal，不知道有没有影响
    也不知道outchannels应该是多少，反正输出通道数量一定是1
    '''
    def __init__(self, in_channels, out_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, [1, 1, 1], stride=1, padding=0)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.nonlin = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.conv2 = nn.Conv3d(in_channels, out_channels, [1, 1, 1], stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv2(self.nonlin(self.norm(self.conv1(x)))))

class AttentionGuidedConcatenationBlock(nn.Module):
    '''
    注意力引导连接模块(Attention-Guided Concatenation, AGC)
    高水平特征f_h反卷积→f'_h，
    → f'_h通过空间注意模块SA() → 0-1特征图A
    → A×f_l → 优化的低水平特征f'_l
    理论上输出通道数量与f_l一致
    '''

    def __init__(self, f_h_channels, f_l_channels):
        super(AttentionGuidedConcatenationBlock, self).__init__()

        # self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.deconv = nn.ConvTranspose3d(f_h_channels, f_l_channels, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        self.SA = SpatialAttentionBlock(f_l_channels, f_l_channels)

    def forward(self, f_h, f_l):
        '''理论上输出通道数量与f_l一致'''
        f_h1 = self.deconv(f_h)
        A = self.SA(self.deconv(f_h))
        return self.SA(self.deconv(f_h)) * f_l

class UNetAgcMsff(nn.Module):
    '''架构和UNetAGC一样，唯一的区别是将StackedConvBlocks3d替换为MSFF模块'''

    def __init__(self, in_channels, out_channels, base_filters_num=16,ds=False):
        super(UNetAgcMsff, self).__init__()
        self.weightInitializer = InitWeights_He(1e-2)
        self.ds = ds
        if base_filters_num == 16:
            features = [in_channels ,16, 32, 64, 128, 256, 256]
        elif base_filters_num == 32:
            features = [in_channels, 32, 64, 128, 256, 512, 512]

        self.down_0 = MultiScaleFeatureFusionConvBlock3d(in_channels, features[1])
        self.pool_0 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_1 = MultiScaleFeatureFusionConvBlock3d(features[1], features[2])
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_2 = MultiScaleFeatureFusionConvBlock3d(features[2], features[3])
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_3 = MultiScaleFeatureFusionConvBlock3d(features[3], features[4])
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_4 = MultiScaleFeatureFusionConvBlock3d(features[4], features[5])
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.bottleneck = MultiScaleFeatureFusionConvBlock3d(features[5], features[6])

        self.AGC_4 = AttentionGuidedConcatenationBlock(features[6], features[5])
        self.trans_4 = nn.ConvTranspose3d(features[6], features[5], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_4 = MultiScaleFeatureFusionConvBlock3d(features[5] * 2 + 384, features[5])
        self.AGC_3 = AttentionGuidedConcatenationBlock(features[5] +384, features[4])
        self.trans_3 = nn.ConvTranspose3d(features[5], features[4], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_3 = MultiScaleFeatureFusionConvBlock3d(features[4] * 2 , features[4])
        self.AGC_2 = AttentionGuidedConcatenationBlock(features[4], features[3])
        self.trans_2 = nn.ConvTranspose3d(features[4], features[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_2 = MultiScaleFeatureFusionConvBlock3d(features[3] * 2, features[3])
        self.AGC_1 = AttentionGuidedConcatenationBlock(features[3], features[2])
        self.trans_1 = nn.ConvTranspose3d(features[3], features[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_1 = MultiScaleFeatureFusionConvBlock3d(features[2] * 2, features[2])
        self.AGC_0 = AttentionGuidedConcatenationBlock(features[2], features[1])
        self.trans_0 = nn.ConvTranspose3d(features[2], features[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_0 = MultiScaleFeatureFusionConvBlock3d(features[1] * 2, features[1])

        self.device = 'cuda'
        model_weight_path = './../sam_med3d_turbo.pth'
        model_type = "vit_b_ori"
        mobile_sam = sam_model_registry3D[model_type](checkpoint=None)
        model_dict = torch.load(model_weight_path, map_location=self.device)
        state_dict = model_dict['model_state_dict']
        mobile_sam.load_state_dict(state_dict)
        self.sam_image_encoder = mobile_sam.image_encoder
        for param in self.sam_image_encoder.parameters():
            param.requires_grad = False


        self.output = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
        self.apply(self.weightInitializer)

    def forward(self, x):
        down_0 = self.down_0(x)
        pool_0 = self.pool_0(down_0)

        down_1 = self.down_1(pool_0)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bottleneck = self.bottleneck(pool_4)
        trans_4_1 = self.trans_4(bottleneck)

        sam_input = x
        sam_input = F.interpolate(sam_input, size=(128, 128, 128), mode='trilinear', align_corners=True)
        sam_embed = self.sam_image_encoder(sam_input)
        sam_embed = F.interpolate(sam_embed, size=(trans_4_1.shape[2], trans_4_1.shape[3], trans_4_1.shape[4]), mode='trilinear',align_corners=True)

        trans_4_1_2 = torch.cat((trans_4_1, sam_embed), dim=1)
        up_4 = self.up_4(torch.cat((trans_4_1_2, self.AGC_4(bottleneck, down_4)), dim=1))

        trans_3 = self.trans_3(up_4)
        up_3 = self.up_3(torch.cat((trans_3, self.AGC_3(trans_4_1_2, down_3)), dim=1))

        trans_2 = self.trans_2(up_3)
        up_2 = self.up_2(torch.cat((trans_2, self.AGC_2(trans_3, down_2)), dim=1))

        trans_1 = self.trans_1(up_2)
        up_1 = self.up_1(torch.cat((trans_1, self.AGC_1(trans_2, down_1)), dim=1))

        trans_0 = self.trans_0(up_1)
        up_0 = self.up_0(torch.cat((trans_0, self.AGC_0(trans_1, down_0)), dim=1))
        output = self.output(up_0)


        return output

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn([2, 1, 160, 160, 96]).to(device)
    # unet = UNet(in_channels=1, out_channels=3).to(device)
    unet = UNetAgcMsff(in_channels=1, out_channels=3,base_filters_num=16).to(device)
    print('#parameters:', sum(param.numel() for param in unet.parameters()))
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     inet1 = DataParallel(unet)
    # for name, param in unet.named_parameters():
    #     print(name, param)
    out1 = unet(tensor)
    print(out1.shape)