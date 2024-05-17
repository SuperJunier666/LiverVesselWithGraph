import torch
from torch import nn
import torch.nn.functional as F


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

class conv_block3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size= 3, stride = 1,padding = 1 ):
        super(conv_block3D, self).__init__()
        self.conv_block3D = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv_block3D(x)

class Res_conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Res_conv_block, self).__init__()
        self.double_conv3d = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_ch)
        )
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1)
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_new = self.double_conv3d(x)
        x_shortcut = self.shortcut(x)
        x = self.out_relu(x_new + x_shortcut)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.MaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#多尺度模块
class MultiScaleFeatureFusionConvBlock3d(nn.Module):

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
        # Channel Attention Module
        self.channel_attention = ChannelAttention(out_channels)
        # Spatial Attention Module
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Res block
        f = self.conv1_in(x)
        # Channel Attention
        f = self.channel_attention(f) * f
        # MSFF block
        f_1 = f[:, 0: self.out_channels_split, :, :, :]
        f_2 = self.conv3_addition_2(f[:, self.out_channels_split: 2 * self.out_channels_split, :, :, :])
        f_3 = self.conv3_addition_3(f[:, 2 * self.out_channels_split: 3 * self.out_channels_split, :, :, :] + f_2)
        f_4 = self.conv3_addition_4(f[:, 3 * self.out_channels_split: 4 * self.out_channels_split, :, :, :] + f_3)
        fusion = f_1 + f_2 + f_3 + f_4
        # Spatial Attention
        fusion = self.spatial_attention(fusion) * fusion
        f_1 = f_1 + fusion
        f_2 = f_2 + fusion
        f_3 = f_3 + fusion
        f_4 = f_4 + fusion

        fm = torch.cat((f_1, f_2, f_3, f_4), dim=1)
        return self.conv1_out(fm)

class MultiScalejJumpConnectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScalejJumpConnectionBlock, self).__init__()
        self.fusion4 = MultiScaleFeatureFusionConvBlock3d(in_channels, out_channels)
        self.fusion3 = MultiScaleFeatureFusionConvBlock3d(in_channels, out_channels//2)
        self.fusion2 = MultiScaleFeatureFusionConvBlock3d((in_channels//4) * 3, out_channels // 4)
        self.fusion1 = MultiScaleFeatureFusionConvBlock3d(in_channels // 2, out_channels // 8)

        self.Upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv4 = nn.Conv3d(in_channels, out_channels//2, kernel_size=1, stride=1, padding=0)

        self.Upsample3_1 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.conv3_1 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.Upsample3_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv3_2 = nn.Conv3d(in_channels//2, out_channels // 4, kernel_size=1, stride=1, padding=0)

        self.Upsample2_1 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.conv2_1 = nn.Conv3d(in_channels, out_channels // 8, kernel_size=1, stride=1, padding=0)
        self.Upsample2_2 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.conv2_2 = nn.Conv3d(in_channels // 2, out_channels // 8, kernel_size=1, stride=1, padding=0)
        self.Upsample2_3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv2_3 = nn.Conv3d(in_channels // 4, out_channels // 8, kernel_size=1, stride=1, padding=0)

    def forward(self,input4,input3,input2,input1):
        ####cat4
        output4 = self.fusion4(input4)
        ####cat3
        input4_3 = self.conv4(self.Upsample4(input4))
        input3_3 = torch.cat([input4_3,input3],dim=1)
        output3 = self.fusion3(input3_3)
        ####cat2
        input4_2 = self.conv3_1(self.Upsample3_1(input4))
        input3_2 = self.conv3_2(self.Upsample3_2(input3))
        input2_2 = torch.cat([input4_2, input3_2,input2], dim=1)
        output2 = self.fusion2(input2_2)
        ####cat1
        input4_1 = self.conv2_1(self.Upsample2_1(input4))
        input3_1 = self.conv2_2(self.Upsample2_2(input3))
        input2_1 = self.conv2_3(self.Upsample2_3(input2))
        input1_1 = torch.cat([input4_1, input3_1, input2_1,input1], dim=1)
        output1 = self.fusion1(input1_1)

        return output4,output3,output2,output1


class EdgeAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAttentionModule, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=1),
                                           nn.InstanceNorm3d(in_channels))
        self.conv_attention = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.bn2 = nn.InstanceNorm3d(1)  # Add batch normalization for the attention map

    def forward(self, fi):
        fi = self.conv(fi)
        erosion,indices = F.max_pool3d(fi, kernel_size=3, stride=1, padding=1,return_indices=True)
        dilation = F.max_unpool3d(fi, indices=indices, kernel_size=3, stride=1, padding=1)
        attention =  torch.sigmoid(self.conv_attention(torch.abs(dilation - erosion)))
        enhanced_features = attention * fi
        f_out = fi + enhanced_features
        return f_out

class MSENet(nn.Module):

    def __init__(self, in_channels, out_channels, base_filters_num=16):
        super(MSENet, self).__init__()
        self.weightInitializer = InitWeights_He(1e-2)
        if base_filters_num == 16:
            features = [in_channels, 16, 32, 64, 128, 256, 256]
        elif base_filters_num == 32:
            features = [in_channels, 16, 32, 64, 128, 256, 512]
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

        self.trans_4 = nn.ConvTranspose3d(features[6], features[5], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_4 = MultiScaleFeatureFusionConvBlock3d(features[5] * 2, features[5])
        # self.ea4 = EdgeAttentionModule(features[5])
        self.trans_3 = nn.ConvTranspose3d(features[5], features[4], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_3 = MultiScaleFeatureFusionConvBlock3d(features[4] * 2, features[4])
        self.ea3 = EdgeAttentionModule(features[4])
        self.trans_2 = nn.ConvTranspose3d(features[4], features[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_2 = MultiScaleFeatureFusionConvBlock3d(features[3] * 2, features[3])
        self.ea2 = EdgeAttentionModule(features[3])
        self.trans_1 = nn.ConvTranspose3d(features[3], features[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_1 = MultiScaleFeatureFusionConvBlock3d(features[2] * 2, features[2])
        self.ea1 = EdgeAttentionModule(features[2])
        self.trans_0 = nn.ConvTranspose3d(features[2], features[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_0 = MultiScaleFeatureFusionConvBlock3d(features[1] * 2, features[1])


        self.seg_output_0 = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
        self.cline_output = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
        self.output3 = nn.Conv3d(features[4], out_channels, kernel_size=1, bias=False)
        self.output2 = nn.Conv3d(features[3], out_channels, kernel_size=1, bias=False)
        self.output1 = nn.Conv3d(features[2], out_channels, kernel_size=1, bias=False)
        self.conv_feature = nn.Conv3d(features[1], 8, kernel_size=1, bias=False)
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

        trans_4 = self.trans_4(bottleneck)
        
        up_4 = self.up_4(torch.cat((trans_4, down_4), dim=1))
        # up_4 = self.ea4(up_4)
        trans_3 = self.trans_3(up_4)
        up_3 = self.up_3(torch.cat((trans_3, down_3), dim=1))
        up_3 = self.ea3(up_3)
        ds_3 = self.output3(up_3)
        trans_2 = self.trans_2(up_3)
        up_2 = self.up_2(torch.cat((trans_2, down_2), dim=1))
        up_2 = self.ea2(up_2)
        ds_2 = self.output2(up_2)
        trans_1 = self.trans_1(up_2)
        up_1 = self.up_1(torch.cat((trans_1, down_1), dim=1))
        up_1 = self.ea1(up_1)
        ds_1 = self.output1(up_1)
        trans_0 = self.trans_0(up_1)
        up_0 = self.up_0(torch.cat((trans_0, down_0), dim=1))

        feature_map = self.conv_feature(up_0)
        seg_output = self.seg_output_0(up_0)
        cline_output = self.cline_output(up_0)

        return seg_output,cline_output,feature_map,ds_3,ds_2,ds_1


class Baseline(nn.Module):

    def __init__(self, in_channels, out_channels, base_filters_num=16):
        super(Baseline, self).__init__()
        self.weightInitializer = InitWeights_He(1e-2)
        if base_filters_num == 16:
            features = [in_channels, 16, 32, 64, 128, 256, 256]
        elif base_filters_num == 32:
            features = [in_channels, 16, 32, 64, 128, 256, 512]
        self.down_0 = Res_conv_block(in_channels, features[1])
        self.pool_0 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_1 = Res_conv_block(features[1], features[2])
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_2 = Res_conv_block(features[2], features[3])
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_3 = Res_conv_block(features[3], features[4])
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_4 = Res_conv_block(features[4], features[5])
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.bottleneck = Res_conv_block(features[5], features[6])

        self.trans_4 = nn.ConvTranspose3d(features[6], features[5], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_4 = Res_conv_block(features[5] * 2, features[5])
        self.trans_3 = nn.ConvTranspose3d(features[5], features[4], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_3 = Res_conv_block(features[4] * 2, features[4])
        self.trans_2 = nn.ConvTranspose3d(features[4], features[3], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_2 = Res_conv_block(features[3] * 2, features[3])

        self.trans_1 = nn.ConvTranspose3d(features[3], features[2], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_1 = Res_conv_block(features[2] * 2, features[2])
        self.trans_0 = nn.ConvTranspose3d(features[2], features[1], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_0 = Res_conv_block(features[1] * 2, features[1])

        self.seg_output_0 = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
        self.cline_output = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
        self.output3 = nn.Conv3d(features[4], out_channels, kernel_size=1, bias=False)
        self.output2 = nn.Conv3d(features[3], out_channels, kernel_size=1, bias=False)
        self.output1 = nn.Conv3d(features[2], out_channels, kernel_size=1, bias=False)
        self.conv_feature = nn.Conv3d(features[1], 8, kernel_size=1, bias=False)
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

        trans_4 = self.trans_4(bottleneck)

        up_4 = self.up_4(torch.cat((trans_4, down_4), dim=1))
        # up_4 = self.ea4(up_4)
        trans_3 = self.trans_3(up_4)
        up_3 = self.up_3(torch.cat((trans_3, down_3), dim=1))
        ds_3 = self.output3(up_3)
        trans_2 = self.trans_2(up_3)
        up_2 = self.up_2(torch.cat((trans_2, down_2), dim=1))
        ds_2 = self.output2(up_2)
        trans_1 = self.trans_1(up_2)
        up_1 = self.up_1(torch.cat((trans_1, down_1), dim=1))
        ds_1 = self.output1(up_1)
        trans_0 = self.trans_0(up_1)
        up_0 = self.up_0(torch.cat((trans_0, down_0), dim=1))

        feature_map = self.conv_feature(up_0)
        seg_output = self.seg_output_0(up_0)
        cline_output = self.cline_output(up_0)

        return seg_output, cline_output, feature_map, ds_3, ds_2, ds_1

class Baseline_MS(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters_num=16):
        super(Baseline_MS, self).__init__()
        self.weightInitializer = InitWeights_He(1e-2)
        if base_filters_num == 16:
            features = [in_channels, 16, 32, 64, 128, 256, 256]
        elif base_filters_num == 32:
            features = [in_channels,32, 64, 128, 256, 512 , 512]
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

        self.trans_4 = nn.ConvTranspose3d(features[6], features[5], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_4 = MultiScaleFeatureFusionConvBlock3d(features[5] * 2, features[5])
        self.trans_3 = nn.ConvTranspose3d(features[5], features[4], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_3 = MultiScaleFeatureFusionConvBlock3d(features[4] * 2, features[4])
        self.trans_2 = nn.ConvTranspose3d(features[4], features[3], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_2 = MultiScaleFeatureFusionConvBlock3d(features[3] * 2, features[3])

        self.trans_1 = nn.ConvTranspose3d(features[3], features[2], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_1 = MultiScaleFeatureFusionConvBlock3d(features[2] * 2, features[2])
        self.trans_0 = nn.ConvTranspose3d(features[2], features[1], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_0 = MultiScaleFeatureFusionConvBlock3d(features[1] * 2, features[1])

        self.seg_output_0 = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
        self.cline_output = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
        self.output3 = nn.Conv3d(features[4], out_channels, kernel_size=1, bias=False)
        self.output2 = nn.Conv3d(features[3], out_channels, kernel_size=1, bias=False)
        self.output1 = nn.Conv3d(features[2], out_channels, kernel_size=1, bias=False)
        self.conv_feature = nn.Conv3d(features[1], 8, kernel_size=1, bias=False)
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

        trans_4 = self.trans_4(bottleneck)

        up_4 = self.up_4(torch.cat((trans_4, down_4), dim=1))
        # up_4 = self.ea4(up_4)
        trans_3 = self.trans_3(up_4)
        up_3 = self.up_3(torch.cat((trans_3, down_3), dim=1))
        ds_3 = self.output3(up_3)
        trans_2 = self.trans_2(up_3)
        up_2 = self.up_2(torch.cat((trans_2, down_2), dim=1))
        ds_2 = self.output2(up_2)
        trans_1 = self.trans_1(up_2)
        up_1 = self.up_1(torch.cat((trans_1, down_1), dim=1))
        ds_1 = self.output1(up_1)
        trans_0 = self.trans_0(up_1)
        up_0 = self.up_0(torch.cat((trans_0, down_0), dim=1))

        feature_map = self.conv_feature(up_0)
        seg_output = self.seg_output_0(up_0)
        cline_output = self.cline_output(up_0)

        return seg_output, cline_output, feature_map, ds_3, ds_2, ds_1

class Baseline_MSJ(nn.Module):

    def __init__(self, in_channels, out_channels, base_filters_num=16):
        super(Baseline_MSJ, self).__init__()
        self.weightInitializer = InitWeights_He(1e-2)
        if base_filters_num == 16:
            features = [in_channels, 16, 32, 64, 128, 256, 256]
        elif base_filters_num == 32:
            features = [in_channels, 32, 64, 128, 256, 256, 512]
        self.down_0 = Res_conv_block(in_channels, features[1])
        self.pool_0 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_1 = Res_conv_block(features[1], features[2])
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_2 = Res_conv_block(features[2], features[3])
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_3 = Res_conv_block(features[3], features[4])
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # 多尺度跳链接
        self.MSJC = MultiScalejJumpConnectionBlock(features[4], features[4])

        self.bottleneck = Res_conv_block(features[4], features[5])

        self.trans_3 = nn.ConvTranspose3d(features[5], features[4], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_3 = Res_conv_block(features[4] * 2, features[4])
        self.trans_2 = nn.ConvTranspose3d(features[4], features[3], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_2 = Res_conv_block(features[3] * 2, features[3])

        self.trans_1 = nn.ConvTranspose3d(features[3], features[2], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_1 = Res_conv_block(features[2] * 2, features[2])
        self.trans_0 = nn.ConvTranspose3d(features[2], features[1], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_0 = Res_conv_block(features[1] * 2, features[1])

        self.seg_output_0 = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
        self.cline_output = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
        self.output3 = nn.Conv3d(features[4], out_channels, kernel_size=1, bias=False)
        self.output2 = nn.Conv3d(features[3], out_channels, kernel_size=1, bias=False)
        self.output1 = nn.Conv3d(features[2], out_channels, kernel_size=1, bias=False)
        self.conv_feature = nn.Conv3d(features[1], 8, kernel_size=1, bias=False)
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

        bottleneck = self.bottleneck(pool_3)
        down_3, down_2, down_1, down_0 = self.MSJC(down_3, down_2, down_1, down_0)

        trans_3 = self.trans_3(bottleneck)
        up_3 = self.up_3(torch.cat((trans_3, down_3), dim=1))
        ds_3 = self.output3(up_3)

        trans_2 = self.trans_2(up_3)
        up_2 = self.up_2(torch.cat((trans_2, down_2), dim=1))
        ds_2 = self.output2(up_2)

        trans_1 = self.trans_1(up_2)
        up_1 = self.up_1(torch.cat((trans_1, down_1), dim=1))
        ds_1 = self.output1(up_1)

        trans_0 = self.trans_0(up_1)
        up_0 = self.up_0(torch.cat((trans_0, down_0), dim=1))

        feature_map = self.conv_feature(up_0)
        seg_output = self.seg_output_0(up_0)
        cline_output = self.cline_output(up_0)

        return seg_output, cline_output, feature_map, ds_3, ds_2, ds_1


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn([2, 1, 160, 160, 96]).to(device)
    # unet = UNet(in_channels=1, out_channels=3).to(device)
    net = MSENet(in_channels=1,out_channels=2,base_filters_num=16).to(device)
    print('#parameters:', sum(param.numel() for param in net.parameters()))
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
        # inet1 = DataParallel(unet)
    # for name, param in unet.named_parameters():
    #     print(name, param)
    out,out2,feature,_,_,_ = net(tensor)
    # out = net(tensor)
    print(out.shape)



