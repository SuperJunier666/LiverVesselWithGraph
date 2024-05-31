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

class conv_block3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size= 3, stride = 1,padding = 1 ):
        super(conv_block3D, self).__init__()
        self.conv_block3D = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        return self.conv_block3D(x)

class Res_conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Res_conv_block, self).__init__()
        self.double_conv3d = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_ch)
        )
        self.shortcut = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1)
        self.out_relu = nn.LeakyReLU(inplace=True)

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
                                nn.LeakyReLU(),
                                nn.Conv3d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = self.fc(self.avg_pool(x))
        # max_out = self.fc(self.max_pool(x))
        # assert torch.isfinite(x).all(), "输入包含 NaN 或无穷大值"
        avg_out = self.fc(self.avg_pool(x))
        # assert torch.isfinite(avg_out).all(), "avg_out 包含 NaN 或无穷大值"
        max_out = self.fc(self.max_pool(x))
        # assert torch.isfinite(max_out).all(), "max_out 包含 NaN 或无穷大值"
        output = self.sigmoid(avg_out + max_out)
        # assert torch.isfinite(output).all(), "output 包含 NaN 或无穷大值"
        # return self.sigmoid(avg_out + max_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # assert torch.isfinite(avg_out).all(), "avg_out包含 NaN 或无穷大值"
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # assert torch.isfinite(max_out).all(), "max_out包含 NaN 或无穷大值"
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # assert torch.isfinite(x).all(), "x包含 NaN 或无穷大值"
        output = self.sigmoid(x)
        # assert torch.isfinite(output).all(), "output包含 NaN 或无穷大值"
        return output

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
        self.conv1_out = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # Channel Attention Module
        self.channel_attention = ChannelAttention(out_channels)
        # Spatial Attention Module
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Res block
        # assert torch.isfinite(x).all(), "x包含 NaN 或无穷大值"
        f = self.conv1_in(x)
        # assert torch.isfinite(f).all(), "f包含 NaN 或无穷大值"
        # Channel Attention
        f = self.channel_attention(f) * f
        # assert torch.isfinite(f).all(), "fca包含 NaN 或无穷大值"
        # MSFF block
        f_1 = f[:, 0: self.out_channels_split, :, :, :]
        f_2 = self.conv3_addition_2(f[:, self.out_channels_split: 2 * self.out_channels_split, :, :, :])
        f_3 = self.conv3_addition_3(f[:, 2 * self.out_channels_split: 3 * self.out_channels_split, :, :, :] + f_2)
        f_4 = self.conv3_addition_4(f[:, 3 * self.out_channels_split: 4 * self.out_channels_split, :, :, :] + f_3)
        fusion = f_1 + f_2 + f_3 + f_4
        # assert torch.isfinite(fusion).all(), "fusion包含 NaN 或无穷大值"
        # Spatial Attention
        fusion = self.spatial_attention(fusion) * fusion
        # assert torch.isfinite(fusion).all(), "fsa包含 NaN 或无穷大值"
        f_1 = f_1 + fusion
        f_2 = f_2 + fusion
        f_3 = f_3 + fusion
        f_4 = f_4 + fusion

        fm = torch.cat((f_1, f_2, f_3, f_4), dim=1)
        # assert torch.isfinite(fusion).all(), "fm包含 NaN 或无穷大值"
        return self.conv1_out(fm)

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

class SubeNet(nn.Module):

    def __init__(self, in_channels, out_channels, base_filters_num):
        super(SubeNet, self).__init__()
        self.weightInitializer = InitWeights_He(1e-2)
        if base_filters_num == 16:
            features = [in_channels, 16, 32, 64, 128, 256, 320]
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

        self.trans_4 = nn.ConvTranspose3d(features[6], features[5], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_4 = MultiScaleFeatureFusionConvBlock3d(features[5] * 2, features[5])
        # self.ea4 = EdgeAttentionModule(features[5])
        self.trans_3 = nn.ConvTranspose3d(features[5]+384, features[4], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_3 = MultiScaleFeatureFusionConvBlock3d(features[4] * 2, features[4])
        self.ea3 = EdgeAttentionModule(features[4])
        self.trans_2 = nn.ConvTranspose3d(features[4], features[3], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_2 = MultiScaleFeatureFusionConvBlock3d(features[3] * 2, features[3])
        self.ea2 = EdgeAttentionModule(features[3])
        self.trans_1 = nn.ConvTranspose3d(features[3], features[2], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_1 = MultiScaleFeatureFusionConvBlock3d(features[2] * 2, features[2])
        self.ea1 = EdgeAttentionModule(features[2])
        self.trans_0 = nn.ConvTranspose3d(features[2], features[1], kernel_size=3, stride=2, padding=1,output_padding=1)
        self.up_0 = MultiScaleFeatureFusionConvBlock3d(features[1] * 2, features[1])

        self.seg_output_0 = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
        self.cline_output = nn.Conv3d(features[1], out_channels, kernel_size=1, bias=False)
        self.output3 = nn.Conv3d(features[4], out_channels, kernel_size=1, bias=False)
        self.output2 = nn.Conv3d(features[3], out_channels, kernel_size=1, bias=False)
        self.output1 = nn.Conv3d(features[2], out_channels, kernel_size=1, bias=False)
        self.conv_feature = nn.Conv3d(features[1], 8, kernel_size=1, bias=False)

        #sam
        self.device = 'cuda'
        model_weight_path = './sam_med3d_turbo.pth'
        model_type = "vit_b_ori"
        mobile_sam = sam_model_registry3D[model_type](checkpoint=None)
        model_dict = torch.load(model_weight_path, map_location=self.device)
        state_dict = model_dict['model_state_dict']
        mobile_sam.load_state_dict(state_dict)
        self.sam_image_encoder = mobile_sam.image_encoder
        for param in self.sam_image_encoder.parameters():
            param.requires_grad = False

        self.apply(self.weightInitializer)

    def forward(self, x):

        # print("输入的统计信息: min={}, max={}, mean={}, std={}".format(x.min().item(), x.max().item(), x.mean().item(),x.std().item()))
        # assert torch.isfinite(x).all(), "x包含 NaN 或无穷大值"
        down_0 = self.down_0(x)
        # print("输入的down_0统计信息: min={}, max={}, mean={}, std={}".format(down_0.min().item(), down_0.max().item(), down_0.mean().item(), down_0.std().item()))
        # assert torch.isfinite(down_0).all(), "down_0包含 NaN 或无穷大值"
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
        # assert torch.isfinite(bottleneck).all(), "bottleneck包含 NaN 或无穷大值"

        sam_input = x
        sam_input = F.interpolate(sam_input, size=(128, 128, 128), mode='trilinear', align_corners=True)
        sam_embed = self.sam_image_encoder(sam_input)
        # assert torch.isfinite(sam_embed).all(), "sam_embed1包含 NaN 或无穷大值"
        # print(sam_embed.shape)
        trans_4 = self.trans_4(bottleneck)

        up_4_1 = self.up_4(torch.cat((trans_4, down_4), dim=1))
        # assert torch.isfinite(up_4_1).all(), "up_4_1包含 NaN 或无穷大值"
        sam_embed = F.interpolate(sam_embed, size=(up_4_1.shape[2], up_4_1.shape[3], up_4_1.shape[4]),mode='trilinear', align_corners=True)
        # assert torch.isfinite(sam_embed).all(), "sam_embed2包含 NaN 或无穷大值"
        up_4_2 = torch.cat((up_4_1, sam_embed), dim=1)
        trans_3 = self.trans_3(up_4_2)
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

        # feature_map = self.conv_feature(up_0)
        seg_output = self.seg_output_0(up_0)
        cline_output = self.cline_output(up_0)
        return seg_output, cline_output, up_0, ds_3, ds_2, ds_1


# class SubeNet(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(SubeNet, self).__init__()
#
#     def forward(self, x):
#
#         return 0
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn([2, 1, 160, 160, 96]).to(device)
    net = SubeNet(in_channels=1,out_channels=2,base_filters_num=16).to(device)
    # net = MultiScaleFeatureFusionConvBlock3d(in_channels=1, out_channels=16).to(device)
    print('#parameters:', sum(param.numel() for param in net.parameters()))
    out,out2,feature,_,_,_ = net(tensor)
    # out = net(tensor)
    print(out.shape)