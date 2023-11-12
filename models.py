import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg_model import vgg19
from torch.nn import Parameter

class ResBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottle_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.bottle_conv(x)
        x = self.double_conv(x) + x
        return x / math.sqrt(2)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, True)
        )
    def forward(self, x):
        x = self.double_conv(x)
        return x

class DownRes(nn.Module):
    """Downscaling with stride conv then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            ResBlock(in_channels, out_channels)
        )    
    def forward(self, x):
        x = self.main(x)
        return x


class SDFT(nn.Module):
    def __init__(self, norm_dim, channels, kernel_size = 3):
        super().__init__()
        # generate global conv weights
        fan_in = channels * kernel_size ** 2
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.scale = 1 / math.sqrt(fan_in)
        self.modulation = nn.Conv2d(norm_dim, channels, 1)
        self.weight = nn.Parameter(
            torch.randn(1, channels, channels, kernel_size, kernel_size)
        )
    def forward(self, fea, norm_feat):
        B, C, H, W = fea.size()
        style = self.modulation(norm_feat).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style
        # demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, C, 1, 1, 1)
        weight = weight.view(
            B * C, C, self.kernel_size, self.kernel_size
        )
        fea = fea.view(1, B * C, H, W)
        fea = F.conv2d(fea, weight, padding=self.padding, groups=B)
        fea = fea.view(B, C, H, W)
        return fea

class UpBlock(nn.Module):
    def __init__(self, normfeat_dim, in_channels, out_channels, kernel_size = 3, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)     
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels // 2 + in_channels // 8, out_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_s = nn.Conv2d(in_channels//2, out_channels, 1, 1, 0)
        # generate global conv weights
        self.SDFT = SDFT(normfeat_dim, out_channels, kernel_size)

    def forward(self, x1, x2, normal_feat):
        x1 = self.up(x1)
        x1_s = self.conv_s(x1)
        x = torch.cat([x1, x2[:, ::4, :, :]], dim=1)
        x = self.conv_cat(x)
        x = self.SDFT(x, normal_feat)
        x = x + x1_s
        return x
    
class NormalEncoder(nn.Module):
    def __init__(self, norm_dim=512):
        super(NormalEncoder, self).__init__()
        # self.vgg = vgg19(pretrained_path=None)
        self.vgg =  vgg19(pretrained_path = '/home/xteam/PaperCode/data_zoo/vgg19-dcbb9e9d.pth', require_grad = False)
        self.feature2vector = nn.Sequential(
            nn.Conv2d(norm_dim, norm_dim, 4, 2, 2), # 8x8
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(norm_dim, norm_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(norm_dim, norm_dim, 4, 2, 2), # 4x4
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(norm_dim, norm_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),

            nn.AdaptiveAvgPool2d((1, 1)), # 1x1

            nn.Conv2d(norm_dim, norm_dim//2, 1), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(norm_dim//2, norm_dim//2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(norm_dim//2, norm_dim, 1), 
        )
        # d = 512
    def forward(self, x):
        # x #[0, 1] RGB
        b, c, w, h = x.shape
        vgg_fea = self.vgg(x.expand([b, 3, w, h]), layer_name='relu5_2') # [B, 512, 16, 16]
        normal_feat = self.feature2vector(vgg_fea[-1]) # [B, 512, 1, 1]
        return normal_feat

class NormalDecoder(nn.Module):
    def __init__(self, n_channels=1, bilinear=True):
        super(NormalDecoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownRes(64, 128)
        self.down2 = DownRes(128, 256)
        self.down3 = DownRes(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownRes(512, 1024 // factor)

        self.up1 = UpBlock(512, 1024, 512 // factor, 3, bilinear)
        self.up2 = UpBlock(512, 512, 256 // factor, 3, bilinear)
        self.up3 = UpBlock(512, 256, 128 // factor, 5, bilinear)
        self.up4 = UpBlock(512, 128, 64, 5, bilinear)
        self.outc = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(64, 3, 3, 1, 1),
                nn.Tanh()
        )

    def forward(self, x):
        normal_feat = x[1] # [B, 512, 1, 1]

        x1 = self.inc(x[0]) # [B, 64, 256, 256]
        x2 = self.down1(x1) # [B, 128, 128, 128]
        x3 = self.down2(x2) # [B, 256, 64, 64]
        x4 = self.down3(x3) # [B, 512, 32, 32]
        x5 = self.down4(x4) # [B, 512, 16, 16]

        x6 = self.up1(x5, x4, normal_feat) # [B, 256, 32, 32]
        x7 = self.up2(x6, x3, normal_feat) # [B, 128, 64, 64]
        x8 = self.up3(x7, x2, normal_feat) # [B, 64, 128, 128]
        x9 = self.up4(x8, x1, normal_feat) # [B, 64, 256, 256]
        normal = self.outc(x9)

        return normal
    


class UNet(nn.Module):
    # initializers
    def __init__(self, channel=1, d=64):
        super(UNet, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(channel, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)

        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        o = torch.tanh(d8)

        return o

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
