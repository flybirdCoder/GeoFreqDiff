import torch
from torch import nn
import torch.nn.functional as F


class AsymmetricConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.horizontal_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), padding=(0, 1))
        self.vertical_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(1, 0))
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        horizontal_out = self.horizontal_conv(x)
        vertical_out = self.vertical_conv(horizontal_out)
        return self.leaky(vertical_out)


class SymmetricConvolution(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=5):
        super(SymmetricConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      bias=False), nn.BatchNorm2d(growthRate)
        )
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.leaky(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class SobelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out


class SpatialPerceptionUnit(nn.Module):
    def __init__(self, nChannels, growthRate=64):
        super(SpatialPerceptionUnit, self).__init__()
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=6, dilation=6,
                               bias=False)
        self.conv2 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
                               bias=False)
        self.conv3 = SobelConv2d(nChannels, growthRate, kernel_size=3, stride=1, padding=(3 - 1) // 2)
        self.conv6 = nn.Conv2d(growthRate * 3, nChannels, kernel_size=1, padding=(1 - 1) // 2,
                               bias=False)
        self.leaky1 = nn.LeakyReLU(0.1, inplace=True)
        self.leaky2 = nn.LeakyReLU(0.1, inplace=True)
        self.leaky3 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x_1 = self.leaky1(self.conv1(x))
        x_2 = self.leaky2(self.conv2(x))
        x_3 = self.leaky3(self.conv3(x))
        x_0 = torch.cat((x_1, x_2, x_3), dim=1)
        x_0 = self.conv6(x_0)

        out = x_0 + x
        return out


class SFCRM(nn.Module):
    def __init__(self, nChannels, growthRate=32):
        super(SFCRM, self).__init__()

        self.spu = SpatialPerceptionUnit(nChannels)
        amp_channel = nChannels
        pha_channel = nChannels

        # Fourier-domain amplitude-phase convolution unit
        self.amp_layers = SymmetricConvolution(amp_channel, growthRate)
        amp_channel += growthRate

        self.pha_layers = SymmetricConvolution(pha_channel, growthRate)
        pha_channel += growthRate

        self.aconv_1 = AsymmetricConv(amp_channel, nChannels)
        self.aconv_2 = AsymmetricConv(pha_channel, nChannels)

    def forward(self, x):
        x = self.spu(x)
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.amp_layers(mag)
        mag = self.aconv_1(mag)
        pha = self.pha_layers(pha)
        pha = self.aconv_2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        out = out + x
        return out
