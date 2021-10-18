import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super(EncoderLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, padding=0)

    def forward(self, x):
        shortcut = self.layer(x)
        out = self.maxpool(shortcut)
        return out, shortcut


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias):
        super(DecoderLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = torch.cat((x, y), dim=1)
        out = self.layer(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias):
        super(BottleNeck, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.config = config
        bias = config['conv_bias']
        self.en_layer_1 = EncoderLayer(1, 30, bias)
        self.en_layer_2 = EncoderLayer(30, 60, bias)
        self.en_layer_3 = EncoderLayer(60, 120, bias)
        self.bottle_neck = BottleNeck(120, 240, 120, bias)
        self.de_layer_3 = DecoderLayer(240, 120, 60, bias)
        self.de_layer_2 = DecoderLayer(120, 60, 30, bias)
        self.de_layer_1 = DecoderLayer(60, 30, 30, bias)
        self.end_layer = nn.Conv2d(30, 1, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        en_1, sc_1 = self.en_layer_1(x)
        en_2, sc_2 = self.en_layer_2(en_1)
        en_3, sc_3 = self.en_layer_3(en_2)
        bneck = self.bottle_neck(en_3)
        de_3 = self.de_layer_3(bneck, sc_3)
        de_2 = self.de_layer_2(de_3, sc_2)
        de_1 = self.de_layer_1(de_2, sc_1)
        out = self.end_layer(de_1)
        out = torch.sigmoid(out)
        return out