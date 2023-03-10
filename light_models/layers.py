import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBR(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding,
                 weight_init=True, use_bn=True, activation='relu'):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm2d(c_out)
        self.activation = nn.Identity()
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(1e-2, inplace=True)
        if weight_init:
            self._init_weights()

    def forward(self, x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        out = self.activation(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DWConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding,
                 weight_init=True, use_bn=True, activation='leaky'):
        super(DWConv, self).__init__()
        self.conv_d = nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=False, groups=c_in)
        self.conv_w = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm2d(c_out)
        self.activation = nn.Identity()
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(1e-2, inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        if weight_init:
            self._initialize_weight()

    def forward(self, x):
        x = self.conv_d(x)
        x = self.conv_w(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, padding=1, expansion=4):
        super(ResBlock, self).__init__()

        self.conv1 = DWConv(c_in, c_in * expansion,
                            kernel_size=3, stride=1, padding=1)
        self.conv2 = DWConv(c_in * expansion, c_in * expansion,
                            kernel_size=3, stride=stride,
                            padding=padding if padding is not None else 1)
        self.conv3 = ConvBR(c_in * expansion, c_out, kernel_size=1,
                            stride=1, padding=0)

        if stride == 1 and c_in == c_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ConvBR(c_in, c_out, kernel_size=1, stride=stride,
                                   padding=0, weight_init=False, use_bn=False, activation='None')

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + shortcut
        return out


class UpSampleBlock(nn.Module):
    def __init__(self, c_in, c_out, use_bn=True, activation='relu'):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',
                                              align_corners=True),
                                  DWConv(c_in, c_out, 3, 1, 1, use_bn=use_bn, activation=activation))

    def forward(self, x):
        x = self.conv(x)
        return x
