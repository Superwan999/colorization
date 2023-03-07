import torch.nn as nn
from torch.nn.init import xavier_normal, kaiming_normal
import torch.nn.functional as F


class ConvBR(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, weight_init=True, bn=True, activation='relu'):
        super(ConvBR, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(1e-2)
        else:
            # activation == 'None':
            self.activation = nn.Identity()
        self.use_bn = bn
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(c_out)
        if weight_init:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResBasicBlock(nn.Module):
    def __init__(self, c_in, c_out,
                 kernel_size, stride=1, padding=1,
                 weight_init=True, activation='relu', bn=True):
        super(ResBasicBlock, self).__init__()
        self.conv1 = ConvBR(c_in=c_in, c_out=c_in,
                            kernel_size=kernel_size, stride=1,
                            padding=padding if padding is not None else (kernel_size - 1)//2,
                            weight_init=weight_init, activation=activation, bn=bn)
        self.conv2 = ConvBR(c_in=c_in, c_out=c_out,
                            kernel_size=kernel_size, stride=1,
                            padding=padding if padding is not None else (kernel_size - 1)//2,
                            weight_init=False, activation='None', bn=False)
        if stride == 1 and c_in == c_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ConvBR(c_in=c_in, c_out=c_out,
                                   kernel_size=1, stride=stride, padding=0,
                                   weight_init=False, activation='None', bn=False)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(1e-2)
        else:
            # self.activation = 'None
            self.activation = nn.Identity()

    def forward(self, x):
        # print('        x shape: ', x.shape)
        identity = self.shortcut(x)
        # print('        identity shape: ', identity.shape)

        out = self.conv1(x)
        # print('        conv1 shape: ', out.shape)
        out = self.conv2(out)
        # print('        conv2 shape: ', out.shape)
        out = identity + out
        # print('        sum shape: ', out.shape)
        out = self.activation(out)
        # print('        act shape: ', out.shape)
        return out


