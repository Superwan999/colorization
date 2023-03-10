import torch
import torch.nn as nn
from layers import *


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.channels = [3, 8, 16, 16, 32, 64]
        self.strides = [2, 2, 2, 1, 1, 1, 1]
        self.convs = self.make_layers()

    def forward(self, x):
        out = self.convs(x)
        return out

    def make_layers(self):
        layers = []
        for i in range(len(self.channels) - 1):
            if i == 0:
                layers.append(DWConv(c_in=self.channels[i], c_out=self.channels[i + 1], kernel_size=7,
                                     stride=self.strides[i], padding=3))
            else:
                layers.append(DWConv(c_in=self.channels[i], c_out=self.channels[i + 1], kernel_size=3,
                                     stride=self.strides[i], padding=1))
                layers.append(ResBlock(self.channels[i + 1], self.channels[i + 1], stride=1,
                                       padding=1, expansion=4))
        return nn.Sequential(*layers)


if __name__ == "__main__":
    z = torch.rand((1, 3, 256, 256))
    net = Discriminator()
    pred = net(z)
    print("pred shape: ", pred.shape)
    print(z.size(2))
    from torchstat import stat
    stat(net, (3, 256, 256))
