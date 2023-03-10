import torch
import torch.nn as nn
from layers import *


class EncodeBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(EncodeBlock, self).__init__()
        self.conv0 = DWConv(c_in, c_out, kernel_size=3, stride=2, padding=1)
        self.conv1 = ResBlock(c_out, c_out, stride=1, padding=1)

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        return out


class DecodeBlock(nn.Module):
    def __init__(self, c_in, c_out, use_bn=True, activation='relu',
                 merge_I=False):
        super(DecodeBlock, self).__init__()

        self.merge_I = merge_I
        self.up_sampleBlock = UpSampleBlock(c_in, c_out)
        self.bridge = DWConv(c_in, c_out, kernel_size=3, stride=1, padding=1)

        self.resBlock = ResBlock(c_in=2 * c_out, c_out=c_out)
        if merge_I:
            self.resBlock = ResBlock(c_in=2 * c_out + 1, c_out=c_out)
        self.postMerge = ConvBR(c_in=c_out, c_out=c_out, kernel_size=1, stride=1,
                                padding=0, use_bn=use_bn, activation=activation)

    def forward(self, up, down, I=None):
        up = self.up_sampleBlock(up)
        down = self.bridge(down)
        if self.merge_I:
            merge = torch.cat([up, down, I], dim=1)
        else:
            merge = torch.cat([up, down], dim=1)
        merge = self.resBlock(merge)
        out = self.postMerge(merge)
        return out


class FeaturePart(nn.Module):
    def __init__(self, c_in, c_out):
        super(FeaturePart, self).__init__()
        self.resblock1 = ResBlock(c_in=c_in + 1, c_out=c_out // 2)
        self.resblock2 = ResBlock(c_in=c_out // 2, c_out=c_out)
        self.resblock3 = ResBlock(c_in=c_out, c_out=c_out)

    def forward(self, x, z):
        merge = torch.cat([x, z], dim=1)
        out = self.resblock1(merge)
        out = self.resblock2(out)
        out = self.resblock3(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode_channels = [8, 16, 32, 64, 128, 128]
        self.decode_channels = [128, 64, 32, 16, 8, 8]
        self.conv0 = DWConv(c_in=1, c_out=self.encode_channels[0], kernel_size=7, stride=1, padding=3)
        self.encode_layers, self.decode_layers = self.make_layers()
        self.featurePart = FeaturePart(self.encode_channels[-1], self.encode_channels[-1])
        self.out_conv = DWConv(c_in=self.decode_channels[-1], c_out=3, kernel_size=3, stride=1, padding=1)
        self.out_conv2 = DWConv(c_in=3, c_out=3, kernel_size=3, stride=1, padding=1, use_bn=False, activation='None')

    def forward(self, I, I_2, I_4, z=0):
        down_features = []
        x = self.conv0(I)
        # print("the len encode_layer:", len(self.encode_layers))

        for i, encode_layer in enumerate(self.encode_layers):
            down_features.append(x)
            x = encode_layer(x)
        # down_features.append(x)
        # print(f"the len decode_layer:{len(self.encode_layers)}, len down_features: {len(down_features)}", )
        up = self.featurePart(x, z)
        # print(f"the up: {up.shape}")
        for j, decode_layer in enumerate(self.decode_layers):
            # print(f"the j:{j} ,up: {up.shape}, down_features[-{j} - 1]: {down_features[-j - 1].shape}")

            # if j < len(self.decode_layers) - 1:
            if I.size(3) == down_features[-j - 1].size(3):
                up = decode_layer(up, down_features[-j - 1], I)
            elif I_2.size(3) == down_features[-j - 1].size(3):
                up = decode_layer(up, down_features[-j - 1], I_2)
            elif I_4.size(3) == down_features[-j - 1].size(3):
                up = decode_layer(up, down_features[-j - 1], I_4)
            else:
                up = decode_layer(up, down_features[-j - 1], None)
        out = self.out_conv(up)
        out = self.out_conv2(out)
        return out

    def make_layers(self):
        encode_layers = nn.ModuleList()
        decode_layers = nn.ModuleList()
        for i in range(len(self.encode_channels) - 1):
            encode_layers.append(EncodeBlock(c_in=self.encode_channels[i], c_out=self.encode_channels[i + 1]))
        for j in range(len(self.decode_channels) - 1):
            if j >= 3:
                decode_layers.append(DecodeBlock(c_in=self.decode_channels[j], c_out=self.decode_channels[j + 1], merge_I=True))
            else:
                decode_layers.append(DecodeBlock(c_in=self.decode_channels[j], c_out=self.decode_channels[j + 1]))

        return encode_layers, decode_layers


if __name__ == "__main__":
    from torchstat import stat
    img = torch.rand((1, 1, 256, 256))
    img2 = torch.rand((1, 1, 128, 128))
    img3 = torch.rand((1, 1, 64, 64))
    z = torch.rand([1, 1, 8, 8])
    net = Generator()
    y = net(img, img2, img3, z)
    print("shape of y:", y.shape)
    input_size1 = (1, 256, 256)
    input_size2 = (1, 128, 128)
    input_size3 = (1, 64, 64)
    input_size4 = (1, 8, 8)
    stat(net, input_size1)
