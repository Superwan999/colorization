import torch
import torch.nn as nn
from layers import ConvBR, ResBasicBlock


class EncodeBlock(nn.Module):
    def __init__(self, c_in, c_out,
                 kernel_size=3, stride=1, padding=1,
                 weight_init=True, bn=True, activation='relu'):
        super(EncodeBlock, self).__init__()
        self.conv = ConvBR(c_in=c_in, c_out=c_out,
                           kernel_size=kernel_size, stride=stride, padding=padding,
                           weight_init=weight_init, bn=bn, activation=activation)
        self.resblock = ResBasicBlock(c_in=c_out, c_out=c_out,
                                      kernel_size=kernel_size, stride=1, padding=padding,
                                      weight_init=weight_init, activation=activation, bn=bn)

    def forward(self, x):
        out = self.conv(x)
        out = self.resblock(out)
        return out


class FeaturePart(nn.Module):
    def __init__(self, c_in, c_out, mode='advanced',
                 weight_init=True, activation='relu', bn=False):
        super(FeaturePart, self).__init__()
        self.mode = mode
        if mode == 'advanced':
            self.resblock1 = ResBasicBlock(c_in=c_in, c_out=c_out,
                                           kernel_size=3, stride=1, padding=1,
                                           weight_init=weight_init, activation=activation, bn=bn)
            self.resblock2 = ResBasicBlock(c_in=c_out, c_out=c_out,
                                           kernel_size=3, stride=1, padding=1,
                                           weight_init=weight_init, activation=activation, bn=bn)
        else:
            self.resblock1 = ResBasicBlock(c_in=c_in + 1, c_out=c_out,
                                           kernel_size=3, stride=1, padding=1,
                                           weight_init=weight_init, activation=activation, bn=bn)
            self.resblock2 = ResBasicBlock(c_in=c_out, c_out=c_out // 2,
                                           kernel_size=3, stride=1, padding=1,
                                           weight_init=weight_init, activation=activation, bn=bn)
            self.resblock3 = ResBasicBlock(c_in=c_out // 2, c_out=c_out,
                                           kernel_size=3, stride=1, padding=1,
                                           weight_init=weight_init, activation=activation, bn=bn)

    def forward(self, x, z):
        if self.mode == 'advanced':
            # print('      before resblock1 shape: ', x.shape)
            out = self.resblock1(x)
            # print('      after resblock1 shape: ', out.shape)
            out = self.resblock2(out)
            # print('      after resblock2 shape: ', out.shape)
        else:
            # gan based method
            out = self.resblock1(torch.cat([x, z], 1))      # concat noise
            out = self.resblock2(out)
            out = self.resblock3(out)
        return out


class DecodeBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3,
                 weight_init=True, bn=True, activation='relu',
                 merge_I=False):
        super(DecodeBlock, self).__init__()
        # print('in: ', c_in, '   out: ', c_out)
        self.merge_I = merge_I
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBR(c_in=c_in, c_out=c_out,
                           kernel_size=3, stride=1, padding=1,
                           weight_init=weight_init, bn=bn, activation=activation)

        self.bridge = ConvBR(c_in=c_in, c_out=c_out,
                             kernel_size=3, stride=1, padding=1,
                             weight_init=weight_init, bn=bn, activation=activation)
        if merge_I:
            self.resblock = ResBasicBlock(c_in=2 * c_out + 1, c_out=c_out,
                                          kernel_size=kernel_size, stride=1, padding=1,
                                          weight_init=weight_init, activation=activation, bn=bn)
        else:
            self.resblock = ResBasicBlock(c_in=2 * c_out, c_out=c_out,
                                          kernel_size=kernel_size, stride=1, padding=1,
                                          weight_init=weight_init, activation=activation, bn=bn)
        self.postmerge = ConvBR(c_in=c_out, c_out=c_out,
                                kernel_size=3, stride=1, padding=1,
                                weight_init=weight_init, bn=bn, activation=activation)

    def forward(self, up, down, I):
        # print('   up shape: ', up.shape)
        up = self.upsample(up)
        # print('   upsample shape: ', up.shape)
        up = self.conv(up)
        # print('   up conv shape: ', up.shape)
        down = self.bridge(down)
        # print('   down conv shape: ', down.shape)
        if self.merge_I:
            merge = self.resblock(torch.cat([up, down, I], 1))
            # print('   MERGE I : merge.shape: ', merge.shape)
        else:

            merge = self.resblock(torch.cat([up, down], 1))
            # print('   NOT MERGE I : merge.shape: ', merge.shape)
        out = self.postmerge(merge)
        # print('    merge shape: ', out.shape)
        return out


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        if args.mode == 'advanced':
            use_batch_norm = True
        else:
            use_batch_norm = args.use_batchnorm
        encode_feature_channel = [16, 32, 64, 128, 256, 256]
        decode_feature_channel = [8, 16, 32, 64, 128, 256]
        # Start
        # 256
        self.conv = ConvBR(c_in=1, c_out=encode_feature_channel[0],
                           kernel_size=5, stride=1, padding=2,
                           weight_init=True, bn=use_batch_norm, activation='leaky')

        # Encode
        # 256->128
        self.conv1 = EncodeBlock(encode_feature_channel[0], encode_feature_channel[1],
                                 kernel_size=5, stride=2, padding=2,
                                 weight_init=True, bn=use_batch_norm, activation='leaky')
        # 128->64
        self.conv2 = EncodeBlock(encode_feature_channel[1], encode_feature_channel[2],
                                 kernel_size=3, stride=2, padding=1,
                                 weight_init=True, bn=use_batch_norm, activation='leaky')
        # 64->32
        self.conv3 = EncodeBlock(encode_feature_channel[2], encode_feature_channel[3],
                                 kernel_size=3, stride=2, padding=1,
                                 weight_init=True, bn=use_batch_norm, activation='leaky')
        # 32->16
        self.conv4 = EncodeBlock(encode_feature_channel[3], encode_feature_channel[4],
                                 kernel_size=3, stride=2, padding=1,
                                 weight_init=True, bn=use_batch_norm, activation='leaky')
        # 16->8
        self.conv5 = EncodeBlock(encode_feature_channel[4], encode_feature_channel[5],
                                 kernel_size=3, stride=2, padding=1,
                                 weight_init=True, bn=use_batch_norm, activation='leaky')

        # Link
        self.features = FeaturePart(encode_feature_channel[5], encode_feature_channel[5], args.mode,
                                    weight_init=True, activation='leaky', bn=use_batch_norm)

        # Decode
        # 265 -> 128
        self.deconv5 = DecodeBlock(decode_feature_channel[5], decode_feature_channel[4], kernel_size=3,
                                   weight_init=True, bn=use_batch_norm, activation='leaky', merge_I=False)
        # 128 -> 64
        self.deconv4 = DecodeBlock(decode_feature_channel[4], decode_feature_channel[3], kernel_size=3,
                                   weight_init=True, bn=use_batch_norm, activation='leaky', merge_I=False)
        # 64 -> 32
        self.deconv3 = DecodeBlock(decode_feature_channel[3], decode_feature_channel[2], kernel_size=3,
                                   weight_init=True, bn=use_batch_norm, activation='leaky', merge_I=True)
        # 32 -> 16
        self.deconv2 = DecodeBlock(decode_feature_channel[2], decode_feature_channel[1], kernel_size=3,
                                   weight_init=True, bn=use_batch_norm, activation='leaky', merge_I=True)
        # 16 -> 8
        self.deconv1 = DecodeBlock(decode_feature_channel[1], decode_feature_channel[0], kernel_size=3,
                                   weight_init=True, bn=use_batch_norm, activation='leaky', merge_I=True)

        # final
        self.outconv2 = ConvBR(c_in= decode_feature_channel[0], c_out=3,
                               kernel_size=3, stride=1, padding=1,
                               weight_init=True, bn=use_batch_norm, activation='leaky')
        self.outconv1 = ConvBR(c_in=3, c_out=3,
                               kernel_size=3, stride=1, padding=1,
                               weight_init=False, bn=False, activation='None')

    def forward(self, I256, I128, I64, z=0):
        # print('I256 shape: ', I256.shape)
        conv = self.conv(I256)
        # print('conv shape: ', conv.shape)

        # encode
        conv1 = self.conv1(conv)
        # print('conv1 shape: ', conv1.shape)
        conv2 = self.conv2(conv1)
        # print('conv2 shape: ', conv2.shape)
        conv3 = self.conv3(conv2)
        # print('conv3 shape: ', conv3.shape)
        conv4 = self.conv4(conv3)
        # print('conv4 shape: ', conv4.shape)
        conv5 = self.conv5(conv4)
        # print('conv5 shape: ', conv5.shape)

        # feature
        feature = self.features(conv5, z)
        # print('feature shape: ', feature.shape)

        # decode
        deconv5 = self.deconv5(feature, conv4, None)
        # print('deconv5 shape: ', deconv5.shape)
        deconv4 = self.deconv4(deconv5, conv3, None)
        # print('deconv4 shape: ', deconv4.shape)
        deconv3 = self.deconv3(deconv4, conv2, I64)
        # print('deconv3 shape: ', deconv3.shape)
        deconv2 = self.deconv2(deconv3, conv1, I128)
        # print('deconv2 shape: ', deconv2.shape)
        deconv1 = self.deconv1(deconv2, conv, I256)
        # print('deconv1 shape: ', deconv1.shape)

        # final
        out2 = self.outconv2(deconv1)
        # print('out2 shape: ', out2.shape)
        out = self.outconv1(out2)
        # print('out shape: ', out.shape)

        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        #  C_in, C_out, kernel_size, stride, padding, bn=True, activation=nn.ReLU()
        use_batchnorm = args.use_batchnorm
        discriminator_channel = [32, 64, 128, 256, 256]
        self.conv0 = ConvBR(c_in=3, c_out=discriminator_channel[0],
                            kernel_size=3, stride=2, padding=1,
                            weight_init=True, bn=use_batchnorm, activation='leaky')
        self.conv1 = ConvBR(c_in=discriminator_channel[0], c_out=discriminator_channel[1],
                            kernel_size=3, stride=2, padding=1,
                            weight_init=True, bn=use_batchnorm, activation='leaky')
        self.conv2 = ConvBR(c_in=discriminator_channel[1], c_out=discriminator_channel[2],
                            kernel_size=3, stride=2, padding=1,
                            weight_init=True, bn=use_batchnorm, activation='leaky')
        self.conv3 = ResBasicBlock(c_in=discriminator_channel[2], c_out=discriminator_channel[3],
                                   kernel_size=3, stride=1, padding=1,
                                   weight_init=True, bn=use_batchnorm, activation='leaky')
        self.conv4 = ResBasicBlock(c_in=discriminator_channel[3], c_out=discriminator_channel[4],
                                   kernel_size=3, stride=1, padding=1,
                                   weight_init=True, bn=use_batchnorm, activation='leaky')
        self.conv5 = ConvBR(c_in=discriminator_channel[-1], c_out=1,
                            kernel_size=3, stride=1, padding=1,
                            weight_init=False, bn=False, activation='None')

    def forward(self, x):
        # 256 -> 128
        x = self.conv0(x)
        # 128 -> 64
        x = self.conv1(x)
        # 64 -> 32
        x = self.conv2(x)
        # 32 -> 32
        x = self.conv3(x)
        # 32 -> 32
        x = self.conv4(x)
        # 32 -> 32
        x = self.conv5(x)       # 8 x 8
        return x

