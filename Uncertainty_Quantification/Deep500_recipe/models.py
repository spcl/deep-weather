import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *

BASE_FILTER = 32  # has to be divideable by 4, originally 16


# Conv Batch Relu module
class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes=3):
        super(ConvBatchRelu, self).__init__()
        self.conv = Conv(in_channels, out_channels, filter_sizes)
        self.bnr = batch_norm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = activation(self.bnr(x))
        return x


# Inception-style layer, without the last convolutional layer used in the original version
class Incep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Incep, self).__init__()
        self.conv1 = Conv(in_channels, out_channels // 4, 1)
        self.conv3 = Conv(in_channels, out_channels // 4, 3)
        self.conv5 = Conv(in_channels, out_channels // 4, 5)
        self.conv7 = Conv(in_channels, out_channels // 4, 7)
        self.bnr = batch_norm((out_channels // 4) * 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x_t = activation(self.bnr(torch.cat((x3, x5, x7), dim=1)))
        x_f = torch.cat((x_t, x1), dim=1)
        return x_f


# 3x3x3 convolution module
def Conv(in_channels, out_channels, filter_sizes=3, dim=2):
    if dim == 2:
        return nn.Conv2d(in_channels,
                         out_channels,
                         filter_sizes,
                         padding=(filter_sizes - 1) // 2)
    else:
        return nn.Conv3d(in_channels,
                         out_channels,
                         filter_sizes,
                         padding=(filter_sizes - 1) // 2)


# Activation function
def activation(x):
    # Simple relu
    return F.relu(x, inplace=True)


# 1x2x2 max pool function
def pool(x, dim=2):
    if dim == 2:
        return F.max_pool2d(x, kernel_size=[2, 2], stride=[2, 2])
    else:
        return F.max_pool3d(x, kernel_size=[1, 2, 2], stride=[1, 2, 2])


# 1x2x2 nearest-neighbor upsample function
def upsample(x, dim=2):
    if dim == 2:
        return F.interpolate(x, scale_factor=[2, 2], mode="bilinear")
    else:
        return F.interpolate(x, scale_factor=[1, 2, 2], mode="trilinear")


# Channel concatenation function
def concat(a, b):
    return torch.cat((a, b), dim=1)


def batch_norm(
    out_channels,
    dim=2
):  # Does not have the same parameters as the original batch normalization used in the tensorflow 1.14 version of this project
    if dim == 2:
        bnr = torch.nn.BatchNorm2d(out_channels)
    else:
        bnr = torch.nn.BatchNorm3d(out_channels)
    return bnr


class unet3d(nn.Module):
    def __init__(self,
                 sample_nr,
                 base_lr,
                 max_lr,
                 in_channels=7,
                 out_channels=1,
                 args=None):
        super(unet3d, self).__init__()

        self.dim = args.dims
        self.half_cycle_nr = sample_nr // 2
        self.base_lr = base_lr
        self.max_lr = max_lr

        # Number of channels per layer
        ic = in_channels
        ec1 = BASE_FILTER * 2
        ec2 = BASE_FILTER * 3
        ec3 = BASE_FILTER * 4
        ec4 = BASE_FILTER * 5
        ec5 = BASE_FILTER * 7
        dc5 = BASE_FILTER * 10
        dc4 = BASE_FILTER * 7
        dc3 = BASE_FILTER * 6
        dc2 = BASE_FILTER * 4
        dc1a = BASE_FILTER * 4
        dc1b = BASE_FILTER * 2
        oc = out_channels

        # Convolutions
        self.enc_conv0 = Conv(ic, ec1, dim=self.dim)
        self.enc_conv1 = Conv(ec1, ec1, dim=self.dim)
        self.enc_conv2 = Conv(ec1, ec2, dim=self.dim)
        self.enc_conv3 = Conv(ec2, ec3, dim=self.dim)
        self.enc_conv4 = Conv(ec3, ec4, dim=self.dim)
        self.enc_conv5 = Conv(ec4, ec5, dim=self.dim)
        self.dec_conv5a = Conv(ec5 + ec4, dc5, dim=self.dim)
        self.dec_conv5b = Conv(dc5, dc5, dim=self.dim)
        self.dec_conv4a = Conv(dc5 + ec3, dc4, dim=self.dim)
        self.dec_conv4b = Conv(dc4, dc4, dim=self.dim)
        self.dec_conv3a = Conv(dc4 + ec2, dc3, dim=self.dim)
        self.dec_conv3b = Conv(dc3, dc3, dim=self.dim)
        self.dec_conv2a = Conv(dc3 + ec1, dc2, dim=self.dim)
        self.dec_conv2b = Conv(dc2, dc2, dim=self.dim)
        self.dec_conv1a = Conv(dc2 + ic, dc1a, dim=self.dim)
        self.dec_conv1b = Conv(dc1a, dc1b, dim=self.dim)
        self.dec_conv0 = Conv(dc1b, oc, dim=self.dim)

    def forward(self, inp):
        # Encoder

        x = activation(self.enc_conv0(inp))  # enc_conv0

        x = activation(self.enc_conv1(x))  # enc_conv1
        x = pool1 = pool(x, dim=self.dim)  # pool1

        x = activation(self.enc_conv2(x))  # enc_conv2
        x = pool2 = pool(x, dim=self.dim)  # pool2

        x = activation(self.enc_conv3(x))  # enc_conv3
        x = pool3 = pool(x, dim=self.dim)  # pool3

        x = activation(self.enc_conv4(x))  # enc_conv4
        x = pool4 = pool(x, dim=self.dim)  # pool4

        x = activation(self.enc_conv5(x))  # enc_conv5
        x = pool(x, dim=self.dim)  # pool5

        # Decoder

        x = upsample(x, dim=self.dim)  # upsample5
        x = concat(x, pool4)  # concat5
        x = activation(self.dec_conv5a(x))  # dec_conv5a
        x = activation(self.dec_conv5b(x))  # dec_conv5b

        x = upsample(x, dim=self.dim)  # upsample4
        x = concat(x, pool3)  # concat4
        x = activation(self.dec_conv4a(x))  # dec_conv4a
        x = activation(self.dec_conv4b(x))  # dec_conv4b

        x = upsample(x, dim=self.dim)  # upsample3
        x = concat(x, pool2)  # concat3
        x = activation(self.dec_conv3a(x))  # dec_conv3a
        x = activation(self.dec_conv3b(x))  # dec_conv3b

        x = upsample(x, dim=self.dim)  # upsample2
        x = concat(x, pool1)  # concat2
        x = activation(self.dec_conv2a(x))  # dec_conv2a
        x = activation(self.dec_conv2b(x))  # dec_conv2b

        x = upsample(x, dim=self.dim)  # upsample1
        x = concat(x, inp)  # concat1
        x = activation(self.dec_conv1a(x))  # dec_conv1a
        x = activation(self.dec_conv1b(x))  # dec_conv1b

        x = self.dec_conv0(x)  # dec_conv0

        return x


class resnet2d_simple(nn.Module):
    def __init__(self, in_channels=7, out_channels=1):
        super(resnet2d_simple, self).__init__()

        # Number of channels per layer
        ic = in_channels
        ec1 = BASE_FILTER * 2
        ec2 = BASE_FILTER * 3
        ec3 = BASE_FILTER * 4
        ec4 = BASE_FILTER * 5
        ec5 = BASE_FILTER * 7
        oc = out_channels

        # TODO: attention for filter dim
        # Convolutions
        self.convBR0 = ConvBatchRelu(ic, ec1)
        self.convBR1 = ConvBatchRelu(ec1, ec2)
        self.inc1 = Incep(ec2 + 1, ec3)
        self.inc2 = Incep(ec3, ec4)
        self.inc3 = Incep(ec4, ec5)
        self.inc4 = Incep(ec5, ec5)
        self.inc5 = Incep(ec5, ec4)
        self.inc6 = Incep(ec4, ec3)
        self.conv4 = Conv(
            ec3 + 1, oc, 1
        )  # can be combined to be a single convolution, replace next layer by attention (TODO)
        # self.conv5 = Conv(ec2, oc, 1)

    def forward(self, inp):

        x = self.convBR0(inp)
        x = self.convBR1(x)
        x = self.inc1(torch.cat((inp[:, 28, None, :, :], x), dim=1))  # 42
        x = self.inc2(x)
        x = self.inc3(x)
        x = self.inc4(x)
        x = self.inc5(x)
        x = self.inc6(x)
        x = self.conv4(torch.cat((inp[:, 28, None, :, :], x), dim=1))  # 42
        # x = self.conv5(x)
        return x
