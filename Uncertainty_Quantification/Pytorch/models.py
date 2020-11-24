## Based off the code of the 2d Unet from https://github.com/OpenImageDenoise/oidn/blob/master/training/model.py

import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ssim import MS_SSIM
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from argparse import ArgumentParser

logger = logging.getLogger(__name__)

BASE_FILTER = 16


class MSSSIMLoss(LightningModule):
    def __init__(self):
        super(MSSSIMLoss, self).__init__()
        self.msssim = MS_SSIM(
            data_range=60.0, channel=2
        )  # after standardization ~0+-60 through analysis, channel = 2 for 2 pressure levels

    def forward(self, x, target):
        return 1.0 - self.msssim(x, target)


class MixLoss(LightningModule):
    def __init__(self, loss1, loss2, alpha):
        super(MixLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha

    def forward(self, x, target):
        x = x[:, 0, :, :, :]
        target = target[:, 0, :, :, :]
        return (1.0 - self.alpha) * self.loss1(x, target) + self.alpha * self.loss2(
            x, target
        )


# 3x3x3 convolution module
def Conv(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, 3, padding=1)


# Activation function
def activation(x):
    # Simple relu
    return F.relu(x, inplace=True)


# 1x2x2 max pool function
def pool(x):
    return F.max_pool3d(x, kernel_size=[1, 2, 2], stride=[1, 2, 2])


# 1x2x2 nearest-neighbor upsample function
def upsample(x):
    return F.interpolate(x, scale_factor=[1, 2, 2], mode="trilinear")


# Channel concatenation function
def concat(a, b):
    return torch.cat((a, b), 1)


# Basic UNet Model


class unet3d(LightningModule):
    def __init__(self, sample_nr, base_lr, max_lr, in_channels=7, out_channels=1):
        super(unet3d, self).__init__()

        self.half_cycle_nr = sample_nr // 2
        self.base_lr = base_lr
        self.max_lr = max_lr

        self.loss_fct = MixLoss(torch.nn.L1Loss(), MSSSIMLoss(), 0.84)

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
        self.enc_conv0 = Conv(ic, ec1)
        self.enc_conv1 = Conv(ec1, ec1)
        self.enc_conv2 = Conv(ec1, ec2)
        self.enc_conv3 = Conv(ec2, ec3)
        self.enc_conv4 = Conv(ec3, ec4)
        self.enc_conv5 = Conv(ec4, ec5)
        self.dec_conv5a = Conv(ec5 + ec4, dc5)
        self.dec_conv5b = Conv(dc5, dc5)
        self.dec_conv4a = Conv(dc5 + ec3, dc4)
        self.dec_conv4b = Conv(dc4, dc4)
        self.dec_conv3a = Conv(dc4 + ec2, dc3)
        self.dec_conv3b = Conv(dc3, dc3)
        self.dec_conv2a = Conv(dc3 + ec1, dc2)
        self.dec_conv2b = Conv(dc2, dc2)
        self.dec_conv1a = Conv(dc2 + ic, dc1a)
        self.dec_conv1b = Conv(dc1a, dc1b)
        self.dec_conv0 = Conv(dc1b, oc)

    def forward(self, inp):
        # Encoder

        x = activation(self.enc_conv0(inp))  # enc_conv0

        x = activation(self.enc_conv1(x))  # enc_conv1
        x = pool1 = pool(x)  # pool1

        x = activation(self.enc_conv2(x))  # enc_conv2
        x = pool2 = pool(x)  # pool2

        x = activation(self.enc_conv3(x))  # enc_conv3
        x = pool3 = pool(x)  # pool3

        x = activation(self.enc_conv4(x))  # enc_conv4
        x = pool4 = pool(x)  # pool4

        x = activation(self.enc_conv5(x))  # enc_conv5
        x = pool(x)  # pool5

        # Decoder

        x = upsample(x)  # upsample5
        x = concat(x, pool4)  # concat5
        x = activation(self.dec_conv5a(x))  # dec_conv5a
        x = activation(self.dec_conv5b(x))  # dec_conv5b

        x = upsample(x)  # upsample4
        x = concat(x, pool3)  # concat4
        x = activation(self.dec_conv4a(x))  # dec_conv4a
        x = activation(self.dec_conv4b(x))  # dec_conv4b

        x = upsample(x)  # upsample3
        x = concat(x, pool2)  # concat3
        x = activation(self.dec_conv3a(x))  # dec_conv3a
        x = activation(self.dec_conv3b(x))  # dec_conv3b

        x = upsample(x)  # upsample2
        x = concat(x, pool1)  # concat2
        x = activation(self.dec_conv2a(x))  # dec_conv2a
        x = activation(self.dec_conv2b(x))  # dec_conv2b

        x = upsample(x)  # upsample1
        x = concat(x, inp)  # concat1
        x = activation(self.dec_conv1a(x))  # dec_conv1a
        x = activation(self.dec_conv1b(x))  # dec_conv1b

        x = self.dec_conv0(x)  # dec_conv0

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.base_lr,
            max_lr=self.max_lr,
            step_size_up=self.half_cycle_nr,
            step_size_down=None,
            mode="triangular2",
            gamma=1.0,
            scale_fn=None,
            scale_mode="cycle",
            cycle_momentum=False,
        )
        return [optimizer]  # , [scheduler]

    def training_step(self, batch, batch_idx):
        logger.debug("Processing training batch %s", batch_idx)
        x, y = batch
        out = self(x)
        loss = self.loss_fct(out, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logger.debug("Processing validation batch %s", batch_idx)
        x, y = batch
        out = self(x)
        loss = self.loss_fct(out, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        logger.debug("Processing test batch %s", batch_idx)
        x, y = batch
        out = self(x)
        loss = torch.nn.MSELoss()(out, y)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser
