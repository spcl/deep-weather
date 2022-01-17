import torch
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM


class MSSSIMLoss(torch.nn.Module):
    def __init__(self, dim=2):
        super(MSSSIMLoss, self).__init__()
        if dim == 2:
            self.msssim = MS_SSIM(data_range=4.3, channel=1)
        else:
            self.msssim = MS_SSIM(
                data_range=4.3,
                channel=2  # 60 for mean
            )  # after standardization ~0+-60 through analysis, channel = 2 for 2 pressure levels

    def forward(self, x, target):
        return 1.0 - self.msssim(x, target)


class MixLoss(torch.nn.Module):
    def __init__(self, loss1, loss2, alpha, dim=2):
        super(MixLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha
        self.dim = dim

    def forward(self, x, target):
        if not x.type() == target.type():
            target = target.type(x.type())
        if self.dim == 3:
            x = x[:, 0, :, :, :]
            target = target[:, 0, :, :, :]
        return (1.0 - self.alpha) * self.loss1(
            x, target) + self.alpha * self.loss2(x, target)
