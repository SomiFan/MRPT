import torch
import torch.nn as nn


class NLRNResBlock(nn.Module):
    """
    For all convolution operations in the residual module,
    the size of the convolution kernel is set to 64×3×3×64.
    The zero-padding operation ensures that the entire
    network runs at the same resolution.
    """

    def __init__(self):
        super(NLRNResBlock, self).__init__()

        self.conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.in1 = nn.InstanceNorm2d(64)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.in2 = nn.InstanceNorm2d(64)
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out = out * self.resweight

        residual = residual + out

        return residual
