import os
import pdb
import logging
import torch.nn as nn


BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    """Only replce the second 3x3 Conv with the TransformerBlocker"""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # 3x3卷积，改变通道数，可能下采样
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # 3x3卷积，什么都不改
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        # 对残差下采样
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 跟bottleneck相比，bottleneck将conv1拆成了1x1和3x3卷积，conv2通道膨胀
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out