import os
import logging
import torch.nn as nn


BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):

    # after processed by this block, channel num expand by 4 times
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # conv1 1x1, from inplanes to planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        # self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # conv2 3x3, if stride!=1, down-sample
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1
        )
        # self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # conv3 1x1, expand channels
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1
        )
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # for residual, downsample or change channel num
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # residual connection
        out += residual
        out = self.relu(out)

        return out


class BottleneckDWP(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckDWP, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=planes,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out