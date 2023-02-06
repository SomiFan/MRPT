import torch.nn as nn


class ResBlock(nn.Module):
    """simple and plain res-block"""

    def __init__(self, inplanes, planes, ks=3):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, ks, 1, padding=(ks - 1) // 2, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, inplanes, ks, 1, padding=(ks - 1) // 2, bias=True)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out += residual
        return self.act2(out)


class DWResBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(DWResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        # self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
        )
        # self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, inplanes, kernel_size=1
        )
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample
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

        #if self.downsample is not None:
        #   residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out