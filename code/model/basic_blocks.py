import torch
import torch.nn as nn



def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResidualBlock(nn.Module):
    """
    Bottleneck of https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, expansion=4):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.expansion = expansion
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualBlock_for_attention_module(nn.Module):
    """
    In->BN->ReLU->Conv->(BN->ReLU->Conv)*2---+-->Out
                |------------Conv------------|
    """
    def __init__(self, in_c, out_c, stride=1):
        self.in_c, self.out_c = in_c, out_c
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_c, out_c//4, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_c//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c//4, out_c//4, 3, stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_c//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_c//4, out_c, 1, 1, bias = False)
        self.conv4 = nn.Conv2d(in_c, out_c , 1, stride, bias = False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.in_c != self.out_c) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out

