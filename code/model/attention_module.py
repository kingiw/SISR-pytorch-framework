import torch
import torch.nn as nn

from basic_blocks import ResidualBlock_for_attention_module as rb
from basic_blocks import ResidualBlock

class Mask(nn.Module):
    """
    Mask branch in a Attention Module
    Input size is assumed to be 28*24
    Channel of input and output are assumed to be the same.
    """
    def __init__(self, channel):
        super(Mask, self).__init__()
        
        # Downsample by scale of 2
        # Unused if the shape of input are too small (like 28 * 24)
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.softmax1_blocks = rb(channel, channel)
        self.skip1_connection = rb(channel, channel)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = rb(channel, channel)
        self.skip2_connection = rb(channel, channel)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax3_blocks = nn.Sequential(
            rb(channel, channel),
            rb(channel, channel)
        )

        self.softmax4_blocks = rb(channel, channel)
        
        self.softmax5_blocks = rb(channel, channel)

    def forward(self, x):
        
        # Not to use the first maxpooling
        # out_mpoo1 = self.mpool1(x)


        # 28*24
        out_softmax1 = self.softmax1_blocks(x)
        out_skip1 = self.skip1_connection(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        # 14*12
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2 = self.skip2_connection(out_softmax2)

        out_mpool3 = self.mpool3(out_softmax2)
        # 7*6
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        
        # 14*12
        out = nn.functional.interpolate(out_softmax3, scale_factor=2, mode='bilinear') + out_softmax2 + out_skip2
        
        out_softmax4 = self.softmax4_blocks(out)

        # 28*24
        out = nn.functionnal.interpolate(out_softmax4, scale_factor=2, mode='bilinear') + out_softmax1 + out_skip1
        
        out_softmax5 = self.softmax5_blocks(out)
        out = nn.functionnal.interpolate(out_softmax5, scale_factor=2, mode='bilinear')
        
        return out

class Attention_Module(nn.Module):
    """
    Attention Module, containing both Mask branch and Trunk branch
    The channel and the size of input and output of trunk branch have to be the same
    """
    def __init__(self, in_c, out_c, trunk=None):
        self.first_residual_blocks = ResidualBlock(channel, channel)

        if trunk is not None:
            self.trunk_branches = trunk
        else:
            self.trunk_branches = nn.Sequential(
                ResidualBlock(channel, channel),
                ResidualBlock(channel, channel)
            )

        self.mask_branches = Mask(channel)

        self.softmax_blocks = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.last_blocks = ResidualBlock(in_c, out_c)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mask = self.mask_branches(x)

        assert(out_trunk.shape == out_mask.shape)

        out = self.softmax_blocks(out_trunk + out_mask)
        out = (1 + out) * out_trunck
        out = self.last_blocks(out)

        return out
        