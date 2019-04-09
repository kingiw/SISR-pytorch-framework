# RDDB_Net + Attention Module


import torch
import torch.nn as nn
import model.basic_blocks as B
from  .attention_module import Attention_Module

class RDDB_enhanced(nn.Module):
    """
    nb -- Number of RDDB in a trunk branch
    na -- Number of Attention Module
    nf -- Number of channel of the extrated feature by RDDB

    norm_type - normalization for RDDB
    act_type - Activation function for RDDB
    mode --  mode for RDDB (CNA, NAC, CNAC)
    """
    def __init__(self, nb=1, nf=64, na=2, norm_type=None, act_type='leakyrelu', mode='CNA'):

        super(RDDB_enhanced, self).__init__()
        self.fea_conv = B.conv_block(3, nf, kernel_size=3, norm_type=None, act_type=None)
        self.na = na
        core = []
        for _ in range(na):
            # Number of attention module
            rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
            LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
            trunk = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)))
            core.append(Attention_Module(nf, nf, trunk=trunk))
            core.append(B.ResidualBlock(nf, nf))
        self.core = B.sequential(*core)
        self.upsampler0 = B.upconv_block(nf, nf, 2)
        self.upsampler1 = B.upconv_block(nf, nf, 2)
        self.HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv1 = B.conv_block(nf, 3, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, x):
        x = self.fea_conv(x)
        x = self.core(x)
        x = self.upsampler0(x)
        x = self.upsampler1(x)
        x = self.HR_conv0(x)
        x = self.HR_conv1(x)
        return x
    