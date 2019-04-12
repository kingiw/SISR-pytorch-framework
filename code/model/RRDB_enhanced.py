# RRDB_Net + Attention Module


import torch
import torch.nn as nn
import model.basic_blocks as B
from  .attention_module import Attention_Module

class RRDB_enhanced(nn.Module):
    """
    nb -- Number of RRDB in a trunk branch
    na -- Number of Attention Module
    nf -- Number of channel of the extrated feature by RRDB

    norm_type - normalization for RRDB
    act_type - Activation function for RRDB
    mode --  mode for RRDB (CNA, NAC, CNAC)
    """
    def __init__(self, args, norm_type=None, act_type='leakyrelu', mode='CNA'):
    # def __init__(self, nb=1, nf=64, na=2, norm_type=None, act_type='leakyrelu', mode='CNA'):

        super(RRDB_enhanced, self).__init__()

        nb = args.a_nb
        na = args.a_na
        nf = args.a_nf
        self.use_dense = args.a_dense_attention_modules
        

        self.fea_conv = B.conv_block(3, nf, kernel_size=3, norm_type=None, act_type=None)
        core = []
        for _ in range(na):
            # Number of attention module
            rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
            LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
            trunk = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)))
            core.append(Attention_Module(nf, nf, trunk=trunk))
            core.append(B.ResidualBlock(nf, nf))

        if self.use_dense:
            self.na1 = B.sequential(core[0], core[1])
            self.conv1 = B.conv_block(nf*2, nf, kernel_size=1, mode='CNA')
            self.na2 = B.sequential(core[2], core[3])
            self.conv2 = B.conv_block(nf*3, nf, kernel_size=1, mode='CNA')
            self.na3 = B.sequential(core[4], core[5])
            self.conv3 = B.conv_block(nf*4, nf, kernel_size=1, mode='CNA')

        else:
            self.core = B.sequential(*core)

        self.upsampler0 = B.upconv_block(nf, nf, 2)
        self.upsampler1 = B.upconv_block(nf, nf, 2)
        self.HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv1 = B.conv_block(nf, 3, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, x):
        x = self.fea_conv(x)
        if self.use_dense:  
            x1_out = self.na1(x)
            x1_cat = torch.cat((x,x1_out), 1)
            x2_in = self.conv1(x1_cat)
            x2_out = self.na2(x)
            x2_cat = torch.cat((x, x1_out, x2_out), 1)
            x3_in = self.conv2(x2_cat)
            x3_out = self.na3(x)
            x3_cat = torch.cat((x, x1_out, x2_out, x3_out), 1)
            x = self.conv3(x3_cat)
        else:
            x = self.core(x)
        x = self.upsampler0(x)
        x = self.upsampler1(x)
        x = self.HR_conv0(x)
        x = self.HR_conv1(x)
        return x

