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
        self.na = na
        

        self.fea_conv = B.conv_block(3, nf, kernel_size=3, norm_type=None, act_type=None)
        self.core = nn.ModuleList()
        for _ in range(na):
            # Number of attention module
            rb_blocks = [
                B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=norm_type, act_type=act_type, mode='CNA', use_ca=args.a_ca) \
                for _ in range(nb)]
            LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
            trunk = B.sequential(B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)))
            self.core.append(Attention_Module(nf, nf, trunk=trunk))
            self.core.append(B.ResidualBlock(nf, nf))


        if self.use_dense:
            self.convList = nn.ModuleList()
            for i in range(na):
                self.convList.append(B.conv_block(nf*(i+2), nf, kernel_size=1, norm_type=None, mode='CNA'))
        else:
            self.core = B.sequential(*self.core)

        self.upsampler0 = B.upconv_block(nf, nf, 2)
        self.upsampler1 = B.upconv_block(nf, nf, 2)
        self.HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv1 = B.conv_block(nf, 3, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, x):
        x = self.fea_conv(x)
        if self.use_dense:
            x_out_list = [x]
            for i in range(self.na):
                x_out = self.core[i+i+1](self.core[i+i](x))
                x_out_list.append(x_out)
                x_cat = torch.cat(x_out_list, 1)
                x = self.convList[i](x_cat)
        
        else:
            x = self.core(x)
        x = self.upsampler0(x)
        x = self.upsampler1(x)
        x = self.HR_conv0(x)
        x = self.HR_conv1(x)
        return x

