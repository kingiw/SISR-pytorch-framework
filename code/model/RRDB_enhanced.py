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
        self.old_version = args.old
        

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

            if self.old_version:
                if self.na == 3:    # 3 densely connected attention modules
                    self.na1 = B.sequential(self.core[0], self.core[1])
                    self.conv1 = B.conv_block(nf*2, nf, kernel_size=1, norm_type= None, mode='CNA')
                    self.na2 = B.sequential(self.core[2], self.core[3])
                    self.conv2 = B.conv_block(nf*3, nf, kernel_size=1, norm_type=None, mode='CNA')
                    self.na3 = B.sequential(self.core[4], self.core[5])
                    self.conv3 = B.conv_block(nf*4, nf, kernel_size=1, norm_type=None, mode='CNA')
                elif self.na == 4: # 4 densely connected attention modules
                    self.na1 = B.sequential(self.core[0], self.core[1])
                    self.conv1 = B.conv_block(nf*2, nf, kernel_size=1, norm_type= None, mode='CNA')
                    self.na2 = B.sequential(self.core[2], self.core[3])
                    self.conv2 = B.conv_block(nf*3, nf, kernel_size=1, norm_type=None, mode='CNA')
                    self.na3 = B.sequential(self.core[4], self.core[5])
                    self.conv3 = B.conv_block(nf*4, nf, kernel_size=1, norm_type=None, mode='CNA')
                    self.na4 = B.sequential(self.core[6], self.core[7])
                    self.conv4 = B.conv_block(nf*5, nf, kernel_size=1, norm_type=None, mode='CNA')
                else:
                    raise NotImplementedError("...")
                self.core = None
            else:
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
            
            if self.old_version:
                if self.na == 3:
                    x1_out = self.na1(x)
                    x1_cat = torch.cat((x,x1_out), 1)
                    x2_in = self.conv1(x1_cat)
                    x2_out = self.na2(x)
                    x2_cat = torch.cat((x, x1_out, x2_out), 1)
                    x3_in = self.conv2(x2_cat)
                    x3_out = self.na3(x)
                    x3_cat = torch.cat((x, x1_out, x2_out, x3_out), 1)
                    x = self.conv3(x3_cat)
                elif self.na == 4:
                    x1_out = self.na1(x)
                    x1_cat = torch.cat((x,x1_out), 1)
                    x2_in = self.conv1(x1_cat)
                    x2_out = self.na2(x)
                    x2_cat = torch.cat((x, x1_out, x2_out), 1)
                    x3_in = self.conv2(x2_cat)
                    x3_out = self.na3(x)
                    x3_cat = torch.cat((x, x1_out, x2_out, x3_out), 1)
                    x4_in = self.conv3(x3_cat)
                    x4_out = self.na4(x4_in)
                    x4_cat = torch.cat((x, x1_out, x2_out, x3_out, x4_out), 1)
                    x = self.conv4(x4_cat)
            else:
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

