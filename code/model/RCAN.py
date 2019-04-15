import torch
import torch.nn as nn
import model.basic_blocks as B
from .attention_module import Attention_Module

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_resblocks, conv=default_conv, n_feat=64, kernel_size=3, reduction=16, act=nn.ReLU(True), res_scale=1):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Upsampler_x4
class Upsampler_x4(nn.Module):
    """
    a x4 Upsampler based on https://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_c, out_c):
        super(Upsampler_x4, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.upsample1(x)
        x = nn.functional.relu(self.conv1(x))
        x = self.upsample2(x)
        x = nn.functional.relu(self.conv2(x))
        return x

class RCAN(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = args.b_n_resgroups
        n_resblocks = args.b_n_resblocks
        n_feats = args.b_n_feats
        kernel_size = 3
        reduction = 16
        scale = 1
        act = nn.ReLU(True)
        
        # define head module
        modules_head = [conv(3, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler_x4(n_feats, n_feats), 
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(inplace=True),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

class RCAN_enhanced(nn.Module):
    def __init__(self, args):
        super(RCAN_enhanced, self).__init__()

        na = args.b_na
        nf = args.b_n_feats
        ng = args.b_n_resgroups 
        nb = args.b_n_resblocks
        self.use_dense = args.b_dense_attention_modules
        self.na = na


        self.fea_conv = B.conv_block(3, nf, kernel_size=3, act_type=None)
        core = []
        for _ in range(na):
            blocks = [
                ResidualGroup(n_resblocks=nb) \
            for _ in range(ng)]
            blocks.append(B.conv_block(nf, nf, 3, act_type=None))
            trunk = B.sequential(B.ShortcutBlock(B.sequential(*blocks)))
            core.append(Attention_Module(nf, nf, trunk=trunk))
            core.append(B.ResidualBlock(nf, nf))

        if self.use_dense:
            if self.na == 3:    # 3 densely connected attention modules
                self.na1 = B.sequential(core[0], core[1])
                self.conv1 = B.conv_block(nf*2, nf, kernel_size=1, norm_type= None, mode='CNA')
                self.na2 = B.sequential(core[2], core[3])
                self.conv2 = B.conv_block(nf*3, nf, kernel_size=1, norm_type=None, mode='CNA')
                self.na3 = B.sequential(core[4], core[5])
                self.conv3 = B.conv_block(nf*4, nf, kernel_size=1, norm_type=None, mode='CNA')
            elif self.na == 4: # 4 densely connected attention modules
                self.na1 = B.sequential(core[0], core[1])
                self.conv1 = B.conv_block(nf*2, nf, kernel_size=1, norm_type= None, mode='CNA')
                self.na2 = B.sequential(core[2], core[3])
                self.conv2 = B.conv_block(nf*3, nf, kernel_size=1, norm_type=None, mode='CNA')
                self.na3 = B.sequential(core[4], core[5])
                self.conv3 = B.conv_block(nf*4, nf, kernel_size=1, norm_type=None, mode='CNA')
                self.na4 = B.sequential(core[6], core[7])
                self.conv4 = B.conv_block(nf*5, nf, kernel_size=1, norm_type=None, mode='CNA')
            else:
                raise NotImplementedError("...")
        else:
            self.core = B.sequential(*core)
        self.upsampler0 = B.upconv_block(nf, nf, 2)
        self.upsampler1 = B.upconv_block(nf, nf, 2)
        self.HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type='relu')
        self.HR_conv1 = B.conv_block(nf, 3, kernel_size=3, norm_type=None, act_type=None)
    
    def forward(self, x):
        x = self.fea_conv(x)
        if self.use_dense:  
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
            x = self.core(x)
        x = self.upsampler0(x)
        x = self.upsampler1(x)
        x = self.HR_conv0(x)
        x = self.HR_conv1(x)
        return x