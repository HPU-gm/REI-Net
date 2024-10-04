#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import os.path as osp
import sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

main_path = osp.join(this_dir, '..')
add_path(main_path)

from .bcsdem import BCSDEM
from .siem import SIEM

from .resnet import Resnet18

from .stdcnet import STDCNet1446, STDCNet813
stdc813_pth = "path/to/STDCNet813M_73.91.tar"
stdc1446_pth = "path/to/STDCNet1446_76.47.tar"

from torch.nn import BatchNorm2d


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor,
                mode='bilinear', align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        # -------------------- Backbone --------------------
        self.resnet = Resnet18()
        channels = [128, 256, 512]
        # self.stdc = STDCNet1446(pretrain_model = stdc1446_pth)
        # channels = [512, 1024]
        # -------------------- Backbone --------------------

        self.bcsdem = nn.Sequential(
            *[BCSDEM(num_channels=64,
                    conv_channels=channels,
                    first_time=True if _ == 0 else False,
                    last_time=False if _  < 3 - 1 else True,
                    attention=True if _ == 0 else False,
                    )
              for _ in range(3)])

        self.catConv = ConvBNReLU(320, 128, ks=3, stride=1, padding=1)

        self.siem = SIEM(channels[0], (20, 12), nn.BatchNorm2d, up_kwargs={'mode': 'bilinear', 'align_corners': True})

        self.init_weight()

    def forward(self, x):
        # -------------------- Backbone --------------------
        feat8, feat16, feat32 = self.resnet(x)
        # _, _, feat8, feat16, feat32 = self.stdc(x)      # _, _, 256, 512, 1024
        # -------------------- Backbone --------------------

        # ------------------BCSDEM-----------------------
        # 输入后三层特征图
        self.featureMap1 = feat16
        feat = self.bcsdem((feat8, feat16, feat32))
        feat16_up = self.catConv(feat)
        self.featureMap2 = feat16_up
        # ------------------BCSDEM-----------------------

        # # -----------------SIEM------------------
        feat8 = self.siem(feat8)
        # # -----------------SIEM------------------

        return feat8, feat16_up

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class IIM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(IIM, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def forward(self, fsp, fcp):
        out = torch.cat([fsp, fcp], dim=1)
        self.featureMap5 = out
        atten = self.pool(out)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = atten.sigmoid()
        out1 = out * atten
        out1 = self.conv1(out1)
        out1 = self.bn1(out1)
        return F.relu(out1 + out)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class REI_Net(nn.Module):

    def __init__(self, n_classes, aux_mode='train', *args, **kwargs):
        super(REI_Net, self).__init__()
        self.cp = ContextPath()
        self.iim = IIM(128, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)
        self.aux_mode = aux_mode
        if self.aux_mode == 'train':
            self.conv_out16 = BiSeNetOutput(128, 64, n_classes, up_factor=8)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_sp, feat_cp8 = self.cp(x)
        feat_fuse = self.iim(feat_sp, feat_cp8)
        self.featureMap6 = feat_fuse

        feat_out = self.conv_out(feat_fuse)
        if self.aux_mode == 'train':
            feat_out16 = self.conv_out16(feat_cp8)
            return feat_out, feat_out16
        elif self.aux_mode == 'eval':
            return feat_out,
        elif self.aux_mode == 'pred':
            feat_out = feat_out.argmax(dim=1)
            return feat_out
        else:
            raise NotImplementedError

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (IIM, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

if __name__ == "__main__":
    m_device = try_gpu(0)
    net = BiSeNetV1(19)
    net.to(device=m_device)
    net.eval()
    in_ten = torch.randn(2, 3, 1024, 1024).to(device=m_device)
    out, out16, out32 = net(in_ten)
    print(out.shape)
    print(out16.shape)
    net.get_params()