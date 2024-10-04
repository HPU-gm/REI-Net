import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, bias = False, *args, **kwargs):
        super(ConvBN, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_chan, in_chan, kernel_size = ks, stride = stride,
                                        padding = padding, bias = bias, groups = in_chan)
        self.pointwise_conv = nn.Conv2d(in_chan, out_chan, kernel_size = 1, bias = True)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class SIEM(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(SIEM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )
        r_channel = int(in_channels/2)
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(r_channel, r_channel, (3, 1), 1, (1, 0), bias=False, groups=r_channel),
            nn.BatchNorm2d(r_channel),
            # nn.ReLU(True),
        )
        dr = 5
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(r_channel, r_channel, (3, 1), 1, (dr, 0), bias=False, dilation=dr,groups=r_channel),
            nn.BatchNorm2d(r_channel),
            # nn.ReLU(True),
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(r_channel, r_channel, (1, 3), 1, (0, 1), bias=False, groups=r_channel),
            nn.BatchNorm2d(r_channel),
            # nn.ReLU(True),
        )                            
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(r_channel, r_channel, (1, 3), 1, (0, dr), bias=False, dilation=dr,groups=r_channel),
            nn.BatchNorm2d(r_channel),
            # nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        # _, _, h, w = x.size()
        x1 = self.conv1(x)
        x1_1, x1_2 = torch.split(x1, int(x1.size(1)/2), dim=1)
        x2_1 = self.conv2_1(x1_1)
        x2_2 = self.conv2_2(x1_2)
        # x3 = x2_1 + x2_2
        x3_1 = self.conv3_1(x2_1 + x2_2)
        x3_2 = self.conv3_2(x2_2 + x2_1)
        x4 = self.conv4(torch.cat([x3_1, x3_2], dim=1))

        return F.relu_(x + x4)