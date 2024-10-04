import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBN, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_chan, in_chan, kernel_size = ks, stride = stride,
                                        padding = padding, bias = False, groups = in_chan)
        self.pointwise_conv = nn.Conv2d(in_chan, out_chan, kernel_size = 1, bias = True)
        self.bn = nn.BatchNorm2d(out_chan, momentum=0.01, eps=1e-3)
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


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        # SwishImplementation.apply(x)等价于SwishImplementation.forward(x)
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class BCSDEM(nn.Module):

    def __init__(self,
                 num_channels,
                 conv_channels,
                 first_time=False,
                 last_time=False,
                 epsilon=1e-4,
                 onnx_export=False,
                 attention=True,
                 use_p8=False):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BCSDEM, self).__init__()
        self.epsilon = epsilon
        self.last_time = last_time
        self.use_p8 = use_p8

        # Conv layers
        self.conv6_up = ConvBN(num_channels, num_channels)
        self.conv5_up = ConvBN(num_channels, num_channels)
        self.conv4_up = ConvBN(num_channels, num_channels)
        self.conv3_up = ConvBN(num_channels, num_channels)

        self.conv4_down = ConvBN(num_channels, num_channels)
        self.conv5_down = ConvBN(num_channels, num_channels)
        self.conv6_down = ConvBN(num_channels, num_channels)
        self.conv7_down = ConvBN(num_channels, num_channels)


        if use_p8:
            self.conv7_up = ConvBN(num_channels, num_channels)
            self.conv8_down = ConvBN(num_channels, num_channels)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.p5_downsample = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.p6_downsample = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.p7_downsample = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)


        if use_p8:
            self.p7_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.p8_downsample = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)


        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                nn.Conv2d(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                nn.Conv2d(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            )
            self.p6_to_p7 = nn.Sequential(
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            )

            if use_p8:
                self.p7_to_p8 = nn.Sequential(
                    nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
                )

            self.p4_down_channel_2 = nn.Sequential(
                nn.Conv2d(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                nn.Conv2d(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

    def forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            if self.use_p8:
                p8_in = self.p7_to_p8(p7_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            if self.use_p8:
                # P3_0, P4_0, P5_0, P6_0, P7_0 and P8_0
                p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            else:
                # P3_0, P4_0, P5_0, P6_0 and P7_0
                p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        if self.use_p8:
            # P8_0 to P8_2
            # Connections for P7_0 and P8_0 to P7_1 respectively
            p7_up = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)))

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)))

        else:
            # Connections for P4_0 and P5_1 to P4_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))


        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))
        
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        if self.last_time:
            if self.use_p8:
                # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
                p7_out = self.conv7_down(
                    self.swish(p7_in + p7_up + self.p7_downsample(p6_out)))

                # Connections for P8_0 and P7_2 to P8_2
                p8_out = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out)))

                p8_out = self.p7_upsample(p8_out)
                fpn_out = torch.cat([fpn_out, p7_out], dim=1)
                fpn_out = self.p6_upsample(fpn_out)
                fpn_out = torch.cat([fpn_out, p6_out], dim=1)
                fpn_out = self.p5_upsample(fpn_out)
                fpn_out = torch.cat([fpn_out, p5_out], dim=1)
                fpn_out = self.p4_upsample(fpn_out)
                fpn_out = torch.cat([fpn_out, p4_out], dim=1)
                fpn_out = self.p3_upsample(fpn_out)
                fpn_out = torch.cat([fpn_out, p3_out], dim=1)

                return fpn_out
            
            else:
                # Connections for P7_0 and P6_2 to P7_2
                p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

                fpn_out = self.p6_upsample(p7_out)
                fpn_out = torch.cat([fpn_out, p6_out], dim=1)
                fpn_out = self.p5_upsample(fpn_out)
                fpn_out = torch.cat([fpn_out, p5_out], dim=1)
                fpn_out = self.p4_upsample(fpn_out)
                fpn_out = torch.cat([fpn_out, p4_out], dim=1)
                fpn_out = self.p3_upsample(fpn_out)
                fpn_out = torch.cat([fpn_out, p3_out], dim=1)

                return fpn_out

        else:
            if self.use_p8:
                # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
                p7_out = self.conv7_down(
                    self.swish(p7_in + p7_up + self.p7_downsample(p6_out)))

                # Connections for P8_0 and P7_2 to P8_2
                p8_out = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out)))

                return p3_out, p4_out, p5_out, p6_out, p7_out, p8_out
            
            else:
                # Connections for P7_0 and P6_2 to P7_2
                p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

                return p3_out, p4_out, p5_out, p6_out, p7_out

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
