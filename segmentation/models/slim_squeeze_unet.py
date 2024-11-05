from torch.nn import MaxPool2d, Dropout

from segmentation import settings
from segmentation.models.slimmable_ops import (
    SlimmableConv2d,
    SwitchableBatchNorm2d,
    SlimmableConvTranspose2d,
)
import torch
from torch import nn

kernel_size = 3
ch_in = [3, 3, 3, 3]
channels = [[3, 3, 3, 3]]


class SlimSqueezeUNet(nn.Module):
    def __init__(self, out_channels, device="cuda:0"):
        super(SlimSqueezeUNet, self).__init__()
        self.device = device

        self.conv1 = SlimmableConv2d(
            [3, 3, 3, 3], [8, 16, 32, 64], kernel_size, 2, padding=1
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #
        self.fire01 = FireModule([8, 16, 32, 64], [2, 4, 8, 16], [8, 16, 32, 64])
        self.fire02 = FireModule([16, 32, 64, 128], [2, 4, 8, 16], [8, 16, 32, 64])
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)

        self.fire03 = FireModule([16, 32, 64, 128], [4, 8, 16, 32], [16, 32, 64, 128])
        self.fire04 = FireModule([32, 64, 128, 256], [4, 8, 16, 32], [16, 32, 64, 128])
        self.maxpool3 = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)

        self.fire05 = FireModule([32, 64, 128, 256], [6, 12, 24, 48], [24, 48, 96, 192])
        self.fire06 = FireModule([48, 96, 192, 384], [6, 12, 24, 48], [24, 48, 96, 192])
        self.fire07 = FireModule(
            [48, 96, 192, 384], [8, 16, 32, 64], [32, 64, 128, 256]
        )
        self.fire08 = FireModule(
            [64, 128, 256, 512], [8, 16, 32, 64], [32, 64, 128, 256]
        )

        if settings.DROPOUT:
            self.dropout = Dropout(settings.DROPOUT)
        else:
            self.dropout = None

        self.conv2 = SlimmableConvTranspose2d(
            [64, 128, 256, 512],
            [24, 48, 96, 192],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
        )
        self.fire09 = FireModule(
            [72, 144, 288, 576], [6, 12, 24, 48], [24, 48, 96, 192]
        )

        self.conv3 = SlimmableConvTranspose2d(
            [48, 96, 192, 384],
            [16, 32, 64, 128],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
        )
        self.fire10 = FireModule([48, 96, 192, 384], [4, 8, 16, 32], [16, 32, 64, 128])

        self.conv4 = SlimmableConvTranspose2d(
            [32, 64, 128, 256],
            [8, 16, 32, 64],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
            stride=2,
            output_padding=1,
        )
        self.fire11 = FireModule([24, 48, 96, 192], [2, 4, 8, 16], [8, 16, 32, 64])

        self.conv5 = SlimmableConvTranspose2d(
            [16, 32, 64, 128],
            [4, 8, 16, 32],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
            stride=2,
            output_padding=1,
        )
        self.fire12 = FireModule([12, 24, 48, 96], [2, 4, 8, 16], [4, 8, 16, 32])

        self.conv6 = SlimmableConv2d(
            [16, 32, 64, 128], [8, 16, 32, 64], kernel_size, 1, padding=1
        )
        self.conv7 = SlimmableConv2d(
            [8, 16, 32, 64],
            [out_channels, out_channels, out_channels, out_channels],
            1,
            1,
            padding=0,
        )

        self.layers = [
            self.conv1,
            self.maxpool1,
            self.fire01,
            self.fire02,
            self.maxpool2,
            self.fire03,
            self.fire04,
            self.maxpool3,
            self.fire05,
            self.fire06,
            self.fire07,
            self.fire08,
            self.conv2,
            self.fire09,
            self.conv3,
            self.fire10,
            self.conv4,
            self.fire11,
            self.conv5,
            self.fire12,
            self.conv6,
            self.conv7,
        ]

    def forward(self, X):
        x01 = self.conv1(X)
        x02 = self.maxpool1(x01)

        x03 = self.fire01(x02)
        x04 = self.fire02(x03)
        x05 = self.maxpool2(x04)

        x06 = self.fire03(x05)
        x07 = self.fire04(x06)
        x08 = self.maxpool3(x07)

        x09 = self.fire05(x08)
        x10 = self.fire06(x09)
        x11 = self.fire07(x10)
        x12 = self.fire08(x11)

        # if self.dropout:
        #     x12 = self.dropout(x12)

        a01 = self.conv2(x12)
        y01 = torch.cat((a01, x10), dim=1)
        y02 = self.fire09(y01)

        a02 = self.conv3(y02)
        y03 = torch.cat((a02, x08), dim=1)
        y04 = self.fire10(y03)

        a03 = self.conv4(y04)
        y05 = torch.cat((a03, x05), dim=1)
        y06 = self.fire11(y05)

        a04 = self.conv5(y06)
        y07 = torch.cat((a04, x02), dim=1)
        y08 = self.fire12(y07)

        y09 = nn.Upsample(scale_factor=2)(y08)
        y10 = torch.cat((y09, x01), dim=1)
        y11 = self.conv6(y10)

        y12 = nn.Upsample(scale_factor=2)(y11)
        y13 = self.conv7(y12)

        return y13

    def set_width(self, width):
        for layer in self.layers:
            # This is to call custom module (i.e. FireModule)
            try:
                layer.set_width(width)
            except AttributeError:
                pass
            # This is to set width on a list of layers or a layer directly
            try:
                for x in layer:
                    x.width_mult = width
            except TypeError:
                layer.width_mult = width


class SlimSqueezeUNetCofly(nn.Module):
    def __init__(self, out_channels, device="cuda:0"):
        super(SlimSqueezeUNetCofly, self).__init__()
        self.device = device
        enlarge = 1

        self.conv1 = SlimmableConv2d(
            [3, 3, 3, 3],
            [8 * enlarge, 16 * enlarge, 32 * enlarge, 64 * enlarge],
            kernel_size,
            2,
            padding=1,
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #
        self.fire01 = FireModule(
            [8 * enlarge, 16 * enlarge, 32 * enlarge, 64 * enlarge],
            [2 * enlarge, 4 * enlarge, 8 * enlarge, 16 * enlarge],
            [8 * enlarge, 16 * enlarge, 32 * enlarge, 64 * enlarge],
        )
        self.fire02 = FireModule(
            [16 * enlarge, 32 * enlarge, 64 * enlarge, 128 * enlarge],
            [2 * enlarge, 4 * enlarge, 8 * enlarge, 16 * enlarge],
            [8 * enlarge, 16 * enlarge, 32 * enlarge, 64 * enlarge],
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)

        self.fire03 = FireModule(
            [16 * enlarge, 32 * enlarge, 64 * enlarge, 128 * enlarge],
            [4 * enlarge, 8 * enlarge, 16 * enlarge, 32 * enlarge],
            [16 * enlarge, 32 * enlarge, 64 * enlarge, 128 * enlarge],
        )
        self.fire04 = FireModule(
            [32 * enlarge, 64 * enlarge, 128 * enlarge, 256 * enlarge],
            [4 * enlarge, 8 * enlarge, 16 * enlarge, 32 * enlarge],
            [16 * enlarge, 32 * enlarge, 64 * enlarge, 128 * enlarge],
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)

        self.fire05 = FireModule(
            [32 * enlarge, 64 * enlarge, 128 * enlarge, 256 * enlarge],
            [6 * enlarge, 12 * enlarge, 24 * enlarge, 48 * enlarge],
            [24 * enlarge, 48 * enlarge, 96 * enlarge, 192 * enlarge],
        )
        self.fire06 = FireModule(
            [48 * enlarge, 96 * enlarge, 192 * enlarge, 384 * enlarge],
            [6 * enlarge, 12 * enlarge, 24 * enlarge, 48 * enlarge],
            [24 * enlarge, 48 * enlarge, 96 * enlarge, 192 * enlarge],
        )
        self.fire07 = FireModule(
            [48 * enlarge, 96 * enlarge, 192 * enlarge, 384 * enlarge],
            [8 * enlarge, 16 * enlarge, 32 * enlarge, 64 * enlarge],
            [32 * enlarge, 64 * enlarge, 128 * enlarge, 256 * enlarge],
        )
        self.fire08 = FireModule(
            [64 * enlarge, 128 * enlarge, 256 * enlarge, 512 * enlarge],
            [8 * enlarge, 16 * enlarge, 32 * enlarge, 64 * enlarge],
            [32 * enlarge, 64 * enlarge, 128 * enlarge, 256 * enlarge],
        )

        if settings.DROPOUT:
            self.dropout = Dropout(settings.DROPOUT)
        else:
            self.dropout = None

        self.conv2 = SlimmableConvTranspose2d(
            [64 * enlarge, 128 * enlarge, 256 * enlarge, 512 * enlarge],
            [24 * enlarge, 48 * enlarge, 96 * enlarge, 192 * enlarge],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
        )
        self.fire09 = FireModule(
            [72 * enlarge, 144 * enlarge, 288 * enlarge, 576 * enlarge],
            [6 * enlarge, 12 * enlarge, 24 * enlarge, 48 * enlarge],
            [24 * enlarge, 48 * enlarge, 96 * enlarge, 192 * enlarge],
        )

        self.conv3 = SlimmableConvTranspose2d(
            [48 * enlarge, 96 * enlarge, 192 * enlarge, 384 * enlarge],
            [16 * enlarge, 32 * enlarge, 64 * enlarge, 128 * enlarge],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
        )
        self.fire10 = FireModule(
            [48 * enlarge, 96 * enlarge, 192 * enlarge, 384 * enlarge],
            [4 * enlarge, 8 * enlarge, 16 * enlarge, 32 * enlarge],
            [16 * enlarge, 32 * enlarge, 64 * enlarge, 128 * enlarge],
        )

        self.conv4 = SlimmableConvTranspose2d(
            [32 * enlarge, 64 * enlarge, 128 * enlarge, 256 * enlarge],
            [8 * enlarge, 16 * enlarge, 32 * enlarge, 64 * enlarge],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
            stride=2,
            output_padding=1,
        )
        self.fire11 = FireModule(
            [24 * enlarge, 48 * enlarge, 96 * enlarge, 192 * enlarge],
            [2 * enlarge, 4 * enlarge, 8 * enlarge, 16 * enlarge],
            [8 * enlarge, 16 * enlarge, 32 * enlarge, 64 * enlarge],
        )

        self.conv5 = SlimmableConvTranspose2d(
            [16 * enlarge, 32 * enlarge, 64 * enlarge, 128 * enlarge],
            [4 * enlarge, 8 * enlarge, 16 * enlarge, 32 * enlarge],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
            stride=2,
            output_padding=1,
        )
        self.fire12 = FireModule(
            [12 * enlarge, 24 * enlarge, 48 * enlarge, 96 * enlarge],
            [2 * enlarge, 4 * enlarge, 8 * enlarge, 16 * enlarge],
            [4 * enlarge, 8 * enlarge, 16 * enlarge, 32 * enlarge],
        )

        self.conv6 = SlimmableConv2d(
            [16 * enlarge, 32 * enlarge, 64 * enlarge, 128 * enlarge],
            [8 * enlarge, 16 * enlarge, 32 * enlarge, 64 * enlarge],
            kernel_size,
            1,
            padding=1,
        )
        self.conv7 = SlimmableConv2d(
            [8 * enlarge, 16 * enlarge, 32 * enlarge, 64 * enlarge],
            [out_channels, out_channels, out_channels, out_channels],
            1,
            1,
            padding=0,
        )

        self.layers = [
            self.conv1,
            self.maxpool1,
            self.fire01,
            self.fire02,
            self.maxpool2,
            self.fire03,
            self.fire04,
            self.maxpool3,
            self.fire05,
            self.fire06,
            self.fire07,
            self.fire08,
            self.conv2,
            self.fire09,
            self.conv3,
            self.fire10,
            self.conv4,
            self.fire11,
            self.conv5,
            self.fire12,
            self.conv6,
            self.conv7,
        ]

    def forward(self, X):
        x01 = self.conv1(X)
        x02 = self.maxpool1(x01)

        x03 = self.fire01(x02)
        x04 = self.fire02(x03)
        x05 = self.maxpool2(x04)

        x06 = self.fire03(x05)
        x07 = self.fire04(x06)
        x08 = self.maxpool3(x07)

        x09 = self.fire05(x08)
        x10 = self.fire06(x09)
        x11 = self.fire07(x10)
        x12 = self.fire08(x11)

        # if self.dropout:
        #     x12 = self.dropout(x12)

        a01 = self.conv2(x12)
        y01 = torch.cat((a01, x10), dim=1)
        y02 = self.fire09(y01)

        a02 = self.conv3(y02)
        y03 = torch.cat((a02, x08), dim=1)
        y04 = self.fire10(y03)

        a03 = self.conv4(y04)
        y05 = torch.cat((a03, x05), dim=1)
        y06 = self.fire11(y05)

        a04 = self.conv5(y06)
        y07 = torch.cat((a04, x02), dim=1)
        y08 = self.fire12(y07)

        y09 = nn.Upsample(scale_factor=2)(y08)
        y10 = torch.cat((y09, x01), dim=1)
        y11 = self.conv6(y10)

        y12 = nn.Upsample(scale_factor=2)(y11)
        y13 = self.conv7(y12)

        return y13

    def set_width(self, width):
        for layer in self.layers:
            # This is to call custom module (i.e. FireModule)
            try:
                layer.set_width(width)
            except AttributeError:
                pass
            # This is to set width on a list of layers or a layer directly
            try:
                for x in layer:
                    x.width_mult = width
            except TypeError:
                layer.width_mult = width


class SlimPrunedSqueezeUNet(nn.Module):
    def __init__(self, out_channels, device="cuda:0", dropout=settings.DROPOUT):
        super(SlimPrunedSqueezeUNet, self).__init__()
        self.device = device

        self.conv1 = SlimmableConv2d(
            [3, 3, 3, 3], [16, 32, 64, 128], kernel_size, 2, padding=1
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)

        self.fire01 = FireModule([8, 16, 32, 64], [2, 4, 8, 16], [8, 16, 32, 64])
        self.fire02 = FireModule([16, 32, 64, 128], [2, 4, 8, 16], [8, 16, 32, 64])
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)

        self.fire03 = FireModule([16, 32, 64, 128], [4, 8, 16, 32], [16, 32, 64, 128])
        self.fire04 = FireModule([32, 64, 128, 256], [4, 8, 16, 32], [16, 32, 64, 128])
        self.maxpool3 = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=1)

        self.fire05 = FireModule([32, 64, 128, 256], [6, 12, 24, 48], [24, 48, 96, 192])
        self.fire06 = FireModule([48, 96, 192, 384], [6, 12, 24, 48], [24, 48, 96, 192])
        self.fire07 = FireModule(
            [48, 96, 192, 384], [8, 16, 32, 64], [32, 64, 128, 256]
        )
        self.fire08 = FireModule(
            [64, 128, 256, 512], [8, 16, 32, 64], [32, 64, 128, 256]
        )

        if dropout:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None

        self.conv2 = SlimmableConvTranspose2d(
            [64, 128, 256, 512],
            [24, 48, 96, 192],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
        )
        self.fire09 = FireModule(
            [72, 144, 288, 576], [6, 12, 24, 48], [24, 48, 96, 192]
        )

        self.conv3 = SlimmableConvTranspose2d(
            [48, 96, 192, 384],
            [16, 32, 64, 128],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
        )
        self.fire10 = FireModule([48, 96, 192, 384], [4, 8, 16, 32], [8, 16, 32, 64])

        self.conv4 = SlimmableConvTranspose2d(
            [32, 64, 128, 256],
            [8, 16, 32, 64],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
            stride=2,
            output_padding=1,
        )
        self.fire11 = FireModule([24, 48, 96, 192], [2, 4, 8, 16], [8, 16, 32, 64])

        self.conv5 = SlimmableConvTranspose2d(
            [16, 32, 64, 128],
            [4, 8, 16, 32],
            kernel_size=kernel_size,
            dilation=1,
            padding=1,
            stride=2,
            output_padding=1,
        )
        self.fire12 = FireModule([20, 40, 80, 160], [2, 4, 8, 16], [4, 8, 16, 32])

        self.conv6 = SlimmableConv2d(
            [24, 48, 96, 192], [8, 16, 32, 64], kernel_size, 1, padding=1
        )
        self.conv7 = SlimmableConv2d(
            [8, 16, 32, 64],
            [out_channels, out_channels, out_channels, out_channels],
            1,
            1,
            padding=0,
        )

        self.layers = [
            self.conv1,
            self.maxpool1,
            self.fire01,
            self.fire02,
            self.maxpool2,
            self.fire03,
            self.fire04,
            self.maxpool3,
            self.fire05,
            self.fire06,
            self.fire07,
            self.fire08,
            self.conv2,
            self.fire09,
            self.conv3,
            self.fire10,
            self.conv4,
            self.fire11,
            self.conv5,
            self.fire12,
            self.conv6,
            self.conv7,
        ]

    def forward(self, X):
        x01 = self.conv1(X)
        x02 = self.maxpool1(x01)

        x03 = self.fire01(x02)
        x04 = self.fire02(x03)
        x05 = self.maxpool2(x04)

        x06 = self.fire03(x05)
        x07 = self.fire04(x06)
        x08 = self.maxpool3(x07)

        x09 = self.fire05(x08)
        x10 = self.fire06(x09)
        x11 = self.fire07(x10)
        x12 = self.fire08(x11)

        if self.dropout:
            x12 = self.dropout(x12)

        a01 = self.conv2(x12)
        y01 = torch.cat((a01, x10), dim=1)
        y02 = self.fire09(y01)

        a02 = self.conv3(y02)
        y03 = torch.cat((a02, x08), dim=1)
        y04 = self.fire10(y03)

        a03 = self.conv4(y04)
        y05 = torch.cat((a03, x05), dim=1)
        y06 = self.fire11(y05)

        a04 = self.conv5(y06)
        y07 = torch.cat((a04, x02), dim=1)
        y08 = self.fire12(y07)

        y09 = nn.Upsample(scale_factor=4)(y08)
        y10 = torch.cat((y09, x01), dim=1)
        y11 = self.conv6(y10)

        y12 = nn.Upsample(scale_factor=2)(y11)
        y13 = self.conv7(y12)

        return y13

    def set_width(self, width):
        for layer in self.layers:
            # This is to call custom module (i.e. FireModule)
            try:
                layer.set_width(width)
            except AttributeError:
                pass
            # This is to set width on a list of layers or a layer directly
            try:
                for x in layer:
                    x.width_mult = width
            except TypeError:
                layer.width_mult = width


class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        self.conv1 = SlimmableConv2d(in_channels, squeeze_channels, 1, padding="same")
        self.bn1 = SwitchableBatchNorm2d(squeeze_channels)
        self.conv_left = SlimmableConv2d(
            squeeze_channels, expand_channels, 1, padding="same"
        )
        self.conv_right = SlimmableConv2d(
            squeeze_channels, expand_channels, 3, padding="same"
        )
        self.layers = [self.conv1, self.bn1, self.conv_left, self.conv_right]

    def set_width(self, width):
        for layer in self.layers:
            try:
                for x in layer:
                    x.width_mult = width
            except TypeError:
                layer.width_mult = width

    def forward(self, X):
        X1 = self.conv1(X)
        X2 = self.bn1(X1)
        left = self.conv_left(X2)
        right = self.conv_right(X2)
        X3 = torch.cat((left, right), 1)
        return X3
