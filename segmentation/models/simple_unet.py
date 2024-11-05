import torch
from torch import nn


class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleConv, self).__init__()

        self.conv1 = self.contract_block(in_channels, 8, 7, 3)
        self.conv2 = self.contract_block(8, 16, 3, 1)
        self.conv3 = self.contract_block(16, 32, 3, 1)

        self.upconv3 = self.expand_block(32, 16, 3, 1)
        self.upconv2 = self.expand_block(16 * 2, 8, 3, 1)
        self.upconv1 = self.expand_block(8 * 2, out_channels, 3, 1)

    def forward(self, X):
        conv1 = self.conv1(X)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        return expand
