# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch.nn as nn
import torch.nn.functional as F


def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest", align_corners=None)


class ConvBlock(nn.Module, ABC):
    """Layer to perform a convolution followed by ELU"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels, kernel_size=kernel_size)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module, ABC):
    """Layer to pad and convolve input"""
    def __init__(self, in_channels, out_channels, use_refl=True, kernel_size=3):
        super().__init__()
        if kernel_size == 3:
            if use_refl:
                self.pad = nn.ReflectionPad2d(1)
            else:
                self.pad = nn.ZeroPad2d(1)
        else:
            self.pad = nn.Identity()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
