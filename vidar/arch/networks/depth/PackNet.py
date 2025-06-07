# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import torch.nn as nn

from knk_vision.vidar.vidar.arch.networks.BaseNet import BaseNet
from knk_vision.vidar.vidar.arch.networks.layers.packnet import \
    PackLayerConv3d, UnpackLayerConv3d, Conv2D, ResidualBlock, InvDepth
from knk_vision.vidar.vidar.utils.depth import inv2depth


class PackNet(BaseNet, ABC):
    """
    PackNet depth network (https://arxiv.org/abs/1905.02693)
    
    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        # Configuration parameters
        self.min_depth = cfg.has('min_depth', 0.5)
        self.dropout = cfg.has('dropout', 0.0)

        # Input/output channels
        in_channels = 3
        out_channels = 1

        # Hyper-parameters
        ni, no = 64, out_channels
        n1, n2, n3, n4, n5 = 64, 64, 128, 256, 512
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]

        # Initial convolutional layer
        self.pre_calc = Conv2D(in_channels, ni, 5, 1)

        # Support for different versions
        n1o, n1i = n1, n1 + ni + no
        n2o, n2i = n2, n2 + n1 + no
        n3o, n3i = n3, n3 + n2 + no
        n4o, n4i = n4, n4 + n3
        n5o, n5i = n5, n5 + n4

        # Encoder

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0])
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1])
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2])
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3])
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4])

        self.conv1 = Conv2D(ni, n1, 7, 1)
        self.conv2 = ResidualBlock(n1, n2, num_blocks[0], 1, dropout=self.dropout)
        self.conv3 = ResidualBlock(n2, n3, num_blocks[1], 1, dropout=self.dropout)
        self.conv4 = ResidualBlock(n3, n4, num_blocks[2], 1, dropout=self.dropout)
        self.conv5 = ResidualBlock(n4, n5, num_blocks[3], 1, dropout=self.dropout)

        # Decoder

        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0])
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1])
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2])
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3])
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4])

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)

        # Depth Layers

        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.disp4_layer = InvDepth(n4, out_channels=out_channels, min_depth=self.min_depth)
        self.disp3_layer = InvDepth(n3, out_channels=out_channels, min_depth=self.min_depth)
        self.disp2_layer = InvDepth(n2, out_channels=out_channels, min_depth=self.min_depth)
        self.disp1_layer = InvDepth(n1, out_channels=out_channels, min_depth=self.min_depth)

        self.init_weights()

    def init_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, rgb, intrinsics=None):
        """Network forward pass"""

        # Initial convolution

        x = self.pre_calc(rgb)

        # Encoder

        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)

        # Skips

        skip1 = x
        skip2 = x1p
        skip3 = x2p
        skip4 = x3p
        skip5 = x4p

        # Decoder

        unpack5 = self.unpack5(x5p)
        concat5 = torch.cat((unpack5, skip5), 1)
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        concat4 = torch.cat((unpack4, skip4), 1)
        iconv4 = self.iconv4(concat4)
        inv_depth4 = self.disp4_layer(iconv4)
        up_inv_depth4 = self.unpack_disp4(inv_depth4)

        unpack3 = self.unpack3(iconv4)
        concat3 = torch.cat((unpack3, skip3, up_inv_depth4), 1)
        iconv3 = self.iconv3(concat3)
        inv_depth3 = self.disp3_layer(iconv3)
        up_inv_depth3 = self.unpack_disp3(inv_depth3)

        unpack2 = self.unpack2(iconv3)
        concat2 = torch.cat((unpack2, skip2, up_inv_depth3), 1)
        iconv2 = self.iconv2(concat2)
        inv_depth2 = self.disp2_layer(iconv2)
        up_inv_depth2 = self.unpack_disp2(inv_depth2)

        unpack1 = self.unpack1(iconv2)
        concat1 = torch.cat((unpack1, skip1, up_inv_depth2), 1)
        iconv1 = self.iconv1(concat1)
        inv_depth1 = self.disp1_layer(iconv1)

        if self.training:
            inv_depths = [inv_depth1, inv_depth2, inv_depth3, inv_depth4]
        else:
            inv_depths = [inv_depth1]

        return {
            'depths': inv2depth(inv_depths),
        }