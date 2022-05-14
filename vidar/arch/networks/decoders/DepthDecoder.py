# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from vidar.arch.networks.layers.convs import ConvBlock, Conv3x3, upsample
from vidar.utils.config import cfg_has


class DepthDecoder(nn.Module, ABC):
    """
    Depth decoder network

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()

        self.num_scales = cfg_has(cfg, 'num_scales', 4)
        self.use_skips = cfg.use_skips

        self.num_ch_enc = cfg.num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_ch_out = cfg.num_ch_out

        self.convs = OrderedDict()
        for i in range(self.num_scales, -1, -1):

            num_ch_in = self.num_ch_enc[-1] if i == self.num_scales else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = ConvBlock(
                num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 1)] = ConvBlock(
                num_ch_in, num_ch_out)

        for i in range(self.num_scales):
            self.convs[('outconv', i)] = Conv3x3(
                self.num_ch_dec[i], self.num_ch_out)

        self.decoder = nn.ModuleList(list(self.convs.values()))

        if cfg.activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif cfg.activation == 'identity':
            self.activation = nn.Identity()
        elif cfg.activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            raise ValueError('Invalid activation function {}'.format(cfg.activation))

    def forward(self, input_features):
        """Network forward pass"""

        outputs = {}

        x = input_features[-1]
        for i in range(self.num_scales, -1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            if i in range(self.num_scales):
                outputs[('features', i)] = x
                outputs[('output', i)] = self.activation(
                    self.convs[('outconv', i)](x))

        return outputs
