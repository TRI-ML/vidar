from abc import ABC

import torch.nn as nn

from knk_vision.vidar.vidar.arch.networks.layers.convolution import Conv3x3, upsample


class ConvDecoder(nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()

        self.num_ch_enc = cfg.num_ch_enc
        self.num_ch_dec = cfg.num_ch_dec
        self.num_ch_out = cfg.num_ch_out

        self.conv_out = nn.ModuleList([
            Conv3x3(self.num_ch_enc,    self.num_ch_out, kernel_size=1),
            Conv3x3(self.num_ch_dec[0], self.num_ch_out, kernel_size=1),
            Conv3x3(self.num_ch_dec[1], self.num_ch_out, kernel_size=1),
        ])

        self.conv0 = nn.ModuleList([
            Conv3x3(self.num_ch_enc,    self.num_ch_dec[0], kernel_size=3),
            Conv3x3(self.num_ch_dec[0], self.num_ch_dec[0], kernel_size=3),
        ])

        self.conv1 = nn.ModuleList([
            Conv3x3(self.num_ch_dec[0], self.num_ch_dec[1], kernel_size=3),
            Conv3x3(self.num_ch_dec[1], self.num_ch_dec[1], kernel_size=3),
        ])

    def forward(self, x):

        x = x.contiguous()

        out0_1 = self.conv0[0](x)
        out0_1_up = upsample(out0_1)
        out0_2 = self.conv0[1](out0_1_up)

        out1_1 = self.conv1[0](out0_2)
        out1_1_up = upsample(out1_1)
        out1_2 = self.conv1[1](out1_1_up)

        out0 = self.conv_out[0](x)
        out1 = self.conv_out[1](out0_2)
        out2 = self.conv_out[2](out1_2)

        return [out0, out1, out2]
