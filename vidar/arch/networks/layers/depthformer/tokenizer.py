# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
from torch import nn
from torchvision.models.densenet import _DenseBlock


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


class TransitionUp(nn.Module):
    """Transposed convolution for upsampling"""
    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        if scale == 2:
            self.convTrans = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=2, padding=0, bias=True)
        elif scale == 4:
            self.convTrans = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ConvTranspose2d(
                    in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=2, padding=0, bias=True)
            )

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class DoubleConv(nn.Module):
    """Helper class with two convolutional layers, plus BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Tokenizer(nn.Module):
    """
    Feature tokenization network
    Base on https://github.com/mli0603/stereo-transformer/blob/main/module/feat_extractor_tokenizer.py

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super(Tokenizer, self).__init__()

        block_config = [4, 4, 4, 4]
        backbone_feat_channel = [64, 128, 128]
        hidden_dim = cfg.channel_dim
        growth_rate = 4

        backbone_feat_channel.reverse()
        block_config.reverse()

        self.num_resolution = len(backbone_feat_channel)
        self.block_config = block_config
        self.growth_rate = growth_rate

        self.bottle_neck = _DenseBlock(
            block_config[0], backbone_feat_channel[0], 4, drop_rate=0.0, growth_rate=growth_rate)
        up = []
        dense_block = []
        prev_block_channels = growth_rate * block_config[0]
        for i in range(self.num_resolution):
            if i == self.num_resolution - 1:
                up.append(TransitionUp(prev_block_channels, hidden_dim, 4))
                dense_block.append(DoubleConv(hidden_dim + 3, hidden_dim))
            else:
                up.append(TransitionUp(prev_block_channels, prev_block_channels))
                cur_channels_count = prev_block_channels + backbone_feat_channel[i + 1]
                dense_block.append(
                    _DenseBlock(block_config[i + 1], cur_channels_count, 4, drop_rate=0.0, growth_rate=growth_rate))
                prev_block_channels = growth_rate * block_config[i + 1]

        self.up = nn.ModuleList(up)
        self.dense_block = nn.ModuleList(dense_block)

    def forward(self, features):
        """Network forward pass"""

        features.reverse()
        output = self.bottle_neck(features[0])
        output = output[:, -(self.block_config[0] * self.growth_rate):]
        for i in range(self.num_resolution):
            hs = self.up[i](output, features[i + 1])
            output = self.dense_block[i](hs)
            if i < self.num_resolution - 1:
                output = output[:, -(self.block_config[i + 1] * self.growth_rate):]
        return output
