# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch

from vidar.arch.networks.BaseNet import BaseNet
from vidar.arch.networks.layers.depthformer.transformer_net import TransformerNet
from vidar.utils.config import cfg_has


class MatchModule(BaseNet, ABC):
    """
    Feature matching module (https://arxiv.org/abs/2204.07616)

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.downsample = cfg_has(cfg, 'downsample', 3)
        self.set_attr(cfg, 'monocular', False)
        self.set_attr(cfg, 'decoder_type', 'regression')
        self.set_attr(cfg, 'pos_enc', True)
        self.set_attr(cfg, 'calc_right', True)

        self.match_module = TransformerNet(cfg, decoder_type=self.decoder_type)
        if cfg_has(cfg, 'fix_layers', True):
            self.match_module.fix_layers()
        self.set_attr(cfg, 'preprocess', False)

    def forward(self, target, context, device, cam):
        """Network forward pass"""

        bs, _, h, w = target.size()

        downsample = 4
        col_offset = int(downsample / 2)
        row_offset = int(downsample / 2)
        sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).to(device)
        sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).to(device)

        return self.match_module(target, context, sampled_rows, sampled_cols, cam)
