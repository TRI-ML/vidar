# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from vidar.utils.config import cfg_has


class BaseModel(nn.Module):
    """Base model super class, that all other models inherit"""
    def __init__(self, cfg=None):
        super().__init__()

        self.blocks = torch.nn.ModuleDict()
        self.networks = torch.nn.ModuleDict()
        self.losses = torch.nn.ModuleDict()

        if cfg is not None:
            self.num_scales = cfg_has(cfg.model, 'num_scales', 99)

    def _forward_unimplemented(self, *args):
        pass

    def forward(self, *args, **kwargs):
        """Model forward pass"""
        raise NotImplementedError(
            'Please implement forward function in your own subclass model.')

    def get_num_scales(self, scales):
        """Return number of predicted scales"""
        return min(self.num_scales, len(scales))

    def compute_pose(self, rgb, net, tgt=0, ctx=None, invert=True):
        """Compute poses from pairs of images"""
        if ctx is None:
            ctx = [key for key in rgb.keys() if key != tgt]
        return {idx: net(
            [rgb[tgt], rgb[idx]], invert=(idx < tgt) and invert)['transformation']
                for idx in ctx}

    def set_attr(self, cfg, key, default):
        """Set an attribute for the model"""
        self.__setattr__(key, cfg_has(cfg, key, default))