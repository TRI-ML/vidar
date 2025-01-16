# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from knk_vision.vidar.vidar.utils.config import cfg_has
from knk_vision.vidar.vidar.utils.types import is_dict


class BaseModel(nn.Module):
    """Base Model class defines APIs for a vidar model."""
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
        """Processes a batch."""
        raise NotImplementedError(
            'Please implement forward function in your own subclass model.')

    def get_num_scales(self, scales):
        """Returns the number of scales to use for a given input."""
        while is_dict(scales):
            scales = list(scales.values())[0]
        return min(self.num_scales, len(scales))

    def compute_pose(self, rgb, net, tgt=0, ctx=None, invert=True):
        """Computes pose between a target and a context frame."""
        if ctx is None:
            ctx = [key for key in rgb.keys() if key != tgt]
        return {idx: net(
            [rgb[tgt], rgb[idx]], invert=(idx < tgt) and invert)['transformation']
                for idx in ctx}

    def set_attr(self, cfg, key, default):
        """Sets an attribute from a config file."""
        self.__setattr__(key, cfg_has(cfg, key, default))
