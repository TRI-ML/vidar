# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from vidar.arch.blocks.depth.SigmoidToInvDepth import SigmoidToInvDepth
from vidar.utils.config import cfg_has


class BaseNet(nn.Module):
    """Base network class for all networks."""
    def __init__(self, cfg=None):
        super().__init__()
        self.networks = torch.nn.ModuleDict()
        self.blocks = torch.nn.ModuleDict()

        if cfg is not None and cfg.has('depth_range'):
            self.to_depth = SigmoidToInvDepth(
                cfg.depth_range[0], cfg.depth_range[1], return_depth=True)
        else:
            self.to_depth = None

    def _forward_unimplemented(self, *args):
        raise NotImplementedError('Forward unimplemented is unimplemented!')

    def set_attr(self, cfg, key, default):
        self.__setattr__(key, cfg_has(cfg, key, default))

    def train(self, mode=True):
        """Modified training mode to set all submodules to train mode."""
        super().train(mode=mode)
        for key, val in self.networks.items():
            val.train(mode=mode)
        for key, val in self.blocks.values():
            val.train(mode=mode)

    def eval(self):
        """Modified eval mode to set all submodules to eval mode (currently works as default)."""
        self.train(mode=False)

    def sigmoid_to_depth(self, sigmoids):
        """Converts sigmoids to depth maps."""
        return self.to_depth(sigmoids) if self.to_depth is not None else sigmoids

    def load(self, ckpt, name):
        """Loads a checkpoint onto the network"""
        state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
        updated_state_dict = {}
        for key, val in state_dict.items():
            idx = key.find(name)
            if idx > -1:
                updated_state_dict[key[idx + len(name) + 1:]] = val
        self.load_state_dict(updated_state_dict)

