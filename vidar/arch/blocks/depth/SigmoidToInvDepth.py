# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch.nn as nn

from vidar.utils.decorators import iterate2
from vidar.utils.depth import inv2depth


class SigmoidToInvDepth(nn.Module, ABC):
    """
    Converts sigmoid to inverse depth map

    Parameters
    ----------
    min_depth : Float
        Minimum depth value
    max_depth
        Maximum depth value
    return_depth:
        Whether the inverse depth map is inverted to depth when returning
    """
    def __init__(self, min_depth, max_depth, return_depth=False):
        super().__init__()
        self.min_inv_depth = 1. / max_depth
        self.max_inv_depth = 1. / min_depth
        self.diff_inv_depth = (self.max_inv_depth - self.min_inv_depth)
        self.return_depth = return_depth

    @iterate2
    def forward(self, sigmoid):
        """Convert sigmoid to inverse depth"""
        inv_depth = self.min_inv_depth + self.diff_inv_depth * sigmoid
        return inv_depth if not self.return_depth else inv2depth(inv_depth)
