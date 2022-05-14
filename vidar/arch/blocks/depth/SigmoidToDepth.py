# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch.nn as nn

from vidar.utils.decorators import iterate2


class SigmoidToDepth(nn.Module, ABC):
    """
    Converts sigmoid to depth map

    Parameters
    ----------
    min_depth : Float
        Minimum depth value
    max_depth
        Maximum depth value
    """
    def __init__(self, min_depth, max_depth):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.diff_depth = (self.max_depth - self.min_depth)

    @iterate2
    def forward(self, sigmoid):
        """Convert sigmoid to depth"""
        return self.min_depth + self.diff_depth * sigmoid
