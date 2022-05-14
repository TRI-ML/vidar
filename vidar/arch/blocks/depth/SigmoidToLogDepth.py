# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import torch.nn as nn

from vidar.utils.decorators import iterate2


class SigmoidToLogDepth(nn.Module, ABC):
    """
    Converts sigmoid to a log depth map
    """
    def __init__(self):
        super().__init__()

    @iterate2
    def forward(self, sigmoid):
        """Convert sigmoid to log depth"""
        return torch.exp(sigmoid)
