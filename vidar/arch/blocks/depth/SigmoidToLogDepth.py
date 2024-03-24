# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import torch.nn as nn

from knk_vision.vidar.vidar.utils.decorators import iterate2


class SigmoidToLogDepth(nn.Module, ABC):
    """Converts sigmoids to an inverse depth map"""
    def __init__(self):
        super().__init__()

    @iterate2
    def forward(self, sigmoid):
        return torch.exp(sigmoid)
