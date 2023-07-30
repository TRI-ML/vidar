# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn


def weights_init_xavier(m):
    """Xavier initialization"""
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


