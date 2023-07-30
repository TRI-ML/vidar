# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch.nn as nn


class PerceiverSelfOutput(nn.Module):
    """Simple class wrapping up a linear layer."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense = nn.Linear(in_channels, out_channels)

    def forward(self, hidden_state):
        """Forward pass of the PerceiverSelfOutput"""
        return self.dense(hidden_state)
