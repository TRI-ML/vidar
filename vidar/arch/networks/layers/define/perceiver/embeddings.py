# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
from knk_vision.vidar.vidar.utils.networks import freeze_layers_and_norms


class PerceiverEmbeddings(nn.Module):
    """Class to create and store the Perceiver IO latent space"""
    def __init__(self, cfg):
        super().__init__()
        self.shape = (cfg.num, cfg.dim)
        self.latents = nn.Parameter(torch.randn(self.shape))
        self.freeze = cfg.has('freeze', False)

    def forward(self, batch_size, scene=None):
        """Forward pass to retrieve the Perceiver IO latent space"""
        if self.training:
            self.latents.requires_grad_(not self.freeze)
        return self.latents.expand(batch_size, -1, -1)
