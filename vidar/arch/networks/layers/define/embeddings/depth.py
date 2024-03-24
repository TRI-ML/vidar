# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch

from knk_vision.vidar.vidar.arch.networks.layers.define.embeddings.base import BaseEmbeddings
from knk_vision.vidar.vidar.utils.nerf import get_camera_origin, get_camera_rays


class DepthEmbeddings(BaseEmbeddings):
    """
    Depth Embeddings class.

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

    @property
    def channels(self):
        """ Returns the number of channels."""
        return super().channels

    def create_embeddings(self, depth):
        """ Creates depth embeddings."""

        embeddings = []

        if self.fourier_encoding_depth is not None:
            depth_embeddings = self.fourier_encoding_depth(
                index_dims=None, pos=depth, batch_size=depth.shape[0], device=depth.device)
            embeddings.append(depth_embeddings)

        return embeddings

    def forward(self, depth):
        """ Forward pass for the gradient loss."""

        embeddings = self.create_embeddings(depth)
        embeddings = torch.cat(embeddings, -1)

        return embeddings
