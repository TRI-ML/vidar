# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch

from vidar.arch.networks.layers.define.embeddings.base import BaseEmbeddings
from vidar.utils.nerf import get_camera_origin, get_camera_rays, apply_idx
from vidar.utils.augmentations import resize_depth_preserve


class CameraEmbeddings(BaseEmbeddings):
    """
    Camera embeddings class.

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

    def create_embeddings(self, origin, rays, depth=None):
        """ Creates camera embeddings."""

        embeddings = []

        if self.fourier_encoding_origin is not None:
            origin_embeddings = self.fourier_encoding_origin(
                index_dims=None, pos=origin, batch_size=origin.shape[0], device=origin.device)
            embeddings.append(origin_embeddings)
        if self.fourier_encoding_rays is not None:
            rays_embeddings = self.fourier_encoding_rays(
                index_dims=None, pos=rays, batch_size=rays.shape[0], device=rays.device)
            embeddings.append(rays_embeddings)
        if self.fourier_encoding_depth is not None and depth is not None:
            depth_embeddings = self.fourier_encoding_depth(
                index_dims=None, pos=depth, batch_size=depth.shape[0], device=depth.device)
            embeddings.append(depth_embeddings)

        return embeddings

    def forward(self, cam, key=None, coords=None, depth=None, grid=None, meta=None):
        """Forward pass: create and return embeddings."""
        if self.downsample != 1.0:
            cam = cam.scaled(1.0 / self.downsample)
        orig = get_camera_origin(cam, key, idx=coords)
        rays = get_camera_rays(cam, key, idx=coords, to_world=self.to_world)

        if depth is not None:
            depth = resize_depth_preserve(depth, cam.hw)
            b, c, h, w = depth.shape
            depth = depth.view(b, c, h * w).permute(0, 2, 1)
            depth = apply_idx(depth, idx=coords)

        embeddings = self.create_embeddings(orig, rays, depth=depth)
        embeddings = torch.cat(embeddings, -1)

        data = torch.cat([orig, rays], -1)
        if coords is None:
            data = data.permute(0, 2, 1).reshape(len(data), 6, *cam.hw)

        return data, embeddings
