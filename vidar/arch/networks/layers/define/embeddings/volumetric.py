# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
from einops import rearrange

from vidar.arch.networks.layers.define.embeddings.utils.fourier_position_encoding import \
    PerceiverFourierPositionEncoding
from vidar.utils.data import get_from_dict, get_from_list
from vidar.utils.nerf import sample_coarse, sample_fine, sample_depth, \
    apply_idx, get_camera_origin, get_camera_rays
from vidar.utils.types import is_list
from vidar.arch.networks.layers.define.embeddings.base import BaseEmbeddings


def flatten(data):
    if data.dim() == 4:
        return rearrange(data, 'b n k d -> b (n k) d')
    else:
        return data


class VolumetricEmbeddings(BaseEmbeddings):
    """
    Volumetric Embeddings class.

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.n_samples = cfg.sample.has('num', None)
        self.sample_range = cfg.sample.has('range', None)
        self.sample_mode = cfg.sample.has('mode', None)
        self.sample_type = cfg.sample.has('type', None)

        self.perturb = cfg.sample.has('perturb', True)
        self.depth_noise = cfg.sample.has('depth_noise', None)
        self.depth_guide = cfg.sample.has('depth_guide', None)
        self.use_prev_samples = cfg.sample.has('use_prev_samples', False)
        self.add_background = cfg.sample.has('add_background', None)

    @property
    def channels(self):
        """ Returns the number of channels."""
        return super().channels

    def sample(self, orig, rays, previous):
        """Return samples along rays given an origin."""
        if self.sample_type == 'coarse':
            near, far = self.sample_range
            if is_list(self.n_samples):
                zvals_coarse = sample_coarse(
                    rays, self.n_samples[0], near, far, depth=previous['depth'],
                    mode=self.sample_mode, perturb=self.perturb, after=False)
                zvals = [zvals_coarse]
                if len(self.n_samples) > 1 and self.n_samples[1] > 0:
                    zvals_depth = sample_depth(
                        rays, self.n_samples[1], near, far, depth=previous['depth'], depth_noise=self.depth_noise)
                    zvals.append(zvals_depth)
                if len(self.n_samples) > 2 and self.n_samples[2] > 0:
                    zvals_after = sample_coarse(
                        rays, self.n_samples[2], near, far, depth=previous['depth'],
                        mode=self.sample_mode, perturb=self.perturb, after=True)
                    zvals.append(zvals_after)
                zvals = torch.cat(zvals, -2)

                zvals_far = sample_coarse(
                    rays, sum(self.n_samples), near, far,
                    mode=self.sample_mode, perturb=self.perturb, after=False)
                same = zvals[:, :, -1, 0] == zvals[:, :, -2, 0]
                zvals[same] = zvals_far[same]

            else:
                zvals = sample_coarse(
                    rays, self.n_samples, near, far, depth=previous['depth'],
                    mode=self.sample_mode, perturb=self.perturb)
        elif self.sample_type == 'fine':
            near, far = self.sample_range
            if is_list(self.n_samples):
                zvals_fine = sample_fine(
                    rays, self.n_samples[0], near, far, previous['weights'])
                zvals_depth = sample_depth(
                    rays, self.n_samples[1], near, far, previous['depth'], self.depth_noise)
                zvals = torch.cat([zvals_fine, zvals_depth], -2)
            else:
                zvals = sample_fine(
                    rays, self.n_samples, near, far, previous['weights'])
        elif self.sample_type == 'depth':
            near, far = self.sample_range
            zvals = sample_depth(
                rays, self.n_samples, near, far, previous['depth'], self.depth_noise)
        else:
            zvals = None

        # Sort and clamp sampled depth values
        if self.sample_range is not None:
            zvals = zvals.clamp(min=self.sample_range[0], max=self.sample_range[1])

        if self.add_background is not None:
            last = self.add_background * rays - orig
            last = torch.linalg.norm(last, dim=2, keepdim=True).unsqueeze(2)
            zvals = torch.cat([zvals, last], 2)

        # Use previous samples if requested
        if self.use_prev_samples:
            zvals = previous['zvals'] if zvals is None else \
                torch.cat([zvals, previous['zvals']], -2)

        # Sort samples
        zvals = torch.sort(zvals, -2)[0]

        # Create 3D points from sampled depth values
        xyz = orig.unsqueeze(-2) + rays.unsqueeze(-2) * zvals

        # Return 3D points and sampled depth values
        return xyz, zvals

    def create_embeddings(self, xyz, orig, rays, zvals, expand=True):
        """Create embeddings for 3D points, origin, rays and depths."""

        b, n, k, _ = zvals.shape

        xyz = flatten(xyz)
        orig = flatten(orig)
        rays = flatten(rays)

        embeddings = []

        # Create 3D embeddings
        if self.fourier_encoding_xyz is not None:
            xyz_embeddings = self.fourier_encoding_xyz(
                index_dims=None, pos=xyz, batch_size=xyz.shape[0], device=xyz.device)
            xyz_embeddings = xyz_embeddings.view(b, n, k, -1)
            embeddings.append(xyz_embeddings)

        # Create origin embeddings
        if self.fourier_encoding_origin is not None:
            orig_embeddings = self.fourier_encoding_origin(
                index_dims=None, pos=orig, batch_size=orig.shape[0], device=orig.device)
            orig_embeddings = orig_embeddings.unsqueeze(2).repeat(1, 1, k, 1) if expand else \
                orig_embeddings.view(b, n, k, -1)
            embeddings.append(orig_embeddings)

        # Create rays embeddings
        if self.fourier_encoding_rays is not None:
            rays_embeddings = self.fourier_encoding_rays(
                index_dims=None, pos=rays, batch_size=rays.shape[0], device=rays.device)
            rays_embeddings = rays_embeddings.unsqueeze(2).repeat(1, 1, k, 1) if expand else \
                rays_embeddings.view(b, n, k, -1)
            embeddings.append(rays_embeddings)

        # Create depth embeddings
        if self.fourier_encoding_depth is not None:
            depth = zvals.view(zvals.shape[0], n * k, 1)
            depth_embeddings = self.fourier_encoding_depth(
                index_dims=None, pos=depth, batch_size=depth.shape[0], device=depth.device)
            depth_embeddings = depth_embeddings.view(b, n, k, -1)
            embeddings.append(depth_embeddings)

        return embeddings

    def get_previous(self, key, data, idx, previous):
        """Get previous predictions if requested. """
        weights = get_from_dict(previous, 'weights', key)
        zvals = get_from_dict(previous, 'info', key)
        zvals = get_from_dict(zvals, 'zvals')

        if self.depth_guide == 'gt':
            depth = rearrange(data[key]['gt']['depth'], 'b c h w -> b (h w) c')
            depth = apply_idx(depth, idx)
        elif self.depth_guide == 'prev':
            depth = get_from_list(get_from_dict(previous, 'depth', key))
        else:
            depth = None

        return {
            'depth': depth,
            'weights': weights,
            'zvals': zvals,
        }

    def get_samples(self, origin, rays, key, data, idx, previous):
        """Get samples from data dict."""
        xyz = get_from_dict(data[key], 'xyz')
        zvals = get_from_dict(data[key], 'zvals')
        if xyz is not None and zvals is not None:
            pass
        elif self.sample_mode is None:
            xyz = get_from_dict(previous['info'], key, 'xyz')
            zvals = get_from_dict(previous['info'], key, 'zvals')
        else:
            previous = self.get_previous(key, data, idx, previous)
            xyz, zvals = self.sample(origin, rays, previous=previous)
        return xyz, zvals

    def forward(self, cam, key, data, coords=None, previous=None):
        """Forward pass: create and return embeddings."""
        orig = get_camera_origin(cam, key, data, idx=coords)
        rays = get_camera_rays(cam, key, data, idx=coords, to_world=self.to_world)
        xyz, zvals = self.get_samples(orig, rays, key, data, coords, previous)

        embeddings = self.create_embeddings(xyz, orig, rays, zvals)
        embeddings = torch.cat(embeddings, -1)

        data = torch.cat([orig, rays], -1)
        if coords is None:
            data = data.permute(0, 2, 1).reshape(len(data), 6, *cam.hw)

        return data, embeddings, {'xyz': xyz, 'zvals': zvals}

