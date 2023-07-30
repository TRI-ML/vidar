# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
from vidar.utils.types import is_list
from vidar.utils.depth import get_depth_bins, get_depth_bins_volume


def get_delta(zvals, inf=1e10):
    """Get delta values from z-values"""
    delta = zvals[:, :, 1:] - zvals[:, :, :-1]
    delta_inf = inf * torch.ones_like(delta[:, :, :1])
    delta = torch.cat([delta, delta_inf], 2)
    return delta


def density2alpha(density, delta):
    """Convert density to alpha"""
    return 1.0 - torch.exp(- density * delta)


def alpha2transmittance(alpha, eps=1e-10):
    """Convert alpha to transmittance"""
    ones = torch.ones_like(alpha[:, :, :1])
    alpha_shifted = torch.cat([ones, 1.0 - alpha + eps], 2)
    return torch.cumprod(alpha_shifted, 2)[:, :, :-1]


def composite(rgb, densities, zvals):
    """Composite RGB and depth values"""
    delta = get_delta(zvals)
    alpha = density2alpha(densities, delta)
    transmittance = alpha2transmittance(alpha)
    weights = alpha * transmittance

    rgb = composite_weights(rgb, weights)
    depth = composite_weights(zvals, weights)

    return {
        'rgb': rgb,
        'depth': depth,
        'weights': weights,
    }


def composite_depth(densities, zvals):
    """Composite depth values"""
    delta = get_delta(zvals)
    alpha = density2alpha(densities, delta)
    transmittance = alpha2transmittance(alpha)
    weights = alpha * transmittance

    depth = composite_weights(zvals, weights)

    return {
        'depth': depth,
        'weights': weights,
    }


def composite_weights(data, weights):
    """Composite data with weights"""
    return torch.sum(weights * data, 2)


def apply_idx(data, idx):
    """Apply index and stack sampled data"""
    if idx is None:
        return data
    if idx.shape[1] == data.shape[1]:
        return data
    return torch.stack([data[i][idx[i]] for i in range(len(idx))], 0)


def get_camera_origin(cam, key, data=None, idx=None):
    """Get camera origin for each pixel"""
    if data is not None and 'orig' in data[key].keys():
        origin = data[key]['orig']
    else:
        origin = cam.get_origin(flatten=True)
        origin = apply_idx(origin, idx)
    return origin


def get_camera_rays(cam, key, data=None, idx=None, to_world=False):
    """Get camera rays for each pixel"""
    if data is not None and 'rays' in data[key].keys():
        rays = data[key]['rays']
    else:
        if to_world:
            rays = cam.get_viewdirs(normalize=True, flatten=True, to_world=True)
        else:
            rays = cam.no_translation().get_viewdirs(normalize=True, flatten=True, to_world=True)
        rays = apply_idx(rays, idx)
    return rays


def sample_coarse(rays, n_coarse, near, far, depth=None, mode='linear', perturb=True, after=False):
    """Sample rays for coarse model"""
    if depth is None:
        samples = get_depth_bins(
            mode, near, far, n_coarse, perturb=perturb, device=rays.device, shape=rays)
    else:
        if not after:
            samples = get_depth_bins_volume(
                mode, near, depth[..., 0], n_coarse, perturb=perturb)
        else:
            samples = get_depth_bins_volume(
                mode, depth[..., 0], far, n_coarse + 1, perturb=perturb)[..., 1:]
    return samples.unsqueeze(-1)


def sample_fine(rays, n_fine, near, far, prev_weights):
    """Sample rays for fine model"""

    # Get predicted weights
    weights = prev_weights[0].squeeze(2).detach()

    # Parse information
    b, device = rays.shape[0], rays.device
    n_coarse = weights.shape[1]

    # Reshape weights
    if weights.dim() == 4:
        weights = weights.reshape(b, n_coarse, -1).permute(0, 2, 1)

    # Parse more information
    b, n, d = weights.shape
    weights = weights.view(b * n, d)

    # Get accumulated weights
    weights = weights.detach() + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)

    # Importance sampling
    u = torch.rand(b * n, n_fine, dtype=torch.float32, device=device)
    idx = torch.searchsorted(cdf, u, right=True).float() - 1.0
    idx = torch.clamp_min(idx, 0.0)

    # Add index noise
    steps = (idx + (2 * torch.rand_like(idx) - 1)) / (n_coarse - 1)
    steps = steps.view(b, n, -1)

    # Scale between near and far
    samples = near + (far - near) * steps

    # Return samples
    return samples.unsqueeze(-1)


def sample_depth(rays, n_depth, near, far, depth, depth_noise):
    """Sample rays for depth model"""

    if is_list(depth_noise):
        depth_noise, depth_buffer = depth_noise
    else:
        depth_noise, depth_buffer = depth_noise, 0.0

    # Get predicted depth
    depth = depth.detach()

    # Parse information
    b = rays.shape[0]

    # Reshape weights
    if depth.dim() == 4:
        depth = depth.reshape(b, 1, -1).permute(0, 2, 1)

    # Produce samples
    samples = depth.repeat((1, 1, n_depth))
    if depth_noise > 0.0:
        samples += torch.randn_like(samples) * depth_noise
    elif depth_noise < 0.0:
        samples += torch.randn_like(samples).abs() * depth_noise - depth_buffer

    samples = samples.clamp(min=near, max=far)

    # Return samples
    return samples.unsqueeze(-1)
