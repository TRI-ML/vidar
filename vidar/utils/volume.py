# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from knk_vision.vidar.vidar.arch.losses.PhotometricLoss import PhotometricLoss
from knk_vision.vidar.vidar.geometry.camera import Camera
from knk_vision.vidar.vidar.utils.config import cfg_from_dict
from knk_vision.vidar.vidar.utils.tensor import grid_sample, interpolate_image
from knk_vision.vidar.vidar.utils.types import is_tensor


def get_depth_bins(mode, min_val, max_val, num_vals):
    """Get depth bins based on min/mas values and number of bins"""
    if mode == 'inverse':
        depth_bins = 1. / np.linspace(
            1. / max_val, 1. / min_val, num_vals)[::-1]
    elif mode == 'linear':
        depth_bins = np.linspace(
            min_val, max_val, num_vals)
    elif mode == 'sid':
        depth_bins = np.array(
            [np.exp(np.log(min_val) + np.log(max_val / min_val) * i / (num_vals - 1))
             for i in range(num_vals)])
    else:
        raise NotImplementedError
    return torch.from_numpy(depth_bins).float()


def depth2idx(depth, bins, invalid=-1):
    """Convert depth values into bin indices"""
    depth = F.relu(depth - bins.reshape(1, -1, 1, 1))
    idx = torch.min(depth, dim=1)[1]
    idx[(idx < 0) | (idx == len(bins) - 1)] = invalid
    return idx.unsqueeze(1)


def idx2radial(idx, feat, bins):
    """Convert bin indices into radial features"""
    b, _, h, w = idx.shape
    c, n = len(bins), feat.shape[1]
    idx = idx.permute(0, 2, 3, 1).reshape(-1, 1)
    invalid = idx == -1
    idx[invalid] = 0
    source = feat.permute(0, 2, 3, 1).reshape(-1, n)
    radial = []
    for i in range(n):
        radial_i = torch.zeros((b, c, h, w)).float().to(idx.device)
        radial_i = radial_i.permute(0, 2, 3, 1).reshape(-1, c)
        radial_i.scatter_(1, idx, source[:, [i]])
        radial_i[invalid.reshape(-1)] = 0
        radial.append(radial_i)
    radial = torch.stack(radial, -1)
    return radial.reshape(b, h, w, c, n).permute(0, 4, 3, 1, 2)


def depth2radial(depth, feat, bins):
    """Convert depth values into radial features"""
    return idx2radial(depth2idx(depth, bins), feat, bins)


def warp_bins(rgb, cam, bins):
    ones = torch.ones((1, *cam.hw)).to(rgb.device)
    volume = torch.stack([depth * ones for depth in bins], 1)
    coords_volume = cam.coords_from_cost_volume(volume)
    return grid_sample(
        rgb.repeat(len(bins), 1, 1, 1), coords_volume[0].type(rgb.dtype),
        padding_mode='zeros', mode='bilinear')


def sample(grid, pred):
    """Perform grid sampling based on prediction coordinates"""
    n, _, h, w = grid.shape
    coords = pred.permute(1, 2, 0).reshape(-1, 1, 1, 1).repeat(1, 1, 1, 2)
    coords = 2 * coords / (n - 1) - 1
    grid = grid.permute(2, 3, 0, 1).reshape(-1, 1, n, 1).repeat(1, 1, 1, 2)
    values = grid_sample(grid, coords, padding_mode='zeros', mode='bilinear')
    return values.reshape(h, w, 1, 1).permute(2, 3, 0, 1)


def compute_depth_bin(min_depth, max_depth, num_bins, i):
    """Get depth from bin index (assumes SID)"""
    return torch.exp(np.log(min_depth) + np.log(max_depth / min_depth) * i / (num_bins - 1)).\
        clamp(min=min_depth, max=max_depth)


def uncompute_depth_bin(min_depth, max_depth, num_bins, depth):
    """Get bin index from depth (assumes SID)"""
    return (num_bins - 1) * ((torch.log(depth) - np.log(min_depth)) / np.log(max_depth / min_depth)).clamp(min=0, max=num_bins)


def compute_depth_bins(min_depth, max_depth, num_bins, mode):
    """Compute depth bins from different modes"""
    if is_tensor(min_depth):
        min_depth = min_depth.detach().cpu()
    if is_tensor(max_depth):
        max_depth = max_depth.detach().cpu()
    if mode == 'inverse':
        depth_bins = 1. / np.linspace(
            1. / max_depth, 1. / min_depth, num_bins)[::-1]
    elif mode == 'linear':
        depth_bins = np.linspace(
            min_depth, max_depth, num_bins)
    elif mode == 'sid':
        depth_bins = np.array(
            [np.exp(np.log(min_depth) + np.log(max_depth / min_depth) * i / (num_bins - 1))
             for i in range(num_bins)])
    else:
        raise NotImplementedError
    return torch.from_numpy(depth_bins).float()
