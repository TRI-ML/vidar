# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch

from vidar.geometry.camera import Camera
from vidar.utils.tensor import grid_sample
from vidar.utils.types import is_tensor


def warp_bins(rgb, cam, bins):
    """
    Warp an image based on depth bins

    Parameters
    ----------
    rgb : torch.Tensor [B,?,H,W]
        Input image for warping
    cam : Camera
        Input camera
    bins : torch.Tensor
        Depth bins for warping

    Returns
    -------
    warped : torch.Tensor
        Warped images for each depth bin
    """
    ones = torch.ones((1, *cam.hw)).to(rgb.device)
    volume = torch.stack([depth * ones for depth in bins], 1)
    coords_volume = cam.coords_from_cost_volume(volume)
    return grid_sample(
        rgb.repeat(len(bins), 1, 1, 1), coords_volume[0].type(rgb.dtype),
        padding_mode='zeros', mode='bilinear', align_corners=True)


def sample(grid, pred):
    """
    Sample a grid based on predictions

    Parameters
    ----------
    grid : torch.Tensor
        Grid to be sampled [B,?,H,W]
    pred : torch.Tensor
        Coordinate predictions [B,2,H,W]

    Returns
    -------
    values : torch.Tensor
        Sampled grid[B,?,H,W]
    """
    n, _, h, w = grid.shape
    coords = pred.permute(1, 2, 0).reshape(-1, 1, 1, 1).repeat(1, 1, 1, 2)
    coords = 2 * coords / (n - 1) - 1
    grid = grid.permute(2, 3, 0, 1).reshape(-1, 1, n, 1).repeat(1, 1, 1, 2)
    values = grid_sample(grid, coords,
        padding_mode='zeros', mode='bilinear', align_corners=True)
    return values.reshape(h, w, 1, 1).permute(2, 3, 0, 1)


def compute_depth_bin(min_depth, max_depth, num_bins, i):
    """
    Calculate a single SID depth bin

    Parameters
    ----------
    min_depth : Float
        Minimum depth value
    max_depth : Float
        Maximum depth value
    num_bins : Int
        Number of depth bins
    i : Int
        Index of the depth bin in the interval

    Returns
    -------
    bin : torch.Tensor
        Corresponding depth bin
    """
    return torch.exp(np.log(min_depth) + np.log(max_depth / min_depth) * i / (num_bins - 1)).\
        clamp(min=min_depth, max=max_depth)


def uncompute_depth_bin(min_depth, max_depth, num_bins, depth):
    """
    Recover the SID bin index from a depth value

    Parameters
    ----------
    min_depth : Float
        Minimum depth value
    max_depth : Float
        Maximum depth value
    num_bins : Int
        Number of depth bins
    depth : torch.Tensor
        Depth value

    Returns
    -------
    index : torch.Tensor
        Index for the depth value in the SID interval
    """
    return (num_bins - 1) * ((torch.log(depth) - np.log(min_depth)) /
                             np.log(max_depth / min_depth)).clamp(min=0, max=num_bins)


def compute_depth_bins(min_depth, max_depth, num_bins, mode):
    """
    Compute depth bins for an interval

    Parameters
    ----------
    min_depth : Float
        Minimum depth value
    max_depth : Float
        Maximum depth value
    num_bins : Int
        Number of depth bins
    mode : String
        Depth discretization mode

    Returns
    -------
    bins : torch.Tensor
        Discretized depth bins
    """
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
