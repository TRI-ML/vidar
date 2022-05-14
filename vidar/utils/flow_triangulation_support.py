# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
import torch.nn.functional as tfunc

from vidar.utils.tensor import pixel_grid, cat_channel_ones


def bearing_grid(rgb, intrinsics):
    """
    Create a homogeneous bearing grid from camera intrinsics and a base image

    Parameters
    ----------
    rgb : torch.Tensor
        Base image for dimensions [B,3,H,W]
    intrinsics : torch.Tensor
        Camera intrinsics [B,3,3]

    Returns
    -------
    grid : torch.Tensor
        Bearing grid [B,3,H,W]
    """
    # Create pixel grid from base image
    b, _, h, w = rgb.shape
    grid = pixel_grid((h, w), b).to(rgb.device)
    # Normalize pixel grid with camera parameters
    grid[:, 0] = (grid[:, 0] - intrinsics[:, 0, 2].unsqueeze(-1).unsqueeze(-1)) / intrinsics[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    grid[:, 1] = (grid[:, 1] - intrinsics[:, 1, 2].unsqueeze(-1).unsqueeze(-1)) / intrinsics[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    # Return bearing grid (with 1s as extra dimension)
    return cat_channel_ones(grid)


def mult_rotation_bearing(rotation, bearing):
    """
    Rotates a bearing grid

    Parameters
    ----------
    rotation : torch.Tensor
        Rotation matrix [B,3,3]
    bearing : torch.Tensor
        Bearing grid [B,3,H,W]

    Returns
    -------
    rot_bearing : torch.Tensor
        Rotated bearing grid [B,3,H,W]
    """
    # Multiply rotation and bearing
    product = torch.bmm(rotation, bearing.view(bearing.shape[0], 3, -1))
    # Return product with bearing shape
    return product.view(bearing.shape)


def pre_triangulation(ref_bearings, ref_translations, tgt_flows,
                      intrinsics, concat=True):
    """
    Triangulates bearings and flows

    Parameters
    ----------
    ref_bearings : list[torch.Tensor]
        Reference bearings [B,3,H,W]
    ref_translations : list[torch.Tensor]
        Reference translations [B,3]
    tgt_flows : list[torch.Tensor]
        Target optical flow values [B,2,H,W]
    intrinsics : torch.Tensor
        Camera intrinsics [B,3,3]
    concat : Bool
        True if cross product results are concatenated

    Returns
    -------
    rs : torch.Tensor or list[torch.Tensor]
        Bearing x translation cross product [B,3,H,W] (concatenated or not)
    ss : torch.Tensor or list[torch.Tensor]
        Bearing x bearing cross product [B,3,H,W] (concatenated or not)
    """
    # Get target bearings from flow
    tgt_bearings = [flow2bearing(flow, intrinsics, normalize=True)
                    for flow in tgt_flows]
    # Bearings x translation cross product
    rs = [torch.cross(tgt_bearing, ref_translation[:, :, None, None].expand_as(tgt_bearing), dim=1)
          for tgt_bearing, ref_translation in zip(tgt_bearings, ref_translations)]
    # Bearings x bearings cross product
    ss = [torch.cross(tgt_bearing, ref_bearing, dim=1)
          for tgt_bearing, ref_bearing in zip(tgt_bearings, ref_bearings)]
    if concat:
        # If results are to be concatenated
        return torch.cat(rs, dim=1), torch.cat(ss, dim=1)
    else:
        # Otherwise, return as lists
        return rs, ss


def depth_ls2views(r, s, clip_range=None):
    """
    Least-squares depth estimation from two views

    Parameters
    ----------
    r : torch.Tensor
        Bearing x translation cross product between images [B,3,H,W]
    s : torch.Tensor
        Bearing x translation cross product between images [B,3,H,W]
    clip_range : Tuple
        Depth clipping range (min, max)

    Returns
    -------
    depth : torch.Tensor
        Calculated depth [B,1,H,W]
    error : torch.Tensor
        Calculated error [B,1,H,W]
    hessian : torch.Tensor
        Calculated hessian [B,1,H,W]

    """
    # Calculate matrices
    hessian = (s * s).sum(dim=1, keepdims=True)
    depth = -(s * r).sum(dim=1, keepdims=True) / (hessian + 1e-30)
    error = (r * r).sum(dim=1, keepdims=True) - hessian * (depth ** 2)

    # Clip depth and other matrices if requested
    if clip_range is not None:

        invalid_mask = (depth <= clip_range[0])
        invalid_mask |= (depth >= clip_range[1])

        depth[invalid_mask] = 0
        error[invalid_mask] = 0
        hessian[invalid_mask] = 0
    # Return calculated matrices
    return depth, error, hessian


def flow2bearing(flow, intrinsics, normalize=True):
    """
    Convert optical flow to bearings

    Parameters
    ----------
    flow : torch.Tensor
        Input optical flow [B,2,H,W]
    intrinsics : torch.Tensor
        Camera intrinsics [B,3,3]
    normalize : Bool
        True if bearings are normalized

    Returns
    -------
    bearings : torch.Tensor
        Calculated bearings [B,3,H,W]
    """
    # Create initial grid
    height, width = flow.shape[2:]
    xx, yy = np.meshgrid(range(width), range(height))
    # Initialize bearing matrix
    bearings = torch.zeros_like(flow)
    # Populate bearings
    match = (flow[:, 0] + torch.from_numpy(xx).to(flow.device),
             flow[:, 1] + torch.from_numpy(yy).to(flow.device))
    bearings[:, 0] = (match[0] - intrinsics[:, 0, 2].unsqueeze(-1).unsqueeze(-1)) / intrinsics[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    bearings[:, 1] = (match[1] - intrinsics[:, 1, 2].unsqueeze(-1).unsqueeze(-1)) / intrinsics[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    # Stack 1s as the last dimension
    bearings = cat_channel_ones(bearings)
    # Normalize if necessary
    if normalize:
        bearings = tfunc.normalize(bearings)
    # Return bearings
    return bearings


def triangulation(ref_bearings, ref_translations,
                  tgt_flows, intrinsics, clip_range=None, residual=False):
    """
    Triangulate optical flow points to produce depth estimates

    Parameters
    ----------
    ref_bearings : list[torch.Tensor]
        Reference bearings [B,3,H,W]
    ref_translations : list[torch.Tensor]
        Reference translations [B,3]
    tgt_flows : list[torch.Tensor]
        Target optical flow to reference [B,2,H,W]
    intrinsics : torch.Tensor
        Camera intrinsics [B,3,3]
    clip_range : Tuple
        Depth clipping range
    residual : Bool
        True to return residual error and squared root of Hessian

    Returns
    -------
    depth : torch.Tensor
        Estimated depth [B,1,H,W]
    error : torch.Tensor
        Estimated error [B,1,H,W]
    sqrt_hessian : torch.Tensor
        Squared root of Hessian [B,1,H,W]
    """
    # Pre-triangulate flows
    rs, ss = pre_triangulation(ref_bearings, ref_translations, tgt_flows, intrinsics, concat=False)
    # Calculate list of triangulations
    outputs = [depth_ls2views(*rs_ss, clip_range=clip_range) for rs_ss in zip(rs, ss)]
    # Calculate predicted hessian and depths
    hessian = sum([output[2] for output in outputs])
    depth = sum([output[0] * output[2] for output in outputs]) / (hessian + 1e-12)
    # Return depth + residual error and hessian matrix
    if residual:
        error = torch.sqrt(sum([output[2] * (depth - output[0]) ** 2 + output[1]
                                for output in outputs]).clamp_min(0))
        sqrt_hessian = torch.sqrt(hessian)
        return depth, (error, sqrt_hessian)
    # Return depth
    else:
        return depth

