# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as tfn

from vidar.geometry.camera import Camera
from vidar.utils.decorators import iterate1
from vidar.utils.types import is_tensor, is_numpy


@iterate1
@iterate1
def inv2depth(inv_depth):
    """
    Invert an inverse depth map to produce a depth map

    Parameters
    ----------
    inv_depth : torch.Tensor or list[torch.Tensor] or np.array or list[np.array]
        Inverse depth map [B,1,H,W]

    Returns
    -------
    depth : torch.Tensor or list[torch.Tensor] or np.array or list[np.array]
        Depth map [B,1,H,W]
    """
    if is_tensor(inv_depth):
        depth = 1. / inv_depth.clamp(min=1e-6, max=None)
    elif is_numpy(inv_depth):
        depth = 1. / inv_depth.clip(min=1e-6, max=None)
    else:
        raise NotImplementedError('Wrong inverse depth format.')
    depth[inv_depth <= 0.] = 0.
    return depth


@iterate1
@iterate1
def depth2inv(depth):
    """
    Invert a depth map to produce an inverse depth map

    Parameters
    ----------
    depth : torch.Tensor or list[torch.Tensor] or np.array or list[np.array]
        Depth map [B,1,H,W]

    Returns
    -------
    inv_depth : torch.Tensor or list[torch.Tensor] pr np.array or list[np.array]
        Inverse depth map [B,1,H,W]

    """
    if is_tensor(depth):
        inv_depth = 1. / depth.clamp(min=1e-6, max=None)
    elif is_numpy(depth):
        inv_depth = 1. / depth.clip(min=1e-6, max=None)
    else:
        raise NotImplementedError('Wrong depth format')
    inv_depth[depth <= 0.] = 0.
    return inv_depth


def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor
        Inverse depth map [B,1,H,W]
    inv_depth_hat : torch.Tensor
        Flipped inverse depth map produced from a flipped image [B,1,H,W]
    method : String
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))


def post_process_inv_depth(inv_depth, inv_depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor
        Inverse depth map [B,1,H,W]
    inv_depth_flipped : torch.Tensor
        Inverse depth map produced from a flipped image [B,1,H,W]
    method : String
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor
        Post-processed inverse depth map [B,1,H,W]
    """
    from vidar.utils.flip import flip_lr
    B, C, H, W = inv_depth.shape
    inv_depth_hat = inv_depth_flipped
    inv_depth_fused = fuse_inv_depth(inv_depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=inv_depth.device,
                        dtype=inv_depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    post_processed = mask_hat * inv_depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused
    mask0, mask_hat0 = inv_depth == 0, inv_depth_hat == 0
    post_processed[mask_hat0] = inv_depth[mask_hat0]
    post_processed[mask0] = inv_depth_hat[mask0]
    post_processed[mask0 & mask_hat0] = 0
    return post_processed


def post_process_depth(depth, depth_flipped, method='mean'):
    """Post-process a depth map and flipped depth map"""
    return inv2depth(post_process_inv_depth(
        depth2inv(depth), depth2inv(depth_flipped), method=method))


def calculate_normals(points, camera=None, intrinsics=None, pad_last=True):
    """
    Calculate normals from a pointcloud map or from a depth map + intrinsics

    Parameters
    ----------
    points : torch.Tensor
        A pointcloud map [B,3,H,W] containing 3D coordinates
        A depth map [B,1,H,W] containing depth values
    camera : Camera
        Camera used for normal calculation, in case a depth map is provided
    intrinsics : torch.Tensor
        Camera intrinsics [B,3,3] necessary in case a depth map is provided, to create the pointcloud map
    pad_last : Bool
        If true, pad the last row and column with zeros

    Returns
    -------
    normals : torch.Tensor
        Normal map [B,3,H,W] containing normal estimates
    """
    if intrinsics is None and camera is None:
        return points
    # Create pointcloud map if intrinsics are provided
    if camera is not None:
        points = camera.reconstruct_depth_map(points)
    elif intrinsics is not None:
        points = Camera(K=intrinsics, hw=points).reconstruct_depth_map(points)
    # Prepare points for cross-product
    p0 = points[:, :, :-1, :-1]
    p1 = points[:, :,  1:, :-1]
    p2 = points[:, :, :-1,  1:]
    # Calculate and normalize normals
    normals = torch.cross(p1 - p0, p2 - p0, 1)
    normals = normals / normals.norm(dim=1, keepdim=True)
    # Pad normals
    if pad_last:
        normals = torch.nn.functional.pad(normals, [0, 1, 0, 1], mode='replicate')
    # # Substitute nan values with zero
    normals[torch.isnan(normals)] = 0.0
    # Return normals
    return normals


def calc_dot_prod(pts, nrm):
    """
    Calculate dot product of 3D points and their normals

    Parameters
    ----------
    pts : torch.Tensor
        Input 3D points [B,3,H,W]
    nrm : torch.Tensor
        Input 3D normal vectors [B,3,H,W]

    Returns
    -------
    dots : torch.Tensor
        Output dot product [B,1,H,W]
    """
    pts = pts / pts.norm(dim=1, keepdim=True)
    nrm = nrm / nrm.norm(dim=1, keepdim=True)
    dots = torch.sum(pts * nrm, dim=1, keepdim=True)
    return dots


def get_depth_bins(mode, min_val, max_val, num_vals):
    """
    Create discretize depth bins

    Parameters
    ----------
    mode : String
        Discretization mode
    min_val : Float
        Minimum depth value
    max_val : Float
        Maximum depth value
    num_vals : Int
        Number of intervals

    Returns
    -------
    bins : torch.Tensor
        Discretized depth values [num_vals]
    """
    if mode == 'inverse':
        depth_bins = 1. / torch.linspace(
            1. / max_val, 1. / min_val, num_vals)[::-1]
    elif mode == 'linear':
        depth_bins = torch.linspace(
            min_val, max_val, num_vals)
    elif mode == 'sid':
        depth_bins = torch.tensor(
            [torch.exp(torch.log(min_val) + torch.log(max_val / min_val) * i / (num_vals - 1))
             for i in range(num_vals)])
    else:
        raise NotImplementedError
    return depth_bins.float()


def depth2index(depth, bins):
    """
    Convert a depth map to discretized indexes

    Parameters
    ----------
    depth : torch.Tensor
        Input depth map [B,1,H,W]
    bins : torch.Tensor
        Discretized depth bins [D]

    Returns
    -------
    idx : torch.Tensor
        Discretized depth indexes [B,1,H,W]
    """
    if depth.dim() == 2:
        depth = tfn.relu(depth - bins.reshape(1, -1))
    elif depth.dim() == 4:
        depth = tfn.relu(depth - bins.reshape(1, -1, 1, 1))
    else:
        raise ValueError('Invalid depth dimension')
    idx = torch.min(depth, dim=1)[1]
    # idx[(idx < 0) | (idx == len(bins) - 1)] = -1
    idx[(idx < 0)] = 0
    idx[idx > len(bins) - 1] = len(bins) - 1
    return idx.unsqueeze(1)


def index2depth(idx, bins):
    """
    Converts discretized indexes to depth map

    Parameters
    ----------
    idx : torch.Tensor
        Discretized indexes [B,1,H,W]
    bins : torch.Tensor
        Discretized depth bins [D]

    Returns
    -------
    depth : torch.Tensor
        Output depth map [B,1,H,W]
    """
    if idx.dim() == 4:
        b, _, h, w = idx.shape
        bins = bins.reshape(1, -1, 1, 1).repeat(idx.shape[0], 1, idx.shape[2], idx.shape[3]).to(idx.device)
    elif idx.dim() == 3:
        b, _, n = idx.shape
        bins = bins.reshape(1, -1, 1).repeat(idx.shape[0], 1, idx.shape[2]).to(idx.device)
    else:
        raise ValueError('Invalid depth dimension')
    return torch.gather(bins, 1, idx)
