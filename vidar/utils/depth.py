# Copyright 2023 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
import torch.nn.functional as tfn

from knk_vision.vidar.vidar.geometry.camera import Camera
from knk_vision.vidar.vidar.utils.decorators import iterate1
from knk_vision.vidar.vidar.utils.tensor import same_shape
from knk_vision.vidar.vidar.utils.types import is_tensor, is_numpy, is_list, is_dict


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
    method : str
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
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor
        Post-processed inverse depth map [B,1,H,W]
    """
    from knk_vision.vidar.vidar.utils.flip import flip_lr
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
    """Post-process a depth map and its flipped version"""
    return inv2depth(post_process_inv_depth(
        depth2inv(depth), depth2inv(depth_flipped), method=method))


def calculate_normals(depth, camera=None, intrinsics=None, to_world=False, pad_last=True):
    """
    Calculate normals from a pointcloud map or from a depth map + intrinsics

    Parameters
    ----------
    depth : torch.Tensor
        A pointcloud map [B,3,H,W] containing 3D coordinates
        A depth map [B,1,H,W] containing depth values
    camera : Camera
        Camera used for normal calculation, in case a depth map is provided
    intrinsics : torch.Tensor
        Camera intrinsics [B,3,3] necessary in case a depth map is provided, to create the pointcloud map
    to_world: bool
        Return world normals
    pad_last : bool
        If true, pad the last row and column with zeros

    Returns
    -------
    normals : torch.tensor
        Normal map [B,3,H,W] containing normal estimates
    """
    if is_dict(depth):
        return {key: calculate_normals(
            depth[key], camera[key], to_world=to_world, pad_last=pad_last) for key in depth.keys()}
    if is_list(depth):
        return [calculate_normals(
            depth[i], camera, to_world=to_world, pad_last=pad_last) for i in range(len(depth))]
    if intrinsics is None and camera is None:
        return depth
    # Create pointcloud map if intrinsics are provided
    if camera is not None:
        if not same_shape(depth.shape[-2:], camera.hw):
            camera = camera.scaled(depth.shape[-2:])
        points = camera.reconstruct_depth_map(depth, to_world=to_world)
    elif intrinsics is not None:
        points = Camera(K=intrinsics, hw=depth).reconstruct_depth_map(depth, to_world=to_world)
    else:
        raise ValueError('Invalid input for calculate_normals')
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
    """Calculate dot product between points and normals"""
    pts = pts / pts.norm(dim=1, keepdim=True)
    nrm = nrm / nrm.norm(dim=1, keepdim=True)
    dots = torch.sum(pts * nrm, dim=1, keepdim=True)
    return dots


def get_depth_bins(mode, near, far, num, perturb=False, shape=None, device=None):
    """Calculate depth bins given a mode and a range"""

    if is_tensor(far):
        return get_depth_bins_volume(mode, near, far, num, perturb=perturb)
    if mode == 'linear':
        depth_bins = torch.linspace(
            near, far, num)
    elif mode == 'inverse':
        depth_bins = 1. / torch.linspace(
            1. / far, 1. / near, num)
        depth_bins = torch.flip(depth_bins, [0])
    elif mode == 'sid':
        depth_bins = torch.tensor(
            [np.exp(np.log(near) + np.log(far / near) * i / (num - 1))
             for i in range(num)])
    else:
        raise NotImplementedError
    depth_bins = depth_bins.float().to(device)

    if shape is not None:
        depth_bins = depth_bins.unsqueeze(0).repeat(*shape.shape[:-1], 1)
        if perturb:
            delta = depth_bins[..., 1:] - depth_bins[..., :-1]
            depth_bins[..., :-1] += torch.rand_like(delta) * delta
            # delta_last = 0.5 * (depth_bins[..., [-1]] - depth_bins[..., [-2]])
            # delta_last = torch.rand_like(delta_last) * delta_last
            # depth_bins[..., [-1]] -= delta_last
    elif perturb:
        delta = depth_bins[..., 1:] - depth_bins[:-1]
        depth_bins[..., :-1] += torch.rand_like(delta) * delta

    return depth_bins


def get_depth_bins_volume(mode, near, far, num, perturb=False):
    """Get depth bins for a cost volume"""
    if is_dict(far):
        return {key: get_depth_bins_volume(mode, near, far[key], num, perturb=perturb)
                for key in far.keys()}
    if is_list(far):
        return [get_depth_bins_volume(mode, near, far[i], num, perturb=perturb)
                for i in range(len(far))]
    bins = get_depth_bins(mode, 0.0, 1.0, num)
    if is_tensor(far):
        near = torch.ones_like(far) * near
    elif is_tensor(near):
        far = torch.ones_like(near) * far
    else:
        raise ValueError('Invalid near/far for get_depth_bins_volume')
    delta = far - near
    depth_bins = torch.stack([delta * b for b in bins], 2) + near.unsqueeze(2)
    if perturb:
        diff = (depth_bins[:, :, 1:] - depth_bins[:, :, :-1])
        depth_bins[:, :, :-1] += torch.rand_like(diff) * diff
    return depth_bins


def depth2index(depth, bins, clip=False):
    """Convert depth map to index map"""
    if depth.dim() == 2:
        depth = tfn.relu(depth - bins.reshape(1, -1))
    elif depth.dim() == 4:
        depth = tfn.relu(depth - bins.reshape(1, -1, 1, 1))
    else:
        raise ValueError('Invalid depth dimension')
    idx = torch.min(depth, dim=1)[1]
    if clip:
        idx[(idx < 0)] = 0
        idx[idx > len(bins) - 1] = len(bins) - 1
    else:
        idx[(idx < 0) | (idx > len(bins) - 1)] = -1
    return idx.unsqueeze(1)


def index2depth(idx, bins):
    """Convert index map to depth map"""
    if idx.dim() == 4:
        b, _, h, w = idx.shape
        bins = bins.reshape(1, -1, 1, 1).repeat(idx.shape[0], 1, idx.shape[2], idx.shape[3]).to(idx.device)
    elif idx.dim() == 3:
        b, _, n = idx.shape
        bins = bins.reshape(1, -1, 1).repeat(idx.shape[0], 1, idx.shape[2]).to(idx.device)
    else:
        raise ValueError('Invalid depth dimension')
    invalid = (idx < 0) | (idx > bins.shape[1] - 1)
    idx[invalid] = 0
    depth = torch.gather(bins, 1, idx)
    depth[invalid] = 0
    return depth


def index2radial(idx, bins, feat=None):
    """Convert index map to radial map"""
    b, _, h, w = idx.shape
    if feat is None:
        feat = torch.ones_like(idx).float()
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


def depth2radial(depth, bins, feat=None):
    """Convert depth map to radial map"""
    return index2radial(depth2index(depth, bins), bins, feat)


# @iterate1
# @iterate1
def to_depth(data, info):
    """Convert data to depth map"""

    if is_dict(data):
        return {key: to_depth(data[key], info[key]) for key in data.keys()}
    elif is_list(data):
        return [to_depth(d, info) for d in data]

    if data.shape[1] == 1:
        depth = data

    elif data.shape[1] == 5:

        mu0, mu1 = data[:, [0]], data[:, [1]]
        std0, std1 = data[:, [2]], data[:, [3]]

        w0 = data[:, [4]]
        w1 = 1.0 - w0

        mask = (w0 / std0 > w1 / std1).float()
        depth = mu0 * mask + mu1 * (1. - mask)

    else:

        bins = info['z_samples'].permute(0, 2, 1).reshape(len(data), -1, *data.shape[-2:])
        # idx = torch.argmax(torch.nn.functional.softmax(data, dim=1), dim=1, keepdim=True)
        idx = torch.argmax(data, dim=1, keepdim=True)
        depth = torch.gather(bins, 1, idx)

    return depth.contiguous()


def to_rgb(rgb, depth):
    """Convert data to rgb image"""

    if is_dict(rgb):
        return {key: to_rgb(rgb[key], depth[key]) for key in rgb.keys()}
    elif is_list(rgb):
        return [to_rgb(r, d) for r, d in zip(rgb, depth)]

    # idx = torch.argmax(torch.nn.functional.softmax(depth, dim=1), dim=1, keepdim=True)
    idx = torch.argmax(depth, dim=1, keepdim=True)
    rgb = torch.gather(rgb, 1, idx.unsqueeze(2).repeat(1, 1, 3, 1, 1)).squeeze(1)

    return rgb.contiguous()


def compute_scale_and_shift(pred, gt, mask=None):
    """Compute scale and shift between two tensors, used to scale a depth map"""

    if mask is None:
        mask = torch.ones_like(pred)

    if len(pred.shape) == 1:
        dims = 0
    elif len(pred.shape) == 4:
        dims = (2, 3)
    else:
        raise ValueError('Invalid shape')

    a_00 = torch.sum(mask * pred * pred, dims)
    a_01 = torch.sum(mask * pred, dims)
    a_11 = torch.sum(mask, dims)

    b_0 = torch.sum(mask * pred * gt, dims)
    b_1 = torch.sum(mask * gt, dims)

    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0

    x_0[valid] = ( a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def scale_and_shift_pred(pred, gt, mask=None):
    """Scale a depth map to match a ground truth depth map"""
    scale, shift = compute_scale_and_shift(pred, gt, mask)
    scaled_pred = scale.view(-1, 1, 1, 1) * pred + shift.view(-1, 1, 1, 1)
    return scale, shift, scaled_pred


def masked_shift_and_scale(depth_preds, depth_gt, mask_valid):
    """Compute the shift and scale of the depth predictions to match the ground truth depth map"""

    depth_preds_nan = depth_preds.clone()
    depth_gt_nan = depth_gt.clone()
    depth_preds_nan[~mask_valid] = np.nan
    depth_gt_nan[~mask_valid] = np.nan

    mask_diff = mask_valid.view(mask_valid.size()[:2] + (-1,)).sum(-1, keepdims=True) + 1

    t_gt = depth_gt_nan.view(depth_gt_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_gt[torch.isnan(t_gt)] = 0
    diff_gt = torch.abs(depth_gt - t_gt)
    diff_gt[~mask_valid] = 0
    s_gt = (diff_gt.view(diff_gt.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    depth_gt_aligned = (depth_gt - t_gt) / (s_gt + 1e-6)


    t_pred = depth_preds_nan.view(depth_preds_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
    t_pred[torch.isnan(t_pred)] = 0
    diff_pred = torch.abs(depth_preds - t_pred)
    diff_pred[~mask_valid] = 0
    s_pred = (diff_pred.view(diff_pred.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
    depth_pred_aligned = (depth_preds - t_pred) / (s_pred + 1e-6)

    return depth_pred_aligned, depth_gt_aligned


def align(depth, mask, eps=1e-6):
    """Align a depth map to the median depth value"""

    b, c, h, w = depth.shape
    depth_nan = depth.clone()
    depth_nan[~mask] = np.nan

    depth_nan = depth_nan.view(b, c, -1)
    shift = depth_nan.nanmedian(-1, keepdim=True)[0].unsqueeze(-1)

    diff = torch.abs(depth - shift)
    diff[~mask] = 0

    mask_diff = mask.view(b, c, -1).sum(-1, keepdims=True) + 1
    scale = (diff.view(b, c, -1).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)

    aligned = (depth - shift) / (scale + eps)
    return shift, scale, aligned
