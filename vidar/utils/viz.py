# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import flow_vis
import numpy as np
import torch
from matplotlib.cm import get_cmap

from vidar.utils.decorators import iterate1
from vidar.utils.depth import depth2inv
from vidar.utils.types import is_tensor, is_list


def flow_to_color(flow_uv, clip_flow=None):
    """
    Calculate color from optical flow

    Parameters
    ----------
    flow_uv : np.Array
        Optical flow [H,W,2]
    clip_flow : Float
        Clipping value for optical flow

    Returns
    -------
    colors : np.array
        Optical flow colormap [H,W,3]
    """
    # Clip if requested
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)
    # Get optical flow channels
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    # Calculate maximum radian
    rad_max = np.sqrt(2) * clip_flow if clip_flow is not None else \
        np.max(np.sqrt(np.square(u) + np.square(v)))
    # Normalize optical flow channels
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    # Return colormap [0,1]
    return flow_vis.flow_uv_to_colors(u, v, convert_to_bgr=False) / 255


@iterate1
@iterate1
def viz_inv_depth(inv_depth, normalizer=None, percentile=95,
                  colormap='plasma', filter_zeros=False):
    """
    Converts an inverse depth map to a colormap for visualization.

    Parameters
    ----------
    inv_depth : torch.Tensor
        Inverse depth map to be converted [B,1,H,W]
    normalizer : Float
        Value for inverse depth map normalization
    percentile : Float
        Percentile value for automatic normalization
    colormap : String
        Colormap to be used
    filter_zeros : Bool
        If True, do not consider zero values during normalization

    Returns
    -------
    colormap : np.Array [H,W,3]
        Colormap generated from the inverse depth map
    """
    if is_list(inv_depth):
        return [viz_inv_depth(
            inv[0], normalizer, percentile, colormap, filter_zeros)
            for inv in inv_depth]
    # If a tensor is provided, convert to numpy
    if is_tensor(inv_depth):
        # If it has a batch size, use first one
        if len(inv_depth.shape) == 4:
            inv_depth = inv_depth[0]
        # Squeeze if depth channel exists
        if len(inv_depth.shape) == 3:
            inv_depth = inv_depth.squeeze(0)
        inv_depth = inv_depth.detach().cpu().numpy()
    cm = get_cmap(colormap)
    if normalizer is None:
        if (inv_depth > 0).sum() == 0:
            normalizer = 1.0
        else:
            normalizer = np.percentile(
                inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    inv_depth = inv_depth / (normalizer + 1e-6)
    colormap = cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]
    colormap[inv_depth == 0] = 0
    return colormap


@iterate1
@iterate1
def viz_depth(depth, *args, **kwargs):
    """Same as viz_inv_depth, but takes depth as input instead"""
    return viz_inv_depth(depth2inv(depth), *args, **kwargs)


@iterate1
@iterate1
def viz_normals(normals):
    """
    Converts normals map to a colormap for visualization.

    Parameters
    ----------
    normals : torch.Tensor
        Inverse depth map to be converted [B,3,H,W]

    Returns
    -------
    colormap : np.Array
        Colormap generated from the normals map [H,W,3]
    """
    # If a tensor is provided, convert to numpy
    if is_tensor(normals):
        normals = normals.permute(1, 2, 0).detach().cpu().numpy()
    return (normals + 1) / 2


@iterate1
@iterate1
def viz_optical_flow(optflow, clip_value=100.):
    """
    Returns a colorized version of an optical flow map

    Parameters
    ----------
    optflow : torch.Tensor
        Optical flow to be colorized (NOT in batch) [2,H,W]
    clip_value : Float
        Optical flow clip value for visualization

    Returns
    -------
    colorized : np.Array
        Colorized version of the input optical flow [H,W,3]
    """
    # If a tensor is provided, convert to numpy
    if is_list(optflow):
        return [viz_optical_flow(opt[0]) for opt in optflow]
    if is_tensor(optflow):
        if len(optflow.shape) == 4:
            optflow = optflow[0]
        optflow = optflow.permute(1, 2, 0).detach().cpu().numpy()
    # Return colorized optical flow
    return flow_to_color(optflow, clip_flow=clip_value)


@iterate1
@iterate1
def viz_photo(photo, colormap='viridis', normalize=False):
    """
    Returns a colorized version of the photometric loss

    Parameters
    ----------
    photo : torch.Tensor
        Per-pixel photometric error
    colormap : String
        Which colormap to use
    normalize : Bool
        Whether the photometric error should be normalized between [0,1]

    Returns
    -------
    colorized : np.Array
        Colorized version of the photometric error [H,W,3]
    """
    if is_tensor(photo):
        if len(photo.shape) == 4:
            photo = photo[0]
        if len(photo.shape) == 3:
            photo = photo.squeeze(0)
        photo = photo.detach().cpu().numpy()
    cm = get_cmap(colormap)
    if normalize:
        photo -= photo.min()
        photo /= photo.max()
    colormap = cm(np.clip(photo, 0., 1.0))[:, :, :3]
    colormap[photo == 0] = 0
    return colormap


@iterate1
@iterate1
def viz_semantic(semantic, ontology):
    """
    Returns a colorized version of a semantic map

    Parameters
    ----------
    semantic : torch.Tensor
        Semantic map to be colorized [B,1,H,W]
    ontology : Dict
        Dictionary mapping between class and color

    Returns
    -------
    colorized : np.Array
        Colorized version of the semantic map [H,W,3]
    """
    # If it is a tensor, cast to numpy
    if is_tensor(semantic):
        if semantic.dim() == 3:
            semantic = semantic.squeeze(0)
        semantic = semantic.detach().cpu().numpy()
    # Create and populate color map
    color = np.zeros((semantic.shape[0], semantic.shape[1], 3))
    for key in ontology.keys():
        key_color = np.array(ontology[key]['color'])
        if is_tensor(key_color):
            key_color = key_color.detach().cpu().numpy()
        color[semantic == int(key)] = key_color / 255.
    # Return colored semantic map
    return color


@iterate1
@iterate1
def viz_camera(camera):
    """
    Returns a colorized version of a camera viewing rays

    Parameters
    ----------
    camera : Camera of torch.Tensor
        Input camera or viewing rays

    Returns
    -------
    colorized : np.Array
        Colorized version of the camera viewing rays [H,W,3]
    """
    if is_tensor(camera):
        # If it's a tensor, reshape it
        rays = camera[-3:].permute(1, 2, 0).detach().cpu().numpy()
    else:
        # If it's a camera, get viewing rays
        rays = camera.no_translation().get_viewdirs(normalize=True, flatten=False, to_world=True)
        rays = rays[0].permute(1, 2, 0).detach().cpu().numpy()
    return (rays + 1) / 2
