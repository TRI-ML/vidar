# Copyright 2023 Toyota Research Institute.  All rights reserved.

from copy import deepcopy

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from knk_vision.vidar.vidar.utils.data import keys_with, align_corners
from knk_vision.vidar.vidar.utils.decorators import iterate1
from knk_vision.vidar.vidar.utils.types import is_seq


@iterate1
def resize_pil(image, shape, interpolation=InterpolationMode.LANCZOS):
    """
    Resizes input image.

    Parameters
    ----------
    image : Image.PIL
        Input image
    shape : tuple [H,W]
        Output shape
    interpolation : int
        Interpolation mode

    Returns
    -------
    image : Image.PIL
        Resized image
    """
    transform = transforms.Resize(shape, interpolation=interpolation)
    return transform(image)


@iterate1
@iterate1
def resize_npy(depth, shape, expand=True):
    """
    Resizes depth map.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape
    expand : bool
        Expand output to [H,W,1]

    Returns
    -------
    depth : np.array [H,W]
        Resized depth map
    """
    # If a single number is provided, use resize ratio
    if not is_seq(shape):
        shape = tuple(int(s * shape) for s in depth.shape)
    # Resize depth map
    depth = cv2.resize(depth, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
    # Return resized depth map
    return np.expand_dims(depth, axis=2) if expand else depth


@iterate1
def resize_npy_preserve(depth, shape):
    """
    Resizes depth map preserving all valid depth pixels
    Multiple downsampled points can be assigned to the same pixel.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W,1]
        Resized depth map
    """
    # If a single number is provided, use resize ratio
    if not is_seq(shape):
        shape = tuple(int(s * shape) for s in depth.shape)
    # Store dimensions and reshapes to single column
    depth = np.squeeze(depth)
    h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (shape[0] / h)).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w)).astype(np.int32)
    # Filters points inside image
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    depth = np.zeros(shape)
    depth[crd[:, 0], crd[:, 1]] = val
    # Return resized depth map
    return np.expand_dims(depth, axis=2)


@iterate1
def resize_torch_preserve(depth, shape):
    if depth.dim() == 4:
        return torch.stack([resize_torch_preserve(depth[i], shape)
                            for i in range(depth.shape[0])], 0)
    # If a single number is provided, use resize ratio
    if not is_seq(shape):
        shape = tuple(int(s * shape) for s in depth.shape)
    # Store dimensions and reshapes to single column
    c, h, w = depth.shape
    # depth = np.squeeze(depth)
    # h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (shape[0] / h)).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w)).astype(np.int32)
    # Filters points inside image
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    depth = torch.zeros(shape, device=depth.device, dtype=depth.dtype)
    depth[crd[:, 0], crd[:, 1]] = val
    # Return resized depth map
    return depth.unsqueeze(0)


@iterate1
@iterate1
def resize_npy_multiply(data, shape):
    if data is None:
        return data
    ratio_w = shape[0] / data.shape[0]
    ratio_h = shape[1] / data.shape[1]
    out = resize_npy(data, shape, expand=False)
    out[..., 0] *= ratio_h
    out[..., 1] *= ratio_w
    return out


@iterate1
def resize_intrinsics(intrinsics, original, resized):
    """
    Resize camera intrinsics matrix to match a target resolution

    Parameters
    ----------
    intrinsics : np.array [3,3]
        Original intrinsics matrix
    original : tuple [W,H]
        Original image resolution
    resized : tuple [w,h]
        Target image resolution
    Returns
    -------
    intrinsics : np.array [3,3]
        Resized intrinsics matrix
    """
    intrinsics = np.copy(intrinsics)

    ratio_w = resized[0] / original[0]
    ratio_h = resized[1] / original[1]

    intrinsics[0, 0] *= ratio_w
    intrinsics[1, 1] *= ratio_h

    # if align_corners():
    intrinsics[0, 2] = intrinsics[0, 2] * ratio_w
    intrinsics[1, 2] = intrinsics[1, 2] * ratio_h
    # else:
    #     intrinsics[0, 2] = (intrinsics[0, 2] - 0.5) * ratio_w + 0.5
    #     intrinsics[1, 2] = (intrinsics[1, 2] - 0.5) * ratio_h + 0.5

    return intrinsics


def resize_sample_input(sample, shape, shape_supervision=None,
                        depth_downsample=1.0, preserve_depth=False,
                        pil_interpolation=InterpolationMode.LANCZOS):
    """
    Resizes the input information of a sample (i.e. that go to the networks)

    Parameters
    ----------
    sample : dict
        Dictionary with sample values (output from a dataset's __getitem__ method)
    shape : tuple (H,W)
        Output shape
    shape_supervision : tuple (H,W)
        Output supervision shape
    depth_downsample: float
        Resize ratio for depth maps
    preserve_depth : bool
        Preserve depth maps when resizing
    pil_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Intrinsics
    for key in keys_with(sample, 'intrinsics', without='raw'):
        # if f'{key}_raw' not in sample.keys():
        #     sample[f'{key}_raw'] = deepcopy(sample[key])
        sample[key] = resize_intrinsics(sample[key], list(sample['rgb'].values())[0].size, shape[::-1])
    # RGB
    for key in keys_with(sample, 'rgb', without='raw'):
        sample[key] = resize_pil(sample[key], shape, interpolation=pil_interpolation)
    # Mask
    for key in keys_with(sample, 'mask', without='raw'):
        sample[key] = resize_pil(sample[key], shape, interpolation=InterpolationMode.NEAREST)
    # Input depth
    for key in keys_with(sample, 'input_depth'):
        shape_depth = [int(s * depth_downsample) for s in shape]
        resize_npy_depth = resize_npy_preserve if preserve_depth else resize_npy
        sample[key] = resize_npy_depth(sample[key], shape_depth)
    return sample


def resize_sample_supervision(sample, shape, depth_downsample=1.0, preserve_depth=False):
    """
    Resizes the output information of a sample (i.e. ground-truth supervision)

    Parameters
    ----------
    sample : dict
        Dictionary with sample values (output from a dataset's __getitem__ method)
    shape : tuple (H,W)
        Output shape
    depth_downsample: float
        Resize ratio for depth maps
    preserve_depth : bool
        Preserve depth maps when resizing

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Depth
    for key in keys_with(sample, 'depth', without='input_depth'):
        shape_depth = [int(s * depth_downsample) for s in shape]
        resize_npy_depth = resize_npy_preserve if preserve_depth else resize_npy
        sample[key] = resize_npy_depth(sample[key], shape_depth)
    # Normals
    for key in keys_with(sample, 'normals'):
        shape_normals = [int(s * depth_downsample) for s in shape]
        sample[key] = resize_npy(sample[key], shape_normals)
    # Semantic
    for key in keys_with(sample, 'semantic'):
        sample[key] = resize_npy(sample[key], shape, expand=False)
    # Optical flow
    for key in keys_with(sample, 'optical_flow'):
        sample[key] = resize_npy_multiply(sample[key], shape)
    # Scene flow
    for key in keys_with(sample, 'scene_flow'):
        sample[key] = resize_npy(sample[key], shape, expand=False)
    # Return resized sample
    return sample


def resize_sample(sample, shape, shape_supervision=None, depth_downsample=1.0, preserve_depth=False,
                  pil_interpolation=InterpolationMode.LANCZOS):
    """
    Resizes a sample, including image, intrinsics and depth maps.

    Parameters
    ----------
    sample : dict
        Dictionary with sample values (output from a dataset's __getitem__ method)
    shape : tuple (H,W)
        Output shape
    shape_supervision : tuple (H,W)
        Output shape
    depth_downsample: float
        Resize ratio for depth maps
    preserve_depth : bool
        Preserve depth maps when resizing
    pil_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize input information
    sample = resize_sample_input(sample, shape,
                                 depth_downsample=depth_downsample,
                                 preserve_depth=preserve_depth,
                                 pil_interpolation=pil_interpolation)
    # Resize output information
    sample = resize_sample_supervision(sample, shape_supervision,
                                       depth_downsample=depth_downsample,
                                       preserve_depth=preserve_depth)
    # Return resized sample
    return sample
