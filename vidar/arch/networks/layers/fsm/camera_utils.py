# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as tf


def construct_K(fx, fy, cx, cy, dtype=torch.float, device=None):
    """Construct a [3,3] camera intrinsics from pinhole parameters"""
    return torch.tensor([[fx,  0, cx],
                         [ 0, fy, cy],
                         [ 0,  0,  1]], dtype=dtype, device=device)


def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K = K.clone()
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    # K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    # K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    K[..., 0, 2] = K[..., 0, 2] * x_scale
    K[..., 1, 2] = K[..., 1, 2] * y_scale
    return K


def invert_intrinsics(K):
    """Invert camera intrinsics"""
    Kinv = K.clone()
    Kinv[:, 0, 0] = 1. / K[:, 0, 0]
    Kinv[:, 1, 1] = 1. / K[:, 1, 1]
    Kinv[:, 0, 2] = -1. * K[:, 0, 2] / K[:, 0, 0]
    Kinv[:, 1, 2] = -1. * K[:, 1, 2] / K[:, 1, 1]
    return Kinv


def view_synthesis(ref_image, depth, ref_cam, cam, scene_flow=None,
                   mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor
        Reference image to be warped [B,3,H,W]
    depth : torch.Tensor
        Depth map from the original image [B,1,H,W]
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    scene_flow : torch.Tensor
        Scene flow use for warping [B,3,H,W]
    mode : str
        Mode for grid sampling
    padding_mode : str
        Padding mode for grid sampling
    align_corners : bool
        Corner alignment for grid sampling

    Returns
    -------
    ref_warped : torch.Tensor
        Warped reference image in the original frame of reference [B,3,H,W]
    """
    assert depth.shape[1] == 1, 'Depth map should have C=1'
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w', scene_flow=scene_flow)
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, frame='w')
    # View-synthesis given the projected reference points
    return tf.grid_sample(ref_image, ref_coords, mode=mode,
                          padding_mode=padding_mode, align_corners=align_corners)


def view_synthesis_generic(ref_image, depth, ref_cam, cam,
                           mode='bilinear', padding_mode='zeros',  align_corners=True,
                           progress=0.0):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor
        Reference image to be warped [B,3,H,W]
    depth : torch.Tensor
        Depth map from the original image [B,1,H,W]
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation
    align_corners : bool
        Corner alignment for grid sampling
    progress : float
        Training process (percentage)

    Returns
    -------
    ref_warped : torch.Tensor
        Warped reference image in the original frame of reference [B,3,H,W]
    """
    assert depth.shape[1] == 1, 'Depth map should have C=1'
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, progress=progress, frame='w')
    # View-synthesis given the projected reference points
    return tf.grid_sample(ref_image, ref_coords, mode=mode,
                          padding_mode=padding_mode, align_corners=align_corners)
