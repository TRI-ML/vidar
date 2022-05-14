# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as tfn

from vidar.utils.data import make_list
from vidar.utils.flow_triangulation_support import bearing_grid, mult_rotation_bearing, triangulation
from vidar.utils.tensor import pixel_grid, norm_pixel_grid, unnorm_pixel_grid
from vidar.utils.types import is_list


def warp_from_coords(tensor, coords, mode='bilinear',
                     padding_mode='zeros', align_corners=True):
    """
    Warp an image from a coordinate map

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor for warping [B,?,H,W]
    coords : torch.Tensor
        Warping coordinates [B,2,H,W]
    mode : String
        Warping mode
    padding_mode : String
        Padding mode
    align_corners : Bool
        Align corners flag

    Returns
    -------
    warp : torch.Tensor
        Warped tensor [B,?,H,W]
    """
    # Sample grid from data with coordinates
    warp = tfn.grid_sample(tensor, coords.permute(0, 2, 3, 1),
                           mode=mode, padding_mode=padding_mode,
                           align_corners=align_corners)
    # Returned warped tensor
    return warp


def coords_from_optical_flow(optflow):
    """
    Get warping coordinates from optical flow
    Parameters
    ----------
    optflow : torch.Tensor
        Input optical flow tensor [B,2,H,W]

    Returns
    -------
    coords : torch.Tensor
        Warping coordinates [B,2,H,W]
    """
    # Create coordinate with optical flow
    coords = pixel_grid(optflow, device=optflow) + optflow
    # Normalize and return coordinate grid
    return norm_pixel_grid(coords)


def warp_depth_from_motion(ref_depth, tgt_depth, ref_cam):
    """
    Warp depth map using motion (depth + ego-motion) information

    Parameters
    ----------
    ref_depth : torch.Tensor
        Reference depth map [B,1,H,W]
    tgt_depth : torch.Tensor
        Target depth map [B,1,H,W]
    ref_cam : Camera
        Reference camera

    Returns
    -------
    warp : torch.Tensor
        Warped depth map [B,1,H,W]
    """
    ref_depth = reproject_depth_from_motion(ref_depth, ref_cam)
    return warp_from_motion(ref_depth, tgt_depth, ref_cam)


def reproject_depth_from_motion(ref_depth, ref_cam):
    """
    Calculate reprojected depth from motion (depth + ego-motion) information

    Parameters
    ----------
    ref_depth : torch.Tensor
        Reference depth map [B,1,H,W]
    ref_cam : Camera
        Reference camera

    Returns
    -------
    coords : torch.Tensor
        Warping coordinates from reprojection [B,2,H,W]
    """
    ref_points = ref_cam.reconstruct_depth_map(ref_depth, to_world=True)
    return ref_cam.project_points(ref_points, from_world=False, return_z=True)[1]


def warp_from_motion(ref_rgb, tgt_depth, ref_cam):
    """
    Warp image using motion (depth + ego-motion) information

    Parameters
    ----------
    ref_rgb : torch.Tensor
        Reference image [B,3,H,W]
    tgt_depth : torch.Tensor
        Target depth map [B,1,H,W]
    ref_cam : Camera
        Reference camera

    Returns
    -------
    warp : torch.Tensor
        Warped image [B,3,H,W]
    """
    tgt_points = ref_cam.reconstruct_depth_map(tgt_depth, to_world=False)
    return warp_from_coords(ref_rgb, ref_cam.project_points(tgt_points, from_world=True).permute(0, 3, 1, 2))


def coords_from_motion(ref_camera, tgt_depth, tgt_camera):
    """
    Get coordinates from motion (depth + ego-motion) information

    Parameters
    ----------
    ref_camera : Camera
        Reference camera
    tgt_depth : torch.Tensor
        Target depth map [B,1,H,W]
    tgt_camera : Camera
        Target camera

    Returns
    -------
    coords : torch.Tensor
        Warping coordinates [B,2,H,W]
    """
    if is_list(ref_camera):
        return [coords_from_motion(camera, tgt_depth, tgt_camera)
                for camera in ref_camera]
    # If there are multiple depth maps, iterate for each
    if is_list(tgt_depth):
        return [coords_from_motion(ref_camera, depth, tgt_camera)
                for depth in tgt_depth]
    world_points = tgt_camera.reconstruct_depth_map(tgt_depth, to_world=True)
    return ref_camera.project_points(world_points, from_world=True).permute(0, 3, 1, 2)


def optflow_from_motion(ref_camera, tgt_depth):
    """
    Get optical flow from motion (depth + ego-motion) information

    Parameters
    ----------
    ref_camera : Camera
        Reference camera
    tgt_depth : torch.Tensor
        Target depth map

    Returns
    -------
    optflow : torch.Tensor
        Optical flow map [B,2,H,W]
    """
    coords = ref_camera.coords_from_depth(tgt_depth).permute(0, 3, 1, 2)
    return optflow_from_coords(coords)


def optflow_from_coords(coords):
    """
    Get optical flow from coordinates

    Parameters
    ----------
    coords : torch.Tensor
        Input warping coordinates [B,2,H,W]

    Returns
    -------
    optflow : torch.Tensor
        Optical flow map [B,2,H,W]
    """
    return unnorm_pixel_grid(coords) - pixel_grid(coords, device=coords)


def warp_from_optflow(ref_rgb, tgt_optflow):
    """
    Warp image using optical flow information

    Parameters
    ----------
    ref_rgb : torch.Tensor
        Reference image [B,3,H,W]
    tgt_optflow : torch.Tensor
        Target optical flow [B,2,H,W]

    Returns
    -------
    warp : torch.Tensor
        Warped image [B,3,H,W]
    """
    coords = coords_from_optical_flow(tgt_optflow)
    return warp_from_coords(ref_rgb, coords, align_corners=True,
                            mode='bilinear', padding_mode='zeros')


def reverse_optflow(tgt_optflow, ref_optflow):
    """
    Reverse optical flow

    Parameters
    ----------
    tgt_optflow : torch.Tensor
        Target optical flow [B,2,H,W]
    ref_optflow : torch.Tensor
        Reference optical flow [B,2,H,W]

    Returns
    -------
    optflow : torch.Tensor
        Reversed optical flow [B,2,H,W]
    """
    return - warp_from_optflow(tgt_optflow, ref_optflow)


def mask_from_coords(coords, align_corners=True):
    """
    Get overlap mask from coordinates

    Parameters
    ----------
    coords : torch.Tensor
        Warping coordinates [B,2,H,W]
    align_corners : Bool
        Align corners flag

    Returns
    -------
    mask : torch.Tensor
        Overlap mask [B,1,H,W]
    """
    if is_list(coords):
        return [mask_from_coords(coord) for coord in coords]
    b, _, h, w = coords.shape
    mask = torch.ones((b, 1, h, w), dtype=torch.float32, device=coords.device, requires_grad=False)
    mask = warp_from_coords(mask, coords, mode='nearest', padding_mode='zeros', align_corners=True)
    return mask.bool()


def depth_from_optflow(rgb, intrinsics, pose_context, flows,
                       residual=False, clip_range=None):
    """
    Get depth from optical flow + camera information

    Parameters
    ----------
    rgb : torch.Tensor
        Base image [B,3,H,W]
    intrinsics : torch.Tensor
        Camera intrinsics [B,3,3]
    pose_context : torch.Tensor or list[torch.Tensor]
        List of relative context camera poses [B,4,4]
    flows : torch.Tensor or list[torch.Tensor]
        List of target optical flows [B,2,H,W]
    residual : Bool
        Return residual error with depth
    clip_range : Tuple
        Depth range clipping values

    Returns
    -------
    depth : torch.Tensor
        Depth map [B,1,H,W]
    """
    # Make lists if necessary
    flows = make_list(flows)
    pose_context = make_list(pose_context)
    # Extract rotations and translations
    rotations = [p[:, :3, :3] for p in pose_context]
    translations = [p[:, :3, -1] for p in pose_context]
    # Get bearings
    bearings = bearing_grid(rgb, intrinsics).to(rgb.device)
    rot_bearings = [mult_rotation_bearing(rotation, bearings)
                    for rotation in rotations]
    # Return triangulation results
    return triangulation(rot_bearings, translations, flows, intrinsics,
                         clip_range=clip_range, residual=residual)
