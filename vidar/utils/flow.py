# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch

from vidar.utils.data import make_list, get_from_dict, update_dict
from vidar.utils.flow_triangulation_support import bearing_grid, mult_rotation_bearing, triangulation
from vidar.utils.tensor import pixel_grid, norm_pixel_grid, unnorm_pixel_grid, grid_sample
from vidar.utils.types import is_dict, is_list


def warp_from_coords(tensor, coords, mode='bilinear', padding_mode='zeros'):
    """Warp an image from a coordinate map"""
    # Sample grid from data with coordinates
    warp = grid_sample(tensor, coords.permute(0, 2, 3, 1), mode=mode, padding_mode=padding_mode)
    # Returned warped tensor
    return warp


def coords_from_optical_flow(optflow):
    """Get coords from optical flow"""
    # Create coordinate with optical flow
    coords = pixel_grid(optflow, device=optflow) + optflow
    # Normalize and return coordinate grid
    return norm_pixel_grid(coords)


def warp_depth_from_motion(ctx_depth, ctx_cam, tgt_depth, tgt_cam,
                           tgt_scnflow=None, tgt_world_scnflow=None,
                           ctx_scnflow=None, ctx_world_scnflow=None):
    """Warp depth map given motion between two cameras"""
    if is_list(ctx_depth) and is_list(tgt_depth):
        return [warp_depth_from_motion(
            ctx_depth[i], ctx_cam, tgt_depth[i], tgt_cam,
            None if tgt_scnflow is None else tgt_scnflow[i],
            None if tgt_world_scnflow is None else tgt_world_scnflow[i],
            None if ctx_scnflow is None else ctx_scnflow[i],
            None if ctx_world_scnflow is None else ctx_world_scnflow[i],
        ) for i in range(len(ctx_depth))]
    ctx_depth_warped = reproject_depth_from_motion(
        ctx_depth, ctx_cam, tgt_cam, ctx_scnflow=ctx_scnflow, ctx_world_scnflow=ctx_world_scnflow)
    return warp_from_motion(ctx_depth_warped, ctx_cam, tgt_depth, tgt_cam, tgt_scnflow, tgt_world_scnflow)


def reproject_depth_from_motion(ctx_depth, ctx_cam, tgt_cam, ctx_scnflow=None, ctx_world_scnflow=None):
    """Reproject depth map given motion between two cameras"""
    ctx_points = ctx_cam.reconstruct_depth_map(
        ctx_depth, to_world=True, scene_flow=ctx_scnflow, world_scene_flow=ctx_world_scnflow)
    return tgt_cam.project_points(ctx_points, from_world=True, return_z=True)[1]


def warp_from_motion(ctx_rgb, ctx_cam, tgt_depth, tgt_cam, tgt_scnflow=None, tgt_world_scnflow=None):
    """Warp image given motion between two cameras"""
    tgt_points = tgt_cam.reconstruct_depth_map(
        tgt_depth, to_world=True, scene_flow=tgt_scnflow, world_scene_flow=tgt_world_scnflow)
    return warp_from_coords(ctx_rgb, ctx_cam.project_points(tgt_points, from_world=True).permute(0, 3, 1, 2))


def coords_from_motion(ctx_cam, tgt_depth, tgt_cam, tgt_scnflow=None, tgt_world_scnflow=None):
    """Get coords from motion between two cameras"""
    if is_list(ctx_cam):
        return [coords_from_motion(camera, tgt_depth, tgt_cam)
                for camera in ctx_cam]
    # If there are multiple depth maps, iterate for each
    if is_list(tgt_depth):
        return [coords_from_motion(ctx_cam, depth, tgt_cam)
                for depth in tgt_depth]
    world_points = tgt_cam.reconstruct_depth_map(
        tgt_depth, to_world=True, scene_flow=tgt_scnflow, world_scene_flow=tgt_world_scnflow)
    return ctx_cam.project_points(world_points, from_world=True).permute(0, 3, 1, 2)


def optflow_from_motion(ctx_cam, tgt_depth, tgt_cam, tgt_scnflow=None, tgt_world_scnflow=None):
    """Get optical flow from motion between two cameras"""
    if is_list(tgt_depth):
        return [optflow_from_motion(
            ctx_cam, tgt_depth[i], tgt_cam,
            tgt_scnflow[i] if tgt_scnflow is not None else None,
            tgt_world_scnflow[i] if tgt_world_scnflow is not None else None,
        ) for i in range(len(tgt_depth))]
    # coords = ctx_cam.coords_from_depth(
    #     tgt_depth, tgt_cam, scene_flow=tgt_scnflow, world_scene_flow=tgt_world_scnflow).permute(0, 3, 1, 2)
    coords = coords_from_motion(
        ctx_cam, tgt_depth, tgt_cam, tgt_scnflow=tgt_scnflow, tgt_world_scnflow=tgt_world_scnflow)
    return optflow_from_coords(coords)


def optflow_from_coords(coords):
    """Get optical flow from coordinates"""
    return unnorm_pixel_grid(coords) - pixel_grid(coords, device=coords)


def warp_from_optflow(ctx_rgb, tgt_optflow):
    """Warp image given optical flow"""
    coords = coords_from_optical_flow(tgt_optflow)
    return warp_from_coords(ctx_rgb, coords, mode='bilinear', padding_mode='zeros')


def reverse_optflow(tgt_optflow, ctx_optflow):
    """Reverse optical flow for forward/backward consistency"""
    if is_list(tgt_optflow) and is_list(ctx_optflow):
        return [reverse_optflow(tgt_optflow[i], ctx_optflow[i]) for i in range(len(tgt_optflow))]
    return - warp_from_optflow(tgt_optflow, ctx_optflow)


def mask_from_coords(coords):
    """Get mask from warped coordinates"""
    if is_list(coords):
        return [mask_from_coords(coord) for coord in coords]
    b, _, h, w = coords.shape
    mask = torch.ones((b, 1, h, w), dtype=torch.float32, device=coords.device, requires_grad=False)
    mask = warp_from_coords(mask, coords, mode='nearest', padding_mode='zeros')
    return mask.bool()


def depth_from_optflow(rgb, intrinsics, pose_context, flows,
                       residual=False, clip_range=None):
    """
    Converts pose + intrinsics + optical flow -> depth estimations

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
    residual : bool
        Return residual error with depth
    clip_range : tuple
        Depth range clipping values

    Returns
    -------
    depth : torch.Tensor
        Estimate depth map [B,1,H,W]
    """
    # Make lists if necessary
    flows = make_list(flows)
    if is_list(flows[0]):
        return [depth_from_optflow(rgb[i], intrinsics, pose_context, flows[i], residual, clip_range)
                for i in range(len(flows))]
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


def scene_flow_from_depth_optflow(depth1, depth2, cam1, cam2, optflow12, optflow21):
    """Get scene flow from depth and optical flow"""
    # Reconstruct points from depth and camera
    pts1 = cam1.reconstruct_depth_map(depth1, to_world=False)
    pts2 = cam2.reconstruct_depth_map(depth2, to_world=False)
    # Get warping coordinates from optical flow
    coords12 = coords_from_optical_flow(optflow12)
    coords21 = coords_from_optical_flow(optflow21)
    # Warp points based on coordinates
    warp12 = warp_from_coords(pts1, coords12)
    warp21 = warp_from_coords(pts2, coords21)
    # Get scene flow as the difference between warped and original points
    scnflow12 = warp12 - pts2
    scnflow21 = warp21 - pts1
    # Return scene flow
    return scnflow12, scnflow21


def residual_scene_flow_from_depth_optflow(depth1, depth2, cam1, cam2, optflow12, optflow21):
    """Get residual scene flow from depth and optical flow"""
    # Calculate scene flow
    scnflow21, scnflow12 = scene_flow_from_depth_optflow(depth1, depth2, cam1, cam2, optflow12, optflow21)
    # Calculate residual scene flow
    res_scnflow12 = residual_scene_flow(depth1, scnflow12, cam1.relative_to(cam2))
    res_scnflow21 = residual_scene_flow(depth2, scnflow21, cam2.relative_to(cam1))
    # Return residual scene flow
    return res_scnflow21, res_scnflow12


def residual_scene_flow(depth, scene_flow, cam_rel):
    """Get residual scene flow from depth, motion, and scene flow"""
    # If depth and scene flow are lists, return residual for each one
    if is_list(depth) and is_list(scene_flow):
        return [residual_scene_flow(d, sf, cam_rel) for d, sf in zip(depth, scene_flow)]
    pts_scn = cam_rel.reconstruct_depth_map(depth, scene_flow=scene_flow, to_world=False)
    pts_mot = cam_rel.reconstruct_depth_map(depth, to_world=True)
    return pts_scn - pts_mot


def to_world_scene_flow(cam, depth, scene_flow):
    """Convert scene flow to world coordinates"""
    pts = cam.reconstruct_depth_map(depth, to_world=True)
    pts_scnflow = cam.reconstruct_depth_map(depth, scene_flow=scene_flow, to_world=True)
    return pts_scnflow - pts


def fwd_bwd_optflow_consistency_check(fwd_flow, bwd_flow, alpha=0.1, beta=0.5):
    """Forward/backward consistency check for optical flow, to produce occlusion masks"""
    if is_list(fwd_flow) and is_list(bwd_flow):
        fwd_bwd = [fwd_bwd_optflow_consistency_check(fwd_flow[i], bwd_flow[i], alpha, beta)
                   for i in range(len(fwd_flow))]
        return [fb[0] for fb in fwd_bwd], [fb[1] for fb in fwd_bwd]

    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)

    warped_bwd_flow = warp_from_optflow(bwd_flow, fwd_flow)
    warped_fwd_flow = warp_from_optflow(fwd_flow, bwd_flow)

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd < threshold).unsqueeze(1)
    bwd_occ = (diff_bwd < threshold).unsqueeze(1)

    return fwd_occ, bwd_occ


def warp_optflow_dict(rgb, optflow, valid=None, keys=None):
    """Warp RGB images based on optical flow dictionary"""
    if keys is None:
        keys = {key: list(val.keys()) for key, val in optflow.items()}
    warps = {}
    for tgt in keys:
        update_dict(warps, tgt)
        for ctx in keys[tgt]:
            warps[tgt][ctx] = warp_from_optflow(rgb[ctx], optflow[tgt][ctx])
            if valid is not None:
                warps[tgt][ctx] *= valid[tgt][ctx]
    return warps


def warp_motion_dict(rgb, depth, cams, scnflow=None, world_scnflow=None, valid=None, keys=None):
    """Warp RGB images based on motion dictionary"""
    if keys is None and scnflow is not None:
        keys = {key: list(val.keys()) for key, val in scnflow.items()}
    if keys is None and world_scnflow is not None:
        keys = {key: list(val.keys()) for key, val in world_scnflow.items()}
    if keys is None and scnflow is None and world_scnflow is None:
        keys = {key: [k for k in cams.keys() if abs(k[0] - key[0]) == 1 and k[1] == key[1]] for key in cams.keys()}
    warps = {}
    for tgt in keys:
        update_dict(warps, tgt)
        for ctx in keys[tgt]:
            warps[tgt][ctx] = warp_from_motion(
                rgb[ctx], cams[ctx],
                depth[tgt][ctx] if is_dict(depth[tgt]) else depth[tgt], cams[tgt],
                tgt_scnflow=get_from_dict(scnflow, tgt, ctx),
                tgt_world_scnflow=get_from_dict(world_scnflow, tgt, ctx)
            )
            if valid is not None:
                warps[tgt][ctx] *= valid[tgt][ctx]
    return warps


def warped_depth_dict(depth, cams, scnflow=None, world_scnflow=None, valid=None, keys=None):
    """Warp depth maps based on motion dictionary"""
    if keys is None and scnflow is not None:
        keys = {key: list(val.keys()) for key, val in scnflow.items()}
    if keys is None and world_scnflow is not None:
        keys = {key: list(val.keys()) for key, val in world_scnflow.items()}
    warps = {}
    for tgt in keys:
        update_dict(warps, tgt)
        for ctx in keys[tgt]:
            warps[tgt][ctx] = warp_depth_from_motion(
                depth[ctx], cams[ctx], depth[tgt], cams[tgt],
                tgt_scnflow=get_from_dict(scnflow, tgt, ctx),
                ctx_scnflow=get_from_dict(scnflow, ctx, tgt),
                tgt_world_scnflow=get_from_dict(world_scnflow, tgt, ctx),
                ctx_world_scnflow=get_from_dict(world_scnflow, ctx, tgt),
            )
            if valid is not None:
                warps[tgt][ctx] *= valid[tgt][ctx]
    return warps


def reverse_optflow_dict(optflow, valid=None, keys=None):
    """Reverse optical flow dictionary"""
    if keys is None:
        keys = {key: list(val.keys()) for key, val in optflow.items()}
    reverse = {}
    for tgt in keys:
        update_dict(reverse, tgt)
        for ctx in keys[tgt]:
            reverse[tgt][ctx] = reverse_optflow(optflow[ctx][tgt], optflow[tgt][ctx])
            if valid is not None:
                reverse[tgt][ctx] *= valid[tgt][ctx]
    return reverse


def optflow_from_motion_dict(depth, cams, scnflow=None, world_scnflow=None, valid=None, keys=None):
    """Get opical flow from motion dictionary"""
    if keys is None and scnflow is not None:
        keys = {key: list(val.keys()) for key, val in scnflow.items()}
    if keys is None and world_scnflow is not None:
        keys = {key: list(val.keys()) for key, val in world_scnflow.items()}
    optflow = {}
    for tgt in keys:
        update_dict(optflow, tgt)
        for ctx in keys[tgt]:
            optflow[tgt][ctx] = optflow_from_motion(
                cams[ctx], depth[tgt], cams[tgt],
                tgt_scnflow=get_from_dict(scnflow, tgt, ctx),
                tgt_world_scnflow=get_from_dict(world_scnflow, tgt, ctx),
            )
            if valid is not None:
                optflow[tgt][ctx] *= valid[tgt][ctx]
    return optflow


def triangulated_depth_dict(optflow, cams, valid=None, keys=None):
    """Triangulage depth maps from optical flow dictionary"""
    if keys is None:
        keys = {key: list(val.keys()) for key, val in optflow.items()}
    depth = {}
    for tgt in keys:
        update_dict(depth, tgt)
        for ctx in keys[tgt]:
            depth[tgt][ctx] = depth_from_optflow(
                optflow[tgt][ctx], cams[ctx].K, cams[ctx].relative_to(cams[tgt]).Twc.T, [optflow[tgt][ctx]])
            if valid is not None:
                depth[tgt][ctx] *= valid[tgt][ctx]
    return depth


def scnflow_from_optflow_dict(optflow, depth, cams, keys=None, valid=None, to_world=True):
    """Get scene flow from optical flow dictionary"""
    if keys is None:
        keys = {key: list(val.keys()) for key, val in optflow.items()}
    scnflow = {}
    for tgt in keys:
        update_dict(scnflow, tgt)
        for ctx in keys[tgt]:
            if ctx not in scnflow.keys() or tgt not in scnflow[ctx]:
                update_dict(scnflow, ctx)
                scnflow[ctx][tgt], scnflow[tgt][ctx] = residual_scene_flow_from_depth_optflow(
                    depth[tgt], depth[ctx], cams[tgt], cams[ctx], optflow[ctx][tgt], optflow[tgt][ctx])
                if to_world:
                    scnflow[tgt][ctx] = to_world_scene_flow(cams[tgt], depth[tgt], scnflow[tgt][ctx])
                    scnflow[ctx][tgt] = to_world_scene_flow(cams[ctx], depth[ctx], scnflow[ctx][tgt])
                if valid is not None:
                    scnflow[tgt][ctx] *= valid[tgt][ctx]
                    scnflow[ctx][tgt] *= valid[ctx][tgt]
    return scnflow
