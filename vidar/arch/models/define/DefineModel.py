# Copyright 2023 Toyota Research Institute.  All rights reserved.

import random
from abc import ABC

import torch

from vidar.geometry.pose import Pose
from vidar.utils.types import is_int


def augment_canonical(pose):
    """Augment canonical pose with random rotation and translation"""

    ctx = list(pose.keys())
    num = list(range(pose[0].shape[1]))

    i = random.choice(ctx)
    j = random.choice(num)

    base = Pose(pose[i][:, j]).inverse().T
    for key in ctx:
        for n in num:
            pose[key][:, n] = pose[key][:, n] @ base

    return pose


def parse_output(output, key, encode_idx, predictions):
    """Parse output from the model, given a key and the encode indices"""
    if key in output.keys():
        pred = [[val] for val in output[key]]
        predictions_key = {}
        for idx, (i, j) in enumerate(encode_idx):
            if j not in predictions_key.keys():
                predictions_key[j] = []
            predictions_key[j].append(output[key][idx])
        predictions_key = {key: [torch.stack(val, 1)]
                           for key, val in predictions_key.items()}
        predictions[key] = predictions_key
    else:
        pred = None
    return pred


def create_virtual_cameras(encode_data, n_samples=1, cam_noise=None, center_noise=None,
                           downsample=1.0, thr=0.1, tries=10, decay=0.9):
    """Create virtual cameras from the given data"""

    gt_cams = [datum['cam'] for datum in encode_data]
    gt_depths = [datum['gt_depth'] for datum in encode_data]
    gt_rgbs = [datum['rgb'] for datum in encode_data]

    if not is_int(thr):
        n = gt_rgbs[0].shape[-2] * gt_rgbs[0].shape[-1]
        thr = int(n * thr)

    # Make projected pointclouds and colors 
    pcls_proj = [cam.scaled(downsample).reconstruct_depth_map(depth, to_world=True)
                 for cam, depth in zip(gt_cams, gt_depths)]
    pcls_proj = [pcl.reshape(*pcl.shape[:2], -1) for pcl in pcls_proj]
    pcl_proj = torch.cat([pcl for pcl in pcls_proj], -1)
    clr_proj = torch.cat([rgb.reshape(*rgb.shape[:2], -1)
                         for rgb in gt_rgbs], -1)

    # Get pointcloud centers
    gt_pcl_centers = [pcl.mean(-1) for pcl in pcls_proj]

    # Create virtual cameras and projections
    virtual_data = []
    for gt_rgb, gt_depth, gt_cam, gt_pcl_center in zip(gt_rgbs, gt_depths, gt_cams, gt_pcl_centers):
        for i in range(n_samples):

            cam = gt_cam.clone()
            pcl_center = gt_pcl_center.clone()

            weight = 1.0
            rgb_proj = depth_proj = None

            # Try multiple times in case projections are empty
            for j in range(tries):

                if center_noise is not None:
                    pcl_center_noise = weight * center_noise * \
                        (2 * torch.rand_like(gt_pcl_center) - 1)
                    pcl_center = gt_pcl_center + pcl_center_noise

                if cam_noise is not None:
                    cam.look_at(pcl_center)
                    cam.Twc.translateUp(
                        weight * cam_noise[0] * (2 * random.random() - 1))
                    cam.Twc.translateLeft(
                        weight * cam_noise[1] * (2 * random.random() - 1))
                    cam.Twc.translateForward(
                        weight * cam_noise[2] * (2 * random.random() - 1))
                    cam.look_at(pcl_center)

                rgb_proj_try, depth_proj_try = cam.project_pointcloud(
                    pcl_proj, clr_proj)

                valid = (depth_proj_try > 0).sum() > thr
                if valid:
                    rgb_proj, depth_proj = rgb_proj_try, depth_proj_try
                    break
                else:
                    weight = weight * decay

            # If all tries failed, use the ground truth
            if rgb_proj is None and depth_proj is None:
                rgb_proj, depth_proj = gt_rgb, gt_depth
                cam = gt_cam.clone()

            virtual_data.append({
                'cam': cam,
                'rgb': rgb_proj.contiguous(),
                'gt_depth': depth_proj.contiguous(),
            })

    return virtual_data


