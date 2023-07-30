# Copyright 2023 Toyota Research Institute.  All rights reserved.

from copy import deepcopy
from math import pi

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from vidar.utils.data import keys_in
from vidar.utils.decorators import iterate1


def duplicate_sample(sample, keys):
    """
    Duplicates sample images and contexts to preserve their unaugmented versions.

    Parameters
    ----------
    sample : dict
        Input sample

    Returns
    -------
    sample : dict
        Sample including [+"_original"] keys with copies of images and contexts.
    """
    for key in keys_in(sample, keys):
        sample[f'raw_{key}'] = deepcopy(sample[key])
    # Return duplicated sample
    return sample


@iterate1
def mask_depth_number(depth, num_points):
    """
    Mask depth map to remove valid pixels given the target number of points to keep.

    Parameters
    ----------
    depth : np.array
        Depth map to be masked
    num_points : int
        Number of input depth points that should be kept at each iteration
    Returns
    -------
    depth : np.array
        Masked depth map (modification done in-place!)
    """
    # Find probability of maintaining
    total_points = depth.shape[0] * depth.shape[1]
    rnd = np.random.rand(depth.shape[0], depth.shape[1])
    percentile = 100 * num_points / total_points
    # Mask depth map
    mask = rnd < np.percentile(rnd, q=100 - percentile)
    depth[mask] = 0.0
    # Return depth map
    return depth


@iterate1
def mask_depth_percentage(depth, percentage):
    """
    Mask depth map to remove valid pixels given a range of percentages.

    Parameters
    ----------
    depth : np.array
        Depth map to be masked
    percentage : tuple (min, max)
        Min/Max percentages to be maintained
    Returns
    -------
    depth : np.array
        Masked depth map (modification done in-place!)
    """
    # Find probability of maintaining
    rnd = np.random.uniform(low=percentage[0], high=percentage[1], size=1)[0]
    # Mask depth map
    depth[np.random.rand(*depth.shape) > rnd] = 0.0
    # Return depth map
    return depth


def clip_depth(sample, max_value):
    for i in range(len(sample)):
        if 'depth' in sample[i]:
            for ctx in sample[i]['depth'].keys():
                sample[i]['depth'][ctx][sample[i]['depth'][ctx] > max_value] = max_value
    return sample


def mask_depth_range(sample, depth_range):
    for i in range(len(sample)):
        if 'depth' in sample[i]:
            for ctx in sample[i]['depth'].keys():
                sample[i]['depth'][ctx][sample[i]['depth'][ctx] < depth_range[0]] = 0.0
                sample[i]['depth'][ctx][sample[i]['depth'][ctx] > depth_range[1]] = 0.0
    return sample


def sample_homography(
        shape, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=100, n_angles=100, scaling_amplitude=0.1, perspective_amplitude=0.4,
        patch_ratio=0.8, max_angle=pi/4):
    """ Sample a random homography that includes perspective, scale, translation and rotation operations."""

    width = float(shape[1])
    hw_ratio = float(shape[0]) / float(shape[1])

    pts1 = np.stack([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], axis=0)
    pts2 = pts1.copy() * patch_ratio
    pts2[:,1] *= hw_ratio

    if perspective:

        perspective_amplitude_x = np.random.normal(0., perspective_amplitude/2, (2))
        perspective_amplitude_y = np.random.normal(0., hw_ratio * perspective_amplitude/2, (2))

        perspective_amplitude_x = np.clip(perspective_amplitude_x, -perspective_amplitude/2, perspective_amplitude/2)
        perspective_amplitude_y = np.clip(perspective_amplitude_y, hw_ratio * -perspective_amplitude/2, hw_ratio * perspective_amplitude/2)

        pts2[0,0] -= perspective_amplitude_x[1]
        pts2[0,1] -= perspective_amplitude_y[1]

        pts2[1,0] -= perspective_amplitude_x[0]
        pts2[1,1] += perspective_amplitude_y[1]

        pts2[2,0] += perspective_amplitude_x[1]
        pts2[2,1] -= perspective_amplitude_y[0]

        pts2[3,0] += perspective_amplitude_x[0]
        pts2[3,1] += perspective_amplitude_y[0]

    if scaling:

        random_scales = np.random.normal(1, scaling_amplitude/2, (n_scales))
        random_scales = np.clip(random_scales, 1-scaling_amplitude/2, 1+scaling_amplitude/2)

        scales = np.concatenate([[1.], random_scales], 0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = np.expand_dims(pts2 - center, axis=0) * np.expand_dims(
                np.expand_dims(scales, 1), 1) + center
        valid = np.arange(n_scales)  # all scales are valid except scale=1
        idx = valid[np.random.randint(valid.shape[0])]
        pts2 = scaled[idx]

    if translation:
        t_min, t_max = np.min(pts2 - [-1., -hw_ratio], axis=0), np.min([1., hw_ratio] - pts2, axis=0)
        pts2 += np.expand_dims(np.stack([np.random.uniform(-t_min[0], t_max[0]),
                                         np.random.uniform(-t_min[1], t_max[1])]),
                               axis=0)

    if rotation:
        angles = np.linspace(-max_angle, max_angle, n_angles)
        angles = np.concatenate([[0.], angles], axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul(
                np.tile(np.expand_dims(pts2 - center, axis=0), [n_angles+1, 1, 1]),
                rot_mat) + center

        valid = np.where(np.all((rotated >= [-1.,-hw_ratio]) & (rotated < [1.,hw_ratio]),
                                        axis=(1, 2)))[0]

        idx = valid[np.random.randint(valid.shape[0])]
        pts2 = rotated[idx]

    pts2[:,1] /= hw_ratio

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]
    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(np.stack(
        [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))

    homography = np.matmul(np.linalg.pinv(a_mat), p_mat).squeeze()
    homography = np.concatenate([homography, [1.]]).reshape(3,3)
    return homography

def warp_homography(sources, homography):
    """Warp features given a homography

    Parameters
    ----------
    sources: torch.tensor (1,H,W,2)
        Keypoint vector.
    homography: torch.Tensor (3,3)
        Homography.

    Returns
    -------
    warped_sources: torch.tensor (1,H,W,2)
        Warped feature vector.
    """
    _, H, W, _ = sources.shape
    warped_sources = sources.clone().squeeze()
    warped_sources = warped_sources.view(-1,2)
    warped_sources = torch.addmm(homography[:,2], warped_sources, homography[:,:2].t())
    warped_sources.mul_(1/warped_sources[:,2].unsqueeze(1))
    warped_sources = warped_sources[:,:2].contiguous().view(1,H,W,2)
    return warped_sources

