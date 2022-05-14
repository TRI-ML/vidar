# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from copy import deepcopy

import numpy as np

from vidar.utils.data import keys_in
from vidar.utils.decorators import iterate1


def duplicate_sample(sample, keys):
    """
    Duplicates sample images and contexts to preserve their un-augmented versions.

    Parameters
    ----------
    sample : Dict
        Input sample

    Returns
    -------
    sample : Dict
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
    depth : np.Array
        Depth map to be masked
    num_points : Int
        Number of input depth points that should be kept at each iteration
    Returns
    -------
    depth : np.Array
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
    depth : np.Array
        Depth map to be masked
    percentage : Tuple
        Min/Max percentages to be maintained (min, max)
    Returns
    -------
    depth : np.Array
        Masked depth map (modification done in-place!)
    """
    # Find probability of maintaining
    rnd = np.random.uniform(low=percentage[0], high=percentage[1], size=1)[0]
    # Mask depth map
    depth[np.random.rand(*depth.shape) > rnd] = 0.0
    # Return depth map
    return depth


def clip_depth(sample, max_value):
    """Clip depth map to a maximum range"""
    for i in range(len(sample)):
        if 'depth' in sample[i]:
            for ctx in sample[i]['depth'].keys():
                sample[i]['depth'][ctx][sample[i]['depth'][ctx] > max_value] = max_value
    return sample


def mask_depth_range(sample, depth_range):
    """Mask out depth map within a range"""
    for i in range(len(sample)):
        if 'depth' in sample[i]:
            for ctx in sample[i]['depth'].keys():
                sample[i]['depth'][ctx][sample[i]['depth'][ctx] < depth_range[0]] = 0.0
                sample[i]['depth'][ctx][sample[i]['depth'][ctx] > depth_range[1]] = 0.0
    return sample
