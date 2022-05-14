# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from copy import deepcopy

import numpy as np

from vidar.utils.data import keys_with
from vidar.utils.decorators import iterate1


@iterate1
def crop_pil(image, borders):
    """
    Crop a PIL Image

    Parameters
    ----------
    image : PIL Image
        Input image
    borders : Tuple
        Borders used for cropping (left, top, right, lower)

    Returns
    -------
    image : PIL.Image
        Cropped image
    """
    return image.crop(borders)


@iterate1
def crop_npy(depth, borders):
    """
    Crop a numpy depth map

    Parameters
    ----------
    depth : np.Array
        Input numpy array
    borders : Tuple
        Borders used for cropping

    Returns
    -------
    image : np.array
        Cropped numpy array
    """
    # Return if depth value is None
    return depth[borders[1]:borders[3], borders[0]:borders[2]]


@iterate1
def crop_intrinsics(intrinsics, borders):
    """
    Crop camera intrinsics matrix

    Parameters
    ----------
    intrinsics : np.Array
        Original intrinsics matrix [3,3]
    borders : Tuple
        Borders used for cropping
    Returns
    -------
    intrinsics : np.Array
        Cropped intrinsics matrix [3,3]
    """
    intrinsics = np.copy(intrinsics)
    intrinsics[0, 2] -= borders[0]
    intrinsics[1, 2] -= borders[1]
    return intrinsics


def crop_sample_input(sample, borders):
    """
    Crops the input information of a sample

    Parameters
    ----------
    sample : Dict
        Dictionary with sample values
    borders : Tuple
        Borders used for cropping

    Returns
    -------
    sample : Dict
        Cropped sample
    """
    # Intrinsics
    for key in keys_with(sample, 'intrinsics', without='raw'):
        # Create copy of full intrinsics
        if f'raw_{key}' not in sample.keys():
            sample[f'raw_{key}'] = deepcopy(sample[key])
        sample[key] = crop_intrinsics(sample[key], borders)
    # RGB
    for key in keys_with(sample, 'rgb', without='raw'):
        sample[key] = crop_pil(sample[key], borders)
    # Mask
    for key in keys_with(sample, 'mask', without='raw'):
        sample[key] = crop_pil(sample[key], borders)
    # Input depth
    for key in keys_with(sample, 'input_depth'):
        sample[key] = crop_npy(sample[key], borders)
    # Return cropped sample
    return sample


def crop_sample_supervision(sample, borders):
    """
    Crops the output information of a sample

    Parameters
    ----------
    sample : Dict
        Dictionary with sample values
    borders : Tuple
        Borders used for cropping

    Returns
    -------
    sample : Dict
        Cropped sample
    """
    for key in keys_with(sample, 'depth', without='input_depth'):
        sample[key] = crop_npy(sample[key], borders)
    for key in keys_with(sample, 'optical_flow'):
        sample[key] = crop_npy(sample[key], borders)
    for key in keys_with(sample, 'scene_flow'):
        sample[key] = crop_npy(sample[key], borders)
    # Return cropped sample
    return sample


def crop_sample(sample, borders):
    """
    Crops a sample, including image, intrinsics and depth maps.

    Parameters
    ----------
    sample : Dict
        Dictionary with sample values
    borders : Tuple
        Borders used for cropping

    Returns
    -------
    sample : Dict
        Cropped sample
    """
    # Crop input information
    sample = crop_sample_input(sample, borders)
    # Crop output information
    sample = crop_sample_supervision(sample, borders)
    # Return cropped sample
    return sample


