# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
import torchvision.transforms as transforms

from vidar.utils.decorators import iterate1
from vidar.utils.data import remove_nones_dict


@iterate1
@iterate1
def to_tensor(matrix, tensor_type='torch.FloatTensor'):
    """Casts a matrix to a torch.Tensor"""
    return None if matrix is None else torch.Tensor(matrix).type(tensor_type)


@iterate1
@iterate1
def to_tensor_image(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return None if image is None else transform(image).type(tensor_type)


@iterate1
def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    # Convert using torchvision
    keys = [
        'rgb', 'mask',
        'input_depth', 'depth', 'disparity',
        'optical_flow', 'scene_flow'
    ]
    for key_sample, val_sample in sample.items():
        for key in keys:
            if key in key_sample:
                sample[key_sample] = to_tensor_image(val_sample, tensor_type)
    # Convert from numpy
    keys = [
        'intrinsics', 'extrinsics', 'pose',
        'pointcloud', 'semantic', 'normals',
    ]
    for key_sample, val_sample in sample.items():
        for key in keys:
            if key in key_sample:
                sample[key_sample] = to_tensor(val_sample, tensor_type)
    # Return converted sample
    return remove_nones_dict(sample)
