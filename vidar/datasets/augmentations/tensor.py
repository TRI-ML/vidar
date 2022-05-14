# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
import torchvision.transforms as transforms

from vidar.utils.decorators import iterate1


@iterate1
def to_tensor(matrix, tensor_type='torch.FloatTensor'):
    """Casts a matrix to a torch.Tensor"""
    return torch.Tensor(matrix).type(tensor_type)


@iterate1
def to_tensor_image(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return transform(image).type(tensor_type)


@iterate1
def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : Dict
        Input sample
    tensor_type : String
        Type of tensor we are casting to

    Returns
    -------
    sample : Dict
        Sample with keys cast as tensors
    """
    # Convert using torchvision
    keys = ['rgb', 'mask', 'input_depth', 'depth', 'disparity',
            'optical_flow', 'scene_flow']
    for key_sample, val_sample in sample.items():
        for key in keys:
            if key in key_sample:
                sample[key_sample] = to_tensor_image(val_sample, tensor_type)
    # Convert from numpy
    keys = ['intrinsics', 'extrinsics', 'pose', 'pointcloud', 'semantic']
    for key_sample, val_sample in sample.items():
        for key in keys:
            if key in key_sample:
                sample[key_sample] = to_tensor(val_sample, tensor_type)
    # Return converted sample
    return sample
