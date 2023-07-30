# Copyright 2023 Toyota Research Institute.  All rights reserved.

import os
import pickle as pkl

import cv2
import numpy as np
import torchvision.transforms as transforms

from vidar.utils.decorators import multi_write
from vidar.utils.types import is_tensor, is_numpy


def create_folder(filename):
    """Create folder if it doesn't exist"""
    if '/' in filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)


def write_pickle(filename, data):
    """Write data to pickle file"""
    create_folder(filename)
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'
    pkl.dump(data, open(filename, 'wb'))


def write_npz(filename, data):
    """Write data to npz file"""
    # Create folder if it doesn't exist
    create_folder(filename)
    # Save npz
    np.savez_compressed(filename, **data)


def write_npy(filename, data):
    """Write data to npy file"""
    # Create folder if it doesn't exist
    create_folder(filename)
    # Save npy
    np.save(filename, data)


@multi_write
@multi_write
def write_image(filename, image):
    """
    Write an image to file

    Parameters
    ----------
    filename : str
        File where image will be saved
    image : np.array [H,W,3]
        RGB image
    """
    # Create folder if it doesn't exist
    create_folder(filename)
    # If image is a tensor
    if is_tensor(image):
        if len(image.shape) == 4:
            image = image[0]
        image = image.float().detach().cpu().numpy().transpose(1, 2, 0)
        cv2.imwrite(filename, image[:, :, ::-1] * 255)
    # If image is a numpy array
    elif is_numpy(image):
        cv2.imwrite(filename, image[:, :, ::-1] * 255)
    # Otherwise, assume it's a PIL image
    else:
        image.save(filename)


@multi_write
def write_depth(filename, depth, intrinsics=None):
    """
    Write a depth map to file, and optionally its corresponding intrinsics.

    Parameters
    ----------
    filename : str
        File where depth map will be saved (.npz or .png)
    depth : np.array or torch.Tensor
        Depth map [H,W]
    intrinsics : np.array
        Optional camera intrinsics matrix [3,3]
    """
    # Create folder if it doesn't exist
    create_folder(filename)
    # If depth is a tensor
    if is_tensor(depth):
        depth = depth.detach().squeeze().cpu().numpy()
    # If intrinsics is a tensor
    if is_tensor(intrinsics):
        intrinsics = intrinsics.detach().cpu().numpy()
    # If we are saving as a .npz
    if filename.endswith('.npz'):
        np.savez_compressed(filename, depth=depth, intrinsics=intrinsics)
    # If we are saving as a .png
    elif filename.endswith('.png'):
        depth = transforms.ToPILImage()((depth * 256).astype(np.int32))
        depth.save(filename)
    # Something is wrong
    else:
        raise NotImplementedError('Depth filename not valid.')


@multi_write
def write_normals(filename, normals, intrinsics=None):
    """
    Write a depth map to file, and optionally its corresponding intrinsics.

    Parameters
    ----------
    filename : str
        File where depth map will be saved (.npz or .png)
    normals : np.array or torch.Tensor
        Normals map [H,W]
    intrinsics : np.array
        Optional camera intrinsics matrix [3,3]
    """
    # Create folder if it doesn't exist
    create_folder(filename)
    # If normals is a tensor
    if is_tensor(normals):
        normals = normals.detach().squeeze().cpu().numpy()
    # If intrinsics is a tensor
    if is_tensor(intrinsics):
        intrinsics = intrinsics.detach().cpu().numpy()
    # If we are saving as a .npz
    if filename.endswith('.npz'):
        np.savez_compressed(filename, normals=normals, intrinsics=intrinsics)
    # Something is wrong
    else:
        raise NotImplementedError('Normals filename not valid.')


@multi_write
def write_optical_flow(filename, optflow):
    """
    Write a depth map to file, and optionally its corresponding intrinsics.

    Parameters
    ----------
    filename : str
        File where depth map will be saved (.npz or .png)
    optflow : np.array or torch.Tensor
        Optical flow map [H,W]
    """
    # Create folder if it doesn't exist
    create_folder(filename)
    # If depth is a tensor
    if is_tensor(optflow):
        optflow = optflow.detach().squeeze().cpu().numpy()
    # If we are saving as a .npz
    if filename.endswith('.npz'):
        np.savez_compressed(filename, optflow=optflow)
    # Something is wrong
    else:
        raise NotImplementedError('Optical flow filename not valid.')
