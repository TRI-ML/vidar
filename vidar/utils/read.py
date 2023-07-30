# Copyright 2023 Toyota Research Institute.  All rights reserved.

import pickle as pkl

import numpy as np
from PIL import Image, ImageOps

from efm_datasets.utils.decorators import iterate1


def read_pickle(filename):
    """Read pickle file"""
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    return pkl.load(open(filename, 'rb'))


@iterate1
def read_image(filename, mode='RGB', invert=False):
    """
    Read an image using PIL

    Parameters
    ----------
    filename : str
        Path to the image

    Returns
    -------
    image : PIL Image
        Loaded image
    """
    image = Image.open(filename)
    if mode != '':
        image = image.convert(mode)
    if invert:
        image = ImageOps.invert(image)
    return image


@iterate1
def read_numpy(filename, key=None):
    """Read numpy file"""
    depth = np.load(filename)
    if key is not None:
        depth = depth[key]
    return depth


@iterate1
def read_depth(filename, div=256., key='depth'):
    """
    Load a depth map from file
    Parameters
    ----------
    file : str
        Depth map filename (.npz or .png or .dpt)

    Returns
    -------
    depth : np.array
        Depth map (invalid pixels are 0) [H,W]
    """
    # If loading a .npz array
    if filename.endswith('npz'):
        return np.load(filename)[key]
    # If loading a .png image
    elif filename.endswith('png'):
        depth_png = np.array(read_image(filename, mode=''), dtype=int)
        # assert (np.max(depth_png) > 255), 'Wrong .png depth file'
        return depth_png.astype(float) / div
    elif filename.endswith('exr'):
        exrfile = exr.InputFile(filename) # .as_posix())
        raw_bytes = exrfile.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(raw_bytes, dtype=np.float32)
        height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
        width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
        depth = np.reshape(depth, (height, width))
        return depth


def read_optical_flow(filename):
    """Read optical flow file"""
    if filename.endswith('flo'):
        with open(filename, 'rb') as f:
            if np.fromfile(f, np.float32, count=1) == 202021.25:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                return np.resize(data, (int(h), int(w), 2))
    else:
        raise ValueError('Invalid optical flow file')


@iterate1
def read_float3(filename):
    """Read float3 file"""
    f = open(filename, 'rb')
    if (f.readline().decode('utf-8')) != 'float\n':
        print('asdfasdfasdf')
    dim = int(f.readline())
    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d
    dims = list(reversed(dims))
    return np.fromfile(f, np.float32, count).reshape(dims)
