import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch

import knk_vision.vidar.vidar.ontologies.convert
from knk_vision.vidar.vidar.utils.decorators import iterate1
from knk_vision.vidar.vidar.utils.types import is_seq, is_tensor, is_dict, is_numpy, is_int, is_list



def stack_sample(sample, lidar_sample=None, radar_sample=None):
    # If there are no tensors, return empty list
    if len(sample) == 0:
        return None
    # If there is only one sensor don't do anything
    if len(sample) == 1:
        sample = sample[0]
        return sample
    # Otherwise, stack sample
    first_sample = sample[0]
    stacked_sample = {}
    for key, val in first_sample.items():
        # Global keys (do not stack)
        if key in ['idx', 'dataset_idx']:
            stacked_sample[key] = first_sample[key]
        # Meta keys
        elif key in ['meta']:
            stacked_sample[key] = {}
            for key2 in first_sample[key].keys():
                stacked_sample[key][key2] = {}
                for key3 in first_sample[key][key2].keys():
                    stacked_sample[key][key2][key3] = torch.stack(
                        [torch.tensor(s[key][key2][key3]) for s in sample], 0)
        # Stack tensors
        elif is_tensor(val):
            stacked_sample[key] = torch.stack([s[key] for s in sample], 0)
        # Stack list
        elif is_seq(first_sample[key]):
            stacked_sample[key] = []
            # Stack list of torch tensors
            if is_tensor(first_sample[key][0]):
                for i in range(len(first_sample[key])):
                    stacked_sample[key].append(
                        torch.stack([s[key][i] for s in sample], 0))
            else:
                stacked_sample[key] = [s[key] for s in sample]
        # Repeat for dictionaries
        elif is_dict(first_sample[key]):
            stacked_sample[key] = stack_sample([s[key] for s in sample])
        # Append lists
        else:
            stacked_sample[key] = [s[key] for s in sample]

    # Return stacked sample
    return stacked_sample


def merge_sample(samples):
    merged_sample = {}
    for sample in samples:
        for key, val in sample.items():
            if key not in merged_sample:
                merged_sample[key] = val
            else:
                merged_sample[key] = merge_sample([merged_sample[key], val])
    return merged_sample


def parse_crop(cfg, shape):
    borders = None
    if cfg.has('crop_borders'):
        borders = parse_crop_borders(cfg.crop_borders, shape)
    if cfg.has('crop_random'):
        if borders is None:
            borders = [0, 0, shape[1], shape[0]]
        borders = parse_crop_random(borders, cfg.crop_random)
    return borders


def parse_crop_borders(borders, shape):
    """
    Calculate borders for cropping.

    Parameters
    ----------
    borders : tuple
        Border input for parsing. Can be one of the following forms:
        (int, int, int, int): y, height, x, width
        (int, int): y, x --> y, height = image_height - y, x, width = image_width - x
        Negative numbers are taken from image borders, according to the shape argument
        Float numbers for y and x are treated as percentage, according to the shape argument,
            and in this case height and width are centered at that point.
    shape : tuple
        Image shape (image_height, image_width), used to determine negative crop boundaries

    Returns
    -------
    borders : tuple (left, top, right, bottom)
        Parsed borders for cropping
    """
    # Return full image if there are no borders to crop
    if len(borders) == 0:
        return 0, 0, shape[1], shape[0]
    # Copy borders for modification
    borders = list(borders).copy()
    # If borders are 4-dimensional
    if len(borders) == 4:
        borders = [borders[2], borders[0], borders[3], borders[1]]
        if is_int(borders[0]):
            # If horizontal cropping is integer (regular cropping)
            borders[0] += shape[1] if borders[0] < 0 else 0
            borders[2] += shape[1] if borders[2] <= 0 else borders[0]
        else:
            # If horizontal cropping is float (center cropping)
            center_w, half_w = borders[0] * shape[1], borders[2] / 2
            borders[0] = int(center_w - half_w)
            borders[2] = int(center_w + half_w)
        if is_int(borders[1]):
            # If vertical cropping is integer (regular cropping)
            borders[1] += shape[0] if borders[1] < 0 else 0
            borders[3] += shape[0] if borders[3] <= 0 else borders[1]
        else:
            # If vertical cropping is float (center cropping)
            center_h, half_h = borders[1] * shape[0], borders[3] / 2
            borders[1] = int(center_h - half_h)
            borders[3] = int(center_h + half_h)
    # If borders are 2-dimensional
    elif len(borders) == 2:
        borders = [borders[1], borders[0]]
        if is_int(borders[0]):
            # If cropping is integer (regular cropping)
            borders = (max(0, borders[0]),
                       max(0, borders[1]),
                       shape[1] + min(0, borders[0]),
                       shape[0] + min(0, borders[1]))
        else:
            # If cropping is float (center cropping)
            center_w, half_w = borders[0] * shape[1], borders[1] / 2
            center_h, half_h = borders[0] * shape[0], borders[1] / 2
            borders = (int(center_w - half_w), int(center_h - half_h),
                       int(center_w + half_w), int(center_h + half_h))
    # Otherwise, invalid
    else:
        raise NotImplementedError('Crop tuple must have 2 or 4 values.')
    # Assert that borders are valid
    assert 0 <= borders[0] < borders[2] <= shape[1] and \
        0 <= borders[1] < borders[3] <= shape[0], 'Crop borders {} are invalid'.format(
            borders)
    # Return updated borders
    return borders


def parse_crop_random(borders, shape):
    """
    Create borders for random cropping.
    Crops are generated anywhere in the image inside the borders

    Parameters
    ----------
    borders : tuple (left, top, right, bottom)
        Area of the image where random cropping can happen
    shape : tuple
        Cropped output shape (height, width)

    Returns
    -------
    borders : tuple
        Parsed borders for cropping (left, top, right, bottom)
    """
    # Return full borders if there is no random crop
    if len(shape) == 0:
        return borders
    shape = [s for s in shape]
    is_float = [None for _ in range(len(shape))]
    div = [None for _ in range(len(shape))]
    for i in range(len(shape)):
        if is_list(shape[i]):
            if len(shape[i]) == 3:
                div[i] = shape[i][2]
            if is_int(shape[i][0]):
                shape[i] = shape[i][0] + (shape[i][1] - shape[i][0]) * random.uniform(0, 1)
                is_float[i] = False
            else:
                shape[i] = shape[i][0] + (shape[i][1] - shape[i][0]) * random.uniform(0, 1)
                is_float[i] = True
        else:
            is_float[i] = not is_int(shape[i])
    if is_float[0]:
        shape[0] = int(shape[0] * (borders[3] - borders[0]))
    if is_float[1]:
        shape[1] = int(shape[1] * (borders[2] - borders[1]))
    # Check if random crop is valid
    assert 0 < shape[1] <= borders[2] - borders[0] and \
        0 < shape[0] <= borders[3] - \
        borders[1], 'Random crop must be smaller than the image'
    # Sample a crop
    for i in range(len(div)):
        if div[i] is not None:
            shape[i] = shape[i] // div[i] * div[i]
    x = random.randint(borders[0], borders[2] - shape[1])
    y = random.randint(borders[1], borders[3] - shape[0])
    # Return new borders
    return x, y, x + shape[1], y + shape[0]


@iterate1
def invert_pose(pose):
    """
    Inverts a transformation matrix (pose)

    Parameters
    ----------
    pose : np.array
        Input pose [4, 4]

    Returns
    -------
    inv_pose : np.array
        Inverted pose [4, 4]
    """
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = np.transpose(pose[:3, :3])
    inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
    # return np.linalg.inv(pose)
    return inv_pose


def make_relative_pose(samples):
    # Do nothing if there is no pose
    if 'pose' not in samples[0]:
        return samples
    # Get inverse current poses
    inv_pose = [invert_pose(samples[i]['pose'][0])
                for i in range(len(samples))]
    # For each camera
    for i in range(len(samples)):
        # For each context
        for j in samples[0]['pose'].keys():
            if j == 0:
                if i > 0:
                    samples[i]['pose'][j] = \
                        samples[i]['pose'][j] @ inv_pose[0]
            else:
                samples[i]['pose'][j] = \
                    samples[i]['pose'][j] @ inv_pose[i]
    return samples


def dummy_intrinsics(image):
    """
    Return dummy intrinsics calculated based on image resolution

    Parameters
    ----------
    image : PIL.Image
        Image from which intrinsics will be calculated

    Returns
    -------
    intrinsics : np.array (3x3)
        Image intrinsics (fx = cx = w/2, fy = cy = h/2)
    """
    w, h = [float(d) for d in image.size]
    return np.array([[w/2, 0., w/2. - 0.5],
                     [0., h/2, h/2. - 0.5],
                     [0., 0., 1.]])


def calculate_normals(depth, intrinsics):

    u, v = np.meshgrid(
        np.linspace(0, depth.shape[0] - 1, depth.shape[0]),
        np.linspace(0, depth.shape[1] - 1, depth.shape[1]), indexing='ij')
    print('bbb', u.shape, v.shape)
    u = 2 * u / (depth.shape[0] - 1) - 1
    v = 2 * v / (depth.shape[1] - 1) - 1
    uv1 = np.stack([u, v, np.ones_like(u)], 2)
    print('asdfasdf', uv1.shape, depth.shape)
    print('qwer', uv1[0, 0], uv1[-1, -1])
    points = uv1 @ np.linalg.inv(intrinsics).T
    points = points * np.expand_dims(depth, 2)
    p0 = points[:-1, :-1]
    p1 = points[ 1:, :-1]
    p2 = points[:-1,  1:]

    normals = np.cross(p1 - p0, p2 - p0, 2)
    normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.pad(normals, ((0, 1), (0, 1), (0, 0)), mode='edge')

    return normals.transpose(2, 0, 1)
