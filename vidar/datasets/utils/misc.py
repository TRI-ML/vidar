# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch

from vidar.utils.decorators import iterate1
from vidar.utils.types import is_seq, is_tensor, is_dict, is_int


def stack_sample(sample, lidar_sample=None, radar_sample=None):
    """
    Stack samples from multiple cameras

    Parameters
    ----------
    sample : list[Dict]
        List of camera samples
    lidar_sample : list[Dict]
        List of lidar samples
    radar_sample : list[Dict]
        List of radar samples

    Returns
    -------
    stacked_sample: Dict
        Stacked sample
    """
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
    """Merge information from multiple samples"""
    merged_sample = {}
    for sample in samples:
        for key, val in sample.items():
            if key not in merged_sample:
                merged_sample[key] = val
            else:
                merged_sample[key] = merge_sample([merged_sample[key], val])
    return merged_sample


def parse_crop(cfg, shape):
    """Parse crop information to generate borders"""
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
    borders : Tuple
        Border input for parsing. Can be one of the following forms:
        (int, int, int, int): y, height, x, width
        (int, int): y, x --> y, height = image_height - y, x, width = image_width - x
        Negative numbers are taken from image borders, according to the shape argument
        Float numbers for y and x are treated as percentage, according to the shape argument,
            and in this case height and width are centered at that point.
    shape : Tuple
        Image shape (image_height, image_width), used to determine negative crop boundaries

    Returns
    -------
    borders : Tuple
        Parsed borders for cropping (left, top, right, bottom)
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
           0 <= borders[1] < borders[3] <= shape[0], 'Crop borders {} are invalid'.format(borders)
    # Return updated borders
    return borders


def parse_crop_random(borders, shape):
    """
    Create borders for random cropping.
    Crops are generated anywhere in the image inside the borders

    Parameters
    ----------
    borders : Tuple
        Area of the image where random cropping can happen (left, top, right, bottom)
    shape : Tuple
        Cropped output shape (height, width)

    Returns
    -------
    borders : tuple
        Parsed borders for cropping (left, top, right, bottom)
    """
    # Return full borders if there is no random crop
    if len(shape) == 0:
        return borders
    # Check if random crop is valid
    assert 0 < shape[1] <= borders[2] - borders[0] and \
           0 < shape[0] <= borders[3] - borders[1], 'Random crop must be smaller than the image'
    # Sample a crop
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
    pose : np.Array
        Input pose [4, 4]

    Returns
    -------
    inv_pose : np.Array
        Inverted pose [4, 4]
    """
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = np.transpose(pose[:3, :3])
    inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
    # return np.linalg.inv(pose)
    return inv_pose


def make_relative_pose(samples):
    """
    Convert sample poses to relative frane of reference (based on the first target frame)

    Parameters
    ----------
    samples : list[Dict]
        Input samples

    Returns
    -------
    samples : list[Dict]
        Relative samples
    """
    # Do nothing if there is no pose
    if 'pose' not in samples[0]:
        return samples
    # Get inverse current poses
    inv_pose = [invert_pose(samples[i]['pose'][0]) for i in range(len(samples))]
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
    image : PIL Image
        Image from which intrinsics will be calculated

    Returns
    -------
    intrinsics : np.Array
        Image intrinsics (fx = cx = w/2, fy = cy = h/2)  [3,3]
    """
    w, h = [float(d) for d in image.size]
    return np.array([[w/2, 0., w/2. - 0.5],
                     [0., h/2, h/2. - 0.5],
                     [0., 0., 1.]])


def load_ontology(name, filter_list=None):
    """Loads ontology from file and optionally filters it"""
    filename = 'vidar/ontologies/{}.json'.format(name)
    if os.path.exists(filename):
        ontology = json.load(open(filename, 'r'))
        if filter_list is not None and len(filter_list) > 0:
            ontology = filter_ontology(ontology, filter_list)
        return ontology
    else:
        return None


def save_ontology(ontology, name):
    """Save ontology to a JSON file"""
    if is_seq(ontology):
        ontology = ontology[0]
    for key in ontology.keys():
        ontology[key]['color'] = ontology[key]['color'].tolist()
    json.dump(ontology, open('ontologies/{}.json'.format(name), 'w'))


def filter_ontology(ontology, values):
    """Filter ontology to remove certain classes"""
    new_ontology = OrderedDict()
    for i, val in enumerate(values[1:]):
        new_ontology[i] = ontology[str(val)]
    return new_ontology


def convert_ontology(semantic_id, ontology_convert):
    """Convert from one ontology to another"""
    if ontology_convert is None:
        return semantic_id
    else:
        semantic_id_convert = semantic_id.copy()
        for key, val in ontology_convert.items():
            semantic_id_convert[semantic_id == key] = val
        return semantic_id_convert


def initialize_ontology(base, ontology):
    """Initialize ontology and conversion table if necessary"""
    return load_ontology(base), None
