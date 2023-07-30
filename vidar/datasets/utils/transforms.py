# Copyright 2023 Toyota Research Institute.  All rights reserved.

from functools import partial

from vidar.datasets.augmentations.image import \
    colorjitter_sample, normalize_sample
from vidar.datasets.augmentations.crop import \
    crop_sample_input, crop_sample
from vidar.datasets.augmentations.misc import \
    duplicate_sample, mask_depth_percentage, mask_depth_number, clip_depth, mask_depth_range
from vidar.datasets.augmentations.resize import resize_sample, resize_sample_input
from vidar.datasets.augmentations.tensor import to_tensor_sample
from vidar.datasets.utils.misc import parse_crop, parse_crop_borders, parse_crop_random
from vidar.utils.types import is_list
from vidar.utils.data import make_list


def train_transforms(sample, cfg):
    """
    Training data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    cfg : config
        Configuration for transformations

    Returns
    -------
    sample : dict
        Augmented sample
    """
    # Make sample a list if it's not
    sample = make_list(sample)
    if cfg.has('calc_normals', False):
        from vidar.datasets.utils.misc import calculate_normals
        for i in range(len(sample)):
            sample[i]['normals'] = {
                tgt: calculate_normals(sample[i]['depth'][tgt], sample[i]['intrinsics'][tgt])
                for tgt in sample[i]['depth'].keys()
            }
    for key in cfg.dict.keys():
        if key.startswith('resize') and not key.endswith('supervision'):
            resize_fn = resize_sample if cfg.has('resize_supervision') else resize_sample_input
            shape_supervision = None if not cfg.has('resize_supervision') else \
                cfg.dict[key] if not is_list(cfg.resize_supervision) else cfg.resize_supervision
            sample = [resize_fn(s, shape=cfg.dict[key], shape_supervision=shape_supervision,
                                depth_downsample=cfg.has('depth_downsample', 1.0),
                                preserve_depth=cfg.has('preserve_depth', False)) for s in sample]
        elif key.startswith('crop_borders'):
            crop_fn = crop_sample if cfg.has('crop_supervision') else crop_sample_input
            sample = [crop_fn(s, parse_crop_borders(cfg.dict[key], s['rgb'][(0, 0)].size[::-1])) for s in sample]
        elif key.startswith('crop_random'):
            crop_fn = crop_sample if cfg.has('crop_supervision') else crop_sample_input
            sample = [crop_fn(s, parse_crop_random((0, 0, *s['rgb'][(0, 0)].size), cfg.dict[key])) for s in sample]
    # # Crop
    # if cfg.has('crop_borders') or cfg.has('crop_random'):
    #     crop_fn = crop_sample if cfg.has('crop_supervision') else crop_sample_input
    #     sample = [crop_fn(s, parse_crop(cfg, s['rgb'][(0, 0)].size[::-1])) for s in sample]
    # # Resize
    # if cfg.has('resize'):
    #     resize_fn = resize_sample if cfg.has('resize_supervision') else resize_sample_input
    #     shape_supervision = None if not cfg.has('resize_supervision') else \
    #         cfg.resize if not is_list(cfg.resize_supervision) else cfg.resize_supervision
    #     sample = [resize_fn(s, shape=cfg.resize, shape_supervision=shape_supervision,
    #                         depth_downsample=cfg.has('depth_downsample', 1.0),
    #                         preserve_depth=cfg.has('preserve_depth', False)) for s in sample]
    # Clip depth to a maximum value
    if cfg.has('clip_depth'):
        sample = clip_depth(sample, cfg.clip_depth)
    if cfg.has('mask_depth_range'):
        sample = mask_depth_range(sample, cfg.mask_depth_range)
    # Change input depth density
    if 'input_depth' in sample:
        if cfg.has('input_depth_number'):
            sample['input_depth'] = mask_depth_number(
                sample['input_depth'], cfg.input_depth_number)
        if cfg.has('input_depth_percentage'):
            sample['input_depth'] = mask_depth_percentage(
                sample['input_depth'], cfg.input_depth_percentage)
    # Apply jittering
    if cfg.has('jittering'):
        sample = duplicate_sample(sample, ['rgb'])
        sample = colorjitter_sample(sample, cfg.jittering, cfg.has('background', None), prob=1.0)
    # Convert to tensor
    sample = to_tensor_sample(sample)
    if cfg.has('normalization'):
        sample = normalize_sample(sample, cfg.normalization[0], cfg.normalization[1])
    # Return augmented sample
    return sample[0]


def no_transform(sample):
    # Make sample a list if it's not
    sample = make_list(sample)
    sample = to_tensor_sample(sample)
    return sample[0]


def get_transforms(mode, cfg=None):
    """
    Get data augmentation transformations for each split

    Parameters
    ----------
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the data augmentation transformations
    cfg : Config
        Configuration file

    Returns
    -------
        XXX_transform: Partial function
            Data augmentation transformation for that mode
    """
    if mode == 'train':
        return partial(train_transforms, cfg=cfg)
    elif mode == 'none':
        return partial(no_transform)
    else:
        raise ValueError('Unknown mode {}'.format(mode))

