# Copyright 2023 Toyota Research Institute.  All rights reserved.

import random

import torch
import torchvision.transforms as transforms

from vidar.utils.data import keys_in
from vidar.utils.decorators import iterate1


def random_colorjitter(parameters):
    """
    Creates a reusable color jitter transformation

    Parameters
    ----------
    parameters : tuple (brightness, contrast, saturation, hue, color)
        Color jittering parameters

    Returns
    -------
    transform : torch.vision.Transform
        Color jitter transformation with fixed parameters
    """
    # Get and unpack values
    brightness, contrast, saturation, hue = parameters
    brightness = [max(0, 1 - brightness), 1 + brightness]
    contrast = [max(0, 1 - contrast), 1 + contrast]
    saturation = [max(0, 1 - saturation), 1 + saturation]
    hue = [-hue, hue]

    # Initialize transformation list
    all_transforms = []

    # Add brightness transformation
    if brightness is not None:
        brightness_factor = random.uniform(brightness[0], brightness[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_brightness(img, brightness_factor)))
    # Add contrast transformation
    if contrast is not None:
        contrast_factor = random.uniform(contrast[0], contrast[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_contrast(img, contrast_factor)))
    # Add saturation transformation
    if saturation is not None:
        saturation_factor = random.uniform(saturation[0], saturation[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_saturation(img, saturation_factor)))
    # Add hue transformation
    if hue is not None:
        hue_factor = random.uniform(hue[0], hue[1])
        all_transforms.append(transforms.Lambda(
            lambda img: transforms.functional.adjust_hue(img, hue_factor)))
    # Shuffle transformation order
    random.shuffle(all_transforms)
    # Return composed transformation
    return transforms.Compose(all_transforms)


def colorjitter_sample(samples, parameters, background=None, prob=1.0):
    """
    Jitters input images as data augmentation.

    Parameters
    ----------
    samples : dict
        Input sample
    parameters : tuple (brightness, contrast, saturation, hue, color)
        Color jittering parameters
    background: None or str
        Which background color should be use
    prob : float
        Jittering probability

    Returns
    -------
    sample : dict
        Jittered sample
    """
    if random.random() < prob:
        # Prepare jitter transformation
        colorjitter_transform = random_colorjitter(parameters[:4])
        # Prepare color transformation if requested
        if len(parameters) > 4 and parameters[4] > 0:
            matrix = (random.uniform(1. - parameters[4], 1 + parameters[4]), 0, 0, 0,
                      0, random.uniform(1. - parameters[4], 1 + parameters[4]), 0, 0,
                      0, 0, random.uniform(1. - parameters[4], 1 + parameters[4]), 0)
        else:
            matrix = None
        for sample in samples:
            # Jitter sample keys
            for key in keys_in(sample, ['rgb']):
                for ctx in sample[key].keys():
                    bkg, color = [], {'white': (255, 255, 255), 'black': (0, 0, 0)}
                    if background is not None:
                        for i in range(sample[key][ctx].size[0]):
                            for j in range(sample[key][ctx].size[1]):
                                if sample[key][ctx].getpixel((i,j)) == color[background]:
                                    bkg.append((i,j))
                    sample[key][ctx] = colorjitter_transform(sample[key][ctx])
                    if matrix is not None:
                        sample[key][ctx] = sample[key][ctx].convert('RGB', matrix)
                    if background is not None:
                        for ij in bkg:
                            sample[key][ctx].putpixel(ij, color[background])
    # Return jittered (?) sample
    return samples


@iterate1
def normalize_sample(sample, mean, std):
    # Get mean and std values in the right shape
    mean = torch.tensor(mean).reshape(3, 1, 1)
    std = torch.tensor(std).reshape(3, 1, 1)
    # Apply mean and std to every image
    for key_sample in keys_in(sample, ['rgb']):
        sample[key_sample] = {key:(val - mean) / std for
                              key, val in sample[key_sample].items()}
    return sample
