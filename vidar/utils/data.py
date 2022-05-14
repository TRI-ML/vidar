# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import random
from collections import OrderedDict
from inspect import signature

import numpy as np
import torch

from vidar.utils.decorators import iterate1, iterate2
from vidar.utils.types import is_list, is_double_list, is_tuple, is_tensor, is_dict, is_seq

KEYS_IMAGE = [
        'rgb', 'mask',
        'input_depth', 'depth',
        'bwd_optical_flow', 'fwd_optical_flow',
    ]

KEYS_MATRIX = [
        'intrinsics', 'extrinsics', 'pose', 'semantic',
    ]


def modrem(v, n):
    """Return round division and remainder"""
    return v // n, v % n


def flatten(lst):
    """Flatten a list of lists into a list"""
    return [l for ls in lst for l in ls] if is_double_list(lst) else lst


def keys_with(dic, string, without=()):
    """Return keys from a dictionary that contain a certain string"""
    return [key for key in dic if string in key and not any(w in key for w in make_list(without))]


def keys_startswith(dic, string):
    """Return keys from a dictionary that contain a certain string"""
    return [key for key in dic if key.startswith(string)]


def keys_in(dic, keys):
    """Return only keys in a dictionary"""
    return [key for key in keys if key in dic]


def str_not_in(string, keys):
    for key in keys:
        if key in string:
            return False
    return True


def make_list(var, n=None):
    """Wraps the input into a list, and optionally repeats it to be size n"""
    var = var if is_list(var) or is_tuple(var) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n, 'Wrong list length for make_list'
        return var * n if len(var) == 1 else var


def filter_args(func, keys):
    """Filters a dictionary, so it only contains keys that are arguments of a function"""
    filtered = {}
    sign = list(signature(func).parameters.keys())
    for k, v in {**keys}.items():
        if k in sign:
            filtered[k] = v
    return filtered


def dict_remove_nones(dic):
    """Filters dictionary to remove keys with None values"""
    return {key: val for key, val in dic.items() if val is not None}


@iterate1
def matmul1(v1, v2):
    """Iteratively multiply tensors"""
    return v1 @ v2


@iterate2
def matmul2(v1, v2):
    """Iteratively multiply tensors"""
    return v1 @ v2


@iterate1
def unsqueeze(x):
    """Iteratively unsqueeze tensors to batch size 1"""
    return x.unsqueeze(0) if is_tensor(x) else x


@iterate1
def fold(data, n):
    """Iteratively folds first and second dimensions into one"""
    shape = list(data.shape)
    if len(shape) == n + 1:
        shape = [shape[0] * shape[1]] + shape[2:]
        return data.view(*shape)
    else:
        return data


@iterate1
def expand(data, n, d):
    """Iteratively folds first and second dimensions into one"""
    shape = list(data.shape)
    if len(shape) == n:
        return data.unsqueeze(d)
    else:
        return data


def fold_batch(batch, device=None):
    """Combine the first (batch) and second (camera) dimensions of a batch"""
    if is_seq(batch):
        return [fold_batch(b, device=device) for b in batch]
    for key in keys_in(batch, KEYS_IMAGE):
        batch[key] = fold(batch[key], 4)
    for key in keys_in(batch, KEYS_MATRIX):
        batch[key] = fold(batch[key], 3)
    if device is not None:
        batch = batch_to_device(batch, device)
    return batch


def expand_batch(batch, d, device=None):
    """Expand the batch to include an additional dimension (0 for batch, 1 for camera)"""
    if is_seq(batch):
        return [expand_batch(b, d, device=device) for b in batch]
    d = {'batch': 0, 'camera': 1}[d]
    for key in keys_in(batch, KEYS_IMAGE):
        batch[key] = expand(batch[key], 4, d)
    for key in keys_in(batch, KEYS_MATRIX):
        batch[key] = expand(batch[key], 3, d)
    if device is not None:
        batch = batch_to_device(batch, device)
    return batch


def batch_to_device(batch, device):
    """Copy batch information to device"""
    if is_dict(batch):
        return {key: batch_to_device(val, device) for key, val in batch.items()}
    if is_list(batch):
        return [batch_to_device(val, device) for val in batch]
    if is_tensor(batch):
        return batch.to(device)
    return batch


def num_trainable_params(model):
    """Return number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seed(seed):
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_batch(batch, device=None):
    """Transforms a sample into a batch"""
    for key in batch.keys():
        if is_dict(batch[key]):
            batch[key] = make_batch(batch[key])
        elif is_tensor(batch[key]):
            batch[key] = batch[key].unsqueeze(0)
    if device is not None:
        batch = batch_to_device(batch, device)
    return batch


def break_key(sample, n=None):
    """Break a multi-camera sample key, so different cameras have their own entries (context, camera)"""
    if sample is None:
        return sample
    new_sample = OrderedDict()
    for ctx in sample.keys():
        if is_dict(sample[ctx]):
            for key2, val in sample[ctx].items():
                if val.dim() == 1:
                    val = val.unsqueeze(1)
                for i in range(val.shape[1]):
                    if (ctx, i) not in new_sample.keys():
                        new_sample[(ctx, i)] = {}
                    new_sample[(ctx, i)][key2] = val[:, [i]]
        elif sample[ctx].dim() == n + 1:
            for i in range(sample[ctx].shape[1]):
                new_sample[(ctx, i)] = sample[ctx][:, i]
    return new_sample


def break_batch(batch):
    """Break a multi-camera batch, so different cameras have their own entries (context, camera)"""
    for key in keys_in(batch, KEYS_IMAGE):
        for ctx in list(batch[key].keys()):
            if batch[key][ctx].dim() == 5:
                for n in range(batch[key][ctx].shape[1]):
                    batch[key][(ctx,n)] = batch[key][ctx][:, n]
                batch[key].pop(ctx)
    for key in keys_in(batch, KEYS_MATRIX):
        for ctx in list(batch[key].keys()):
            if batch[key][ctx].dim() == 4:
                for n in range(batch[key][ctx].shape[1]):
                    batch[key][(ctx,n)] = batch[key][ctx][:, n]
                batch[key].pop(ctx)
    return batch


def dict_has(dic, key):
    """Check if a dictionary has a certain key"""
    return key in dic


def get_from_dict(dic, key):
    """Get value from a dictionary (return None if key is not in dictionary)"""
    return None if key not in dic else dic[key]


def get_mask_from_list(mask, i, return_ones=None):
    """Retrieve mask from a list (if it's not a list, return the mask itself, and create one if requested)"""
    if return_ones is None:
        return None if mask is None else mask[i] if is_list(mask) else mask
    else:
        mask = torch.ones_like(return_ones[i] if is_list(return_ones) else return_ones).bool() if mask is None \
            else mask[i].clone().bool() if is_list(mask) else mask.clone().bool()
        if mask.dim() == 4:
            return mask[:, [0]]
        elif mask.dim() == 3:
            return mask[..., [0]]


def get_from_list(lst, i):
    """Get information from a list (return None if input is None, and return value directly if it's not a list)"""
    return None if lst is None else lst[i] if is_seq(lst) else lst
