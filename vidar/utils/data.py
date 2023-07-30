# Copyright 2023 Toyota Research Institute.  All rights reserved.

import os
import random
from collections import OrderedDict
from inspect import signature, isfunction

import numpy as np
import torch

from vidar.utils.decorators import iterate1, iterate2, iterate12
from vidar.utils.types import is_list, is_double_list, is_tuple, is_tensor, is_dict, is_seq

KEYS_IMAGE = [
        'rgb', 'mask',
        'input_depth', 'depth',
        'bwd_optical_flow', 'fwd_optical_flow',
        'bwd_scene_flow', 'fwd_scene_flow',
    ]

KEYS_MATRIX = [
        'intrinsics', 'raw_intrinsics', 'extrinsics', 'pose', 'semantic',
    ]


def exists(v):
    """Check if value is not None"""
    return v is not None


def default(v, d):
    """Check if value exists, and returns default if not"""
    return v if exists(v) else d() if isfunction(d) else d


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


def all_in(dic, *keys):
    """Return only keys in a dictionary"""
    return all([key in dic for key in keys])


def strs_not_in_key(key, strings):
    """Check if a key does not contain any of the strings"""
    for string in strings:
        if string in key:
            return False
    return True


def str_not_in(string, keys):
    """Check if a string is not in any of the keys"""
    for key in keys:
        if key in string:
            return False
    return True


def not_none(*args):
    """Check if all arguments are not None"""
    return not any([arg is None for arg in args])


def one_is(val, *args):
    """Check if one of the arguments is equal to val"""
    return any([any([arg == v for v in make_list(val)]) for arg in args])


def tensor_like(data, like):
    """Returns a tensor with the same type and device as another tensor"""
    return torch.tensor(data, dtype=like.dtype, device=like.device)


def make_list(var, n=None):
    """
    Wraps the input into a list, and optionally repeats it to be size n

    Parameters
    ----------
    var : Any
        Variable to be wrapped in a list
    n : int
        How much the wrapped variable will be repeated

    Returns
    -------
    var_list : list[Any]
        List generated from var
    """
    var = var if is_list(var) or is_tuple(var) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n, 'Wrong list length for make_list'
        return var * n if len(var) == 1 else var


def filter_args(func, keys):
    """
    Filters a dictionary so it only contains keys that are arguments of a function

    Parameters
    ----------
    func : Function
        Function for which we are filtering the dictionary
    keys : dict
        Dictionary with keys we are filtering

    Returns
    -------
    filtered : dict
        Dictionary containing only keys that are arguments of func
    """
    filtered = {}
    sign = list(signature(func).parameters.keys())
    for k, v in {**keys}.items():
        if k in sign:
            filtered[k] = v
    return filtered


def remove_nones_dict(sample):
    """Removes all None values from a dictionary"""
    for key in list(sample.keys()):
        if is_dict(sample[key]):
            sample[key] = remove_nones_dict(sample[key])
        elif sample[key] is None:
            sample.pop(key)
    return sample


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
    if len(shape) == n or n is None:
        return data.unsqueeze(d)
    else:
        return data


def expand_and_break(data, d, n):
    """Expand and break a tensor"""
    return break_key(expand(data, n, d), n)


def prepare_batch(batch, break_flow=False):
    """Prepare a batch for processing"""
    prepared = {}
    for key in keys_in(batch, [
        'filename', 'idx', 'timestep', 'scene',
    ]):
        prepared[key] = batch[key]
    for key in keys_in(batch, [
        'rgb',
        'mask_rgb', 'mask_motion',
        'depth',
        'fwd_optical_flow', 'bwd_optical_flow',
        'fwd_valid_optical_flow', 'bwd_valid_optical_flow',
        'fwd_scene_flow', 'bwd_scene_flow',
    ]):
        prepared[key] = expand_and_break(batch[key], 1, 4)
    for key in keys_in(batch, [
        'intrinsics', 'pose'
    ]):
        prepared[key] = expand_and_break(batch[key], 1, 3)
    if break_flow:
        for pref in ['fwd_', 'bwd_']:
            for key in list(prepared.keys()):
                if key.startswith(pref):
                    key_mod = key.replace(pref, '')
                    if key_mod not in prepared.keys():
                        prepared[key_mod] = {}
                    for key2 in prepared[key].keys():
                        add = 1 if pref == 'fwd_' else -1
                        if key2 not in prepared[key_mod].keys():
                            prepared[key_mod][key2] = {}
                        prepared[key_mod][key2][(key2[0] + add), key2[1]] = prepared[key][key2]
                    prepared.pop(key)
    return prepared


def fold_batch(batch, device=None):
    """Fold a batch for processing (remove spatial dimension for multi-camera)"""
    if is_seq(batch):
        return [fold_batch(b, device=device) for b in batch]
    for key in keys_in(batch, KEYS_IMAGE):
        batch[key] = fold(batch[key], 4)
    for key in keys_in(batch, KEYS_MATRIX):
        batch[key] = fold(batch[key], 3)
    if device is not None:
        batch = batch_to_device(batch, device)
    return batch


def expand_batch(batch, d, all=False, device=None):
    """Expand a batch for processing (add spatial dimension for multi-camera)"""
    if is_seq(batch):
        return [expand_batch(b, d, device=device) for b in batch]
    d = {'batch': 0, 'camera': 1}[d]
    for key in keys_in(batch, KEYS_IMAGE):
        batch[key] = expand(batch[key], 4 if not all else None, d)
    for key in keys_in(batch, KEYS_MATRIX):
        batch[key] = expand(batch[key], 3 if not all else None, d)
    if device is not None:
        batch = batch_to_device(batch, device)
    return batch


def batch_to_device(batch, device):
    """Move batch to device"""
    if is_dict(batch):
        return {key: batch_to_device(val, device) for key, val in batch.items()}
    if is_list(batch):
        return [batch_to_device(val, device) for val in batch]
    if is_tensor(batch):
        return batch.to(device)
    return batch


def num_trainable_params(model):
    """Return number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seed(seed):
    """Set random seed for reproducibility"""
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
    """Break a key into multiple keys following the new tuple format (timestep, camera)"""
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
    """Break a batch into multiple batches following the new tuple format (timestep, camera)"""
    for key in keys_in(batch, KEYS_IMAGE):
        for ctx in list(batch[key].keys()):
            if batch[key][ctx].dim() == 5:
                for n in range(batch[key][ctx].shape[1]):
                    batch[key][(ctx, n)] = batch[key][ctx][:, n]
                batch[key].pop(ctx)
    for key in keys_in(batch, KEYS_MATRIX):
        for ctx in list(batch[key].keys()):
            if batch[key][ctx].dim() == 4:
                for n in range(batch[key][ctx].shape[1]):
                    batch[key][(ctx, n)] = batch[key][ctx][:, n]
                batch[key].pop(ctx)
    return batch


def dict_has(dic, key):
    """Checks if a key is in a dictionary"""
    return key in dic


def get_from_dict(data, key, key2=None):
    """Get a value from a dictionary"""
    if data is None:
        return None
    if key2 is not None:
        out = get_from_dict(data, key)
        return out if out is None else get_from_dict(out, key2)
    if is_list(key):
        for k in key:
            if k in data:
                return data[k]
        return None
    elif not is_dict(data):
        return data
    else:
        return None if key not in data else data[key]


def get_mask_from_list(mask, i, return_ones=None):
    """Get a mask from a list of masks"""
    if return_ones is None:
        return None if mask is None else mask[i] if is_list(mask) else mask
    else:
        mask = torch.ones_like(return_ones[i] if is_list(return_ones) else return_ones).bool() if mask is None \
            else mask[i].clone().bool() if is_list(mask) else mask.clone().bool()
        if mask.dim() == 4:
            return mask[:, [0]]
        elif mask.dim() == 3:
            return mask[..., [0]]


def get_from_list(lst, i=0):
    """Get a value from a list, with safe indexing"""
    return None if lst is None else lst[i] if is_seq(lst) else lst


def get_scale_from_dict(dic, i):
    """Get value from a multi-scale dictionary"""
    return None if dic is None else [val[i] for val in dic.values()]


def flip_nested_list(data):
    """Flip a nested list"""
    return [[data[j][i] for j in range(len(data))] for i in range(len(data[0]))]


def update_dict(data, key, val=None):
    """Update a dictionary with a new key and corresponding value"""
    if key not in data.keys():
        data[key] = {}
    if val is not None:
        data[key].update(val)


def update_dict_nested(data, key1, key2, key3, val, mode='append'):
    """Update a dictionary with a new nested key and corresponding value"""
    if val is None:
        return
    if key1 not in data:
        data[key1] = {}
    if key2 not in data[key1]:
        data[key1][key2] = {}
    if key3 not in data[key1][key2]:
        data[key1][key2][key3] = val
    else:
        if mode == 'append':
            data[key1][key2][key3] = make_list(data[key1][key2][key3]) + make_list(val)
        elif mode == 'multiply':
            data[key1][key2][key3] *= val
        else:
            raise ValueError('Key already exists')


def sum_list(data, empty=None):
    """Sum a list of tensors or dictionaries"""
    if len(data) == 0:
        return empty
    output = data[0]
    for i in range(1, len(data)):
        if is_tensor(output):
            output += data[i]
        elif is_dict(output):
            for key in output.keys():
                output[key] += data[i][key]
        else:
            raise ValueError('Invalid sum')
    return output


@iterate12
def interleave_dict(val1, val2):
    """Interleave two dictionaries"""
    b, c, h, w = val1.shape
    val1 = val1.permute(0, 2, 3, 1).view(b, -1, 3)
    val2 = val2.permute(0, 2, 3, 1).view(b, -1, 3)
    out = torch.zeros((b, val1.shape[1] + val2.shape[1], 3), device=val1.device, dtype=val1.dtype)
    out[:, 0::2], out[:, 1::2] = val1, val2
    return out


def align_corners(value=None):
    """Get the align_corners value from the environment variable"""
    if value is not None:
        return value
    if os.getenv('ALIGN_CORNERS') is not None:
        return True if os.getenv('ALIGN_CORNERS') == 'True' else \
               False if os.getenv('ALIGN_CORNERS') == 'False' else None
    return True


def detach_dict(data):
    """Detach a dictionary from the computation graph"""
    if data is None:
        return None
    elif is_dict(data):
        return {key: detach_dict(val) for key, val in data.items()}
    elif is_list(data):
        return [d.detach() for d in data]
    else:
        return data.detach()


def cat_text(txt, add_txt=None):
    """Concatenate text"""
    if add_txt is None or len(add_txt) == 0:
        return txt
    if len(txt) > 0:
        txt = str(txt) + ', '
    for add in add_txt:
        txt += add + ', '
    if len(add_txt) > 0:
        txt = txt[:-2]
    return txt


def get_random(val=None):
    """Get a random value"""
    if val is None:
        return 2 * random.random() - 1
    else:
        return 2  * torch.rand_like(val) - 1


def shuffle_list(data, n):
    """Shuffle a list"""
    random.shuffle(data)
    if n is not None:
        data = data[:n]
    return data


def shuffle_dict(data, n=None):
    """Shuffle a dictionary"""
    keys = list(data.keys())
    random.shuffle(keys)
    if n is not None:
        keys = keys[:n]
    return {key: data[key] for key in keys[:n]}
