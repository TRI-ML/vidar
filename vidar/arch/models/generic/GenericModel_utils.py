# Copyright 2023 Toyota Research Institute.  All rights reserved.

import random

import torch

from vidar.geometry.camera import Camera
from vidar.utils.data import make_list, get_from_dict
from vidar.utils.tensor import interpolate_nearest, same_shape
from vidar.utils.types import is_dict, is_seq, is_double_seq, is_list, is_tuple


def prepare_gt(data):
    """Prepare GT information in the same format as predictions."""
    if is_dict(data):
        return {key: prepare_gt(val) for key, val in data.items()}
    elif is_list(data):
        return data[0].detach()
    else:
        return data.detach()


def apply_masks(masks, mask_valid_tgt):
    """Apply binary masks to GT and predictions."""
    if mask_valid_tgt is not None:
        for key in masks.keys():
            for i in range(len(masks[key])):
                masks[key][i] *= interpolate_nearest(
                    get_if_dict(mask_valid_tgt, key), size=masks[key][i]).bool()
    return masks


def get_gt(cfg, batch, predictions, task, pred_key):
    """Get GT information from batch and predictions"""
    if cfg.gt == 'gt':
        return get_if_not_none(batch, pred_key, get_if_not_none(batch, task))
    else:
        src, key = cfg.gt.split('|')
        gt = batch[key] if src == 'gt' else predictions[key] if src == 'pred' else None
        return prepare_gt(gt)


def apply_loss(pred_key, apply_to):
    """Apply loss to specific predictions."""
    return len(apply_to) == 0 or any([pred == pred_key for pred in apply_to])


def get_mask_valid(cfg, batch, predictions):
    """Get binary masks for valid pixels."""
    mask_valid = None if cfg.masks[0] is None else \
        [batch[mask] if mask in batch else predictions[mask] for mask in cfg.masks]
    if mask_valid is not None:
        for key1 in mask_valid[0].keys():
            if is_dict(mask_valid[0][key1]):
                for key2 in mask_valid[0][key1].keys():
                    for i in range(1, len(mask_valid)):
                        mask_valid[0][key1][key2] *= get_if_dict(mask_valid[i][key1], key2).bool()
            else:
                for i in range(1, len(mask_valid)):
                    mask_valid[0][key1] *= mask_valid[i][key1].bool()
        mask_valid = mask_valid[0]
    return mask_valid


def sample_from_coords(data, coords, cams, tgt=None):
    """Sample data from coordinates."""
    if data is None:
        return None
    data_tgt = get_if_dict(data, tgt)
    coords_tgt = get_if_dict(coords, tgt)
    cams_tgt = get_if_dict(cams, tgt)
    if is_dict(data_tgt):
        return {key: sample_from_coords(data_tgt, coords_tgt, cams_tgt, key)
                for key in data_tgt.keys()}
    else:
        if same_shape(data_tgt.shape[-2:], cams_tgt.hw):
            return data_tgt
        else:
            b, c, h, w = data_tgt.shape
            data_key = data_tgt.permute(0, 2, 3, 1).view(b, -1, c)
            data_key = torch.stack([data_key[i][coords_tgt[i]] for i in range(b)], 0)
            if cams_tgt is not None:
                data_key = data_key.permute(0, 2, 1).view(b, c, *cams_tgt.hw)
            return data_key


def sample_supervision(pred_key, task_key, predictions, data, tgt=None):
    """Sample supervision from predictions."""
    if data is None:
        return None
    coords_sampled = get_if_not_none(predictions, pred_key.replace(task_key, 'coords'))
    if coords_sampled is None:
        return get_if_dict(data, tgt)
    cams_sampled = get_if_not_none(predictions, pred_key.replace(task_key, 'cams'))
    return sample_from_coords(data, coords_sampled, cams_sampled, tgt)


def is_dense(data):
    """Check if data is dense (no zero values)."""
    return (data <= 0).sum() == 0


def is_valid(data):
    """Check if data is valid (no NaN values)."""
    return data.max() > 0


def dense_batches(data):
    """Get indices of dense batches."""
    return [i for i in range(data.shape[0]) if is_dense(data[i])]


def valid_batches(data):
    """Get indices of valid batches."""
    return [i for i in range(data.shape[0]) if is_valid(data[i])]


def parse_params(params):
    """Parse parameters for filtering, given a string."""
    params = str(params)

    prev_idx, num = 0, None
    if params.endswith(')'):
        params, num = params[:-3], int(params[-2:-1])

    keys, signs = [], ['+']
    for idx in range(0, len(params)):
        if params[idx] in ['+', '-'] and params[idx-1] not in ['[', ',', ' ']:
            signs.append(params[idx])
            keys.append(params[prev_idx:idx])
            prev_idx = idx + 1
    keys.append(params[prev_idx:])

    for i in range(len(keys)):
        if keys[i].startswith('['):
            keys[i] = tuple(eval(keys[i]))

    return keys, signs, num


def get_if_not_none(data, key, default=None):
    """Get information from dictionary if available and not None"""
    return data[key] if data is not None and key in data.keys() else default


def get_if_dict(data, key):
    """Get information from a dictionary if it's a dictionary"""
    return None if data is None else data[key] if is_dict(data) else data


def make_dict_list(data):
    """Make a dictionary of lists from a dictionary of dictionaries"""
    for key in data.keys():
        if is_dict(data[key]):
            data[key] = make_dict_list(data[key])
        else:
            data[key] = make_list(data[key])
    return data


def add_key_to_dict(data, keys, sub_keys=None):
    """Add keys to a dictionary"""
    for key in make_list(keys):
        if key not in data.keys():
            data[key] = {}
    if sub_keys is not None:
        if not is_double_seq(sub_keys):
            sub_keys = [sub_keys]
        for i in range(len(keys)):
            data[keys[i]] = add_key_to_dict(data[keys[i]], sub_keys[i])
    return data


def sum_valid(values):
    """Sum values that are not None and not NaN"""
    if is_seq(values):
        return None if len(values) == 0 else \
            sum([val for val in values if val is not None and not torch.isnan(val)])
    elif is_dict(values):
        return None if len(values) == 0 else \
            sum([val for val in values.values() if val is not None and not torch.isnan(val)])
    else:
        return None


def create_cameras(rgb, intrinsics, pose, zero_origin=True, scaled=None, tgt=(0, 0)):
    """Create cameras from intrinsics and pose."""
    if pose is None:
        return None
    cams = {key: Camera(
        K=intrinsics[key] if is_dict(intrinsics) else intrinsics,
        Twc=pose[key], hw=rgb[key] if is_dict(rgb) else rgb,
    ).scaled(scaled).to(pose[key].device) for key in pose.keys()}
    if zero_origin:
        cams[tgt] = Camera(
            K=intrinsics[tgt] if is_dict(intrinsics) else intrinsics,
            hw=rgb[tgt] if is_dict(rgb) else rgb,
        ).scaled(scaled).to(rgb.device)
    return cams


def check_assert(pred, gt):
    """Check if predictions and GT are the same."""
    if is_dict(pred) and is_dict(gt):
        for key in gt.keys():
            print(key)
            check_assert(pred[key], gt[key])
    elif is_seq(pred) and is_seq(gt):
        for i in range(len(gt)):
            print(i)
            check_assert(pred[i], gt[i])
    else:
        assert torch.allclose(pred, gt)


def filter_params(target, keys, params):
    """Filter parameters for a specific target, given a string"""
    ### INITIALIZE AS DEFAULT
    params, signs, num = parse_params(params)
    ### SPATIAL AND TEMPORAL CONTEXTS
    spatial = [key for key in keys if key[0] == target[0] and key[1] != target[1]]
    temporal = [key for key in keys if key[1] == target[1] and key[0] != target[0]]
    ### CREATE TARGETS LIST
    targets = []
    for i in range(len(params)):
        if is_tuple(params[i]):
            if signs[i] == '+':
                targets.append(params[i])
            elif signs[i] == '-' and params[i] in targets:
                targets.remove(params[i])
        elif params[i].startswith('target'):
            if signs[i] == '+':
                targets.append(target)
            elif signs[i] == '-' and target in targets:
                targets.remove(target)
        elif params[i].startswith('spatial'):
            spatial_i = spatial if params[i] == 'spatial' else \
                [s for s in spatial if abs(s[1] - target[1]) <= int(params[i][len('spatial'):])]
            if signs[i] == '+':
                targets += spatial_i
            elif signs[i] == '-':
                targets += [t for t in targets if t not in spatial_i]
        elif params[i].startswith('extrinsics'):
            spatial_i = spatial if params[i] == 'extrinsics' else \
                [s for s in spatial if abs(s[1] - target[1]) <= int(params[i][len('extrinsics'):])]
            if signs[i] == '+':
                targets += spatial_i
            elif signs[i] == '-':
                targets += [t for t in targets if t not in spatial_i]
        elif params[i].startswith('temporal'):
            temporal_i = temporal if params[i] == 'temporal' else \
                [s for s in temporal if abs(s[0] - target[0]) <= int(params[i][len('temporal'):])]
            if signs[i] == '+':
                targets += temporal_i
            elif signs[i] == '-':
                targets = [t for t in targets if t not in temporal_i]
        elif params[i].startswith('all'):
            if signs[i] == '+':
                targets += keys
            elif signs[i] == '-':
                targets = [t for t in targets if t not in keys]
    targets = list(set(targets))
    if num is not None:
        random.shuffle(targets)
        targets = targets[:num]
    return targets


def filter_targets(tgt, keys, params):
    """Filter targets for a specific target, given a string"""
    return filter_params(tgt, keys, params)


def filter_losses(tgt, keys, params):
    """Filter losses for a specific target, given a string"""
    return filter_params(tgt, keys, params)


def filter_contexts(tgts, keys, params):
    """Filter contexts for a specific target, given a string"""
    if is_list(tgts):
        return {tgt: filter_contexts(tgt, keys, params) for tgt in tgts}
    return filter_params(tgts, keys, params)


def filter_encodes(tgt, keys, params):
    """Filter encodes for a specific target, given a string"""
    return filter_params(tgt, keys, params)


def make_pairs(tgts, ctxs, only_both_ways=False):
    """Make pairs of targets and contexts"""
    pairs = []
    for tgt in tgts:
        for ctx in ctxs[tgt]:
            if tgt != ctx:
                pairs.append([tgt, ctx])
    if only_both_ways:
        pairs = [pair for pair in pairs if [pair[0], pair[1]] in pairs and [pair[1], pair[0]] in pairs]
    new_pairs = []
    for pair in pairs:
        if [pair[1], pair[0]] not in new_pairs:
            new_pairs.append(pair)
    pairs = new_pairs
    keys = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    return keys, pairs


def parse_keys(all_keys, keys):
    """Parse keys for a specific target, given a string"""
    new_idxs = []
    for key in keys:
        if is_list(key[0]) and is_list(key[1]):
            for i in key[0]:
                for j in key[1]:
                    new_idxs.append([i, j])
        elif is_list(key[0]):
            for i in key[0]:
                new_idxs.append([i, key[1]])
        elif is_list(key[1]):
            for i in key[1]:
                new_idxs.append([key[0], i])
        elif key[0] == '*':
            for i in all_keys:
                if i[1] == key[1]:
                    new_idxs.append(i)
        elif key[1] == '*':
            for i in all_keys:
                if i[0] == key[0]:
                    new_idxs.append(i)
        else:
            new_idxs.append(tuple(key))
    return new_idxs


def parse_source(string, batch, predictions):
    """Parse source for a specific target, given a string"""
    if string is None:
        return None, None, None
    else:
        src, task = string.split('|')
        return src, task, get_from_dict(batch, task) if src == 'gt' else get_from_dict(predictions, task)


def update_losses(losses, metrics, extras, output, key, mode):
    """Update losses, metrics, and extras from output"""
    if output is not None:
        if output['loss'] is not None:
            losses[f'{key}_{mode}'] = output['loss']
            metrics[f'{key}_{mode}'] = output['loss']
        if output['metrics'] is not None:
            metrics.update(**{f'{key}_{mode}_{metric}': val
                              for metric, val in output['metrics'].items()})
        if output['extras'] is not None:
            extras.update(**output['extras'])


def update_predictions(batch, predictions, losses, extra, output, task_key):
    """Update predictions, losses, and extras from output"""
    if 'predictions' in output:
        predictions.update(**output['predictions'])
    if 'losses' in output:
        losses.update(**output['losses'])
    if 'extra' in output:
        extra.update(**{f'{task_key}_{key}': val for key, val in output['extra'].items()})
    if 'gt' in output:
        for key, val in output['gt'].items():
            batch[f'{key}_{task_key}'] = val

