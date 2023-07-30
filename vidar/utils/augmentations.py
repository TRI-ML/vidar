# Copyright 2023 Toyota Research Institute.  All rights reserved.

import random

import numpy as np
import torch

from vidar.utils.decorators import iterate1
from vidar.utils.tensor import interpolate_nearest, interpolate_image
from vidar.utils.types import is_seq, is_int
from vidar.utils.data import keys_in, get_random
from vidar.geometry.pose_utils import invert_pose


def parse_resize(shape, raw):
    """Parse resize parameters"""
    parsed = [None, None]
    for k in [0, 1]:
        if is_int(raw[k]):
            parsed[k] = raw[k]
        else:
            if is_seq(raw[k]):
                rnd = raw[k][0] + (raw[k][1] - raw[k][0]) * random.random()
            else:
                rnd = raw[k]
            parsed[k] = int(shape[k] * rnd)
            if is_seq(raw[k]) and len(raw[k]) == 3:
                parsed[k] = parsed[k] // raw[k][2] * raw[k][2]
    return parsed


@iterate1
@iterate1
def resize_rgb(rgb, resized):
    """Resize RGB image to a given shape"""
    return interpolate_image(rgb, shape=resized)


@iterate1
@iterate1
def resize_depth(depth, resized):
    """Resize depth map to a given shape"""
    if depth is None:
        return None
    if depth.min() == 0:
        return resize_depth_preserve(depth, resized=resized)
    else:
        return interpolate_nearest(depth, size=resized)


@iterate1
@iterate1
def resize_depth_preserve(depth, resized):
    """Resize depth map to a given shape, preserving density of valid pixels"""
    if depth.dim() == 4:
        return torch.stack([resize_depth_preserve(depth[i], resized)
                            for i in range(depth.shape[0])], 0)
    # Store dimensions and reshapes to single column
    c, h, w = depth.shape
    # depth = np.squeeze(depth)
    # h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    uv = torch.tensor(uv, device=depth.device, dtype=torch.long)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (resized[0] / h)).long()
    crd[:, 1] = (crd[:, 1] * (resized[1] / w)).long()
    # Filters points inside image
    idx = (crd[:, 0] < resized[0]) & (crd[:, 1] < resized[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    depth = torch.zeros(resized, device=depth.device, dtype=depth.dtype)
    depth[crd[:, 0], crd[:, 1]] = val
    # Return resized depth map
    return depth.unsqueeze(0)


@iterate1
@iterate1
def resize_optical_flow(optflow, resized):
    """Resize optical flow to a given shape"""
    ratio_w = float(resized[0]) / float(optflow.shape[2])
    ratio_h = float(resized[1]) / float(optflow.shape[3])
    optflow = interpolate_nearest(optflow, size=resized)
    optflow[:, 0] *= ratio_h
    optflow[:, 1] *= ratio_w
    return optflow


@iterate1
@iterate1
def resize_intrinsics(intrinsics, original, resized):
    """Resize intrinsics matrix to a given shape"""
    intrinsics = intrinsics.clone()

    ratio_w = resized[1] / original[1]
    ratio_h = resized[0] / original[0]

    intrinsics[:, 0, 0] *= ratio_w
    intrinsics[:, 1, 1] *= ratio_h

    intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio_w
    intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio_h

    return intrinsics


@iterate1
@iterate1
def resize_camera(camera, resized):
    """Resize camera to a given shape"""
    return camera.scaled(resized)


def resize_batch(batch, params):
    """Resize batch to a given shape"""
    if len(params) == 3 and random.random() > torch.rand(1):
        return batch

    batch_key = list(batch['rgb'].keys())[0]
    shape = batch['rgb'][batch_key].shape[-2:]
    params = parse_resize(shape, params)

    for key in keys_in(batch, ['rgb']):
        batch[key] = resize_rgb(batch[key], resized=params)
    for key in keys_in(batch, ['intrinsics']):
        batch[key] = resize_intrinsics(batch[key], original=shape, resized=params)
    for key in keys_in(batch, ['depth']):
        batch[key] = resize_depth(batch[key], resized=params)
    for key in keys_in(batch, ['optical_flow']):
        batch[key] = resize_optical_flow(batch[key], resized=params)

    return batch


def resize_define(data, params):
    """Resize DeFiNe data to a given shape"""
    if len(params) == 3 and random.random() > torch.rand(1):
        return data

    data_key = list(data.keys())[0]
    shape = data[data_key]['gt']['rgb'].shape[-2:]
    params_key = parse_resize(shape, params)

    for data_key in data.keys():
        for key in ['rgb']:
            data[data_key]['gt'][key] = resize_rgb(data[data_key]['gt'][key], resized=params_key)
        for key in ['depth']:
            data[data_key]['gt'][key] = resize_depth(data[data_key]['gt'][key], resized=params_key)
        for key in ['cam']:
            data[data_key][key] = resize_camera(data[data_key][key], resized=params_key)
    return data


def parse_crop(shape, raw):
    """Parse crop parameters"""
    parsed = [None for _ in range(len(shape))]
    for k in range(len(shape)):
        if is_int(raw[k]):
            val = raw[k]
        else:
            if is_seq(raw[k]):
                rnd = raw[k][0] + (raw[k][1] - raw[k][0]) * random.random()
                div = raw[k][2] if len(raw[k]) == 3 else None
            else:
                rnd, div = raw[k], None
            val = int(shape[k] * rnd)
            if div is not None:
                val = val // div * div
        diff = shape[k] - val
        st = int(random.random() * diff)
        parsed[k] = [st, st + val]
    return parsed


@iterate1
@iterate1
def crop_rgb(rgb, resized):
    """Crop RGB image to a given shape"""
    return rgb[..., resized[0][0]: resized[0][1], resized[1][0]: resized[1][1]].contiguous()


@iterate1
@iterate1
def crop_depth(depth, resized):
    """Crop depth image to a given shape"""
    return depth[..., resized[0][0]: resized[0][1], resized[1][0]: resized[1][1]].contiguous()


@iterate1
@iterate1
def crop_optical_flow(optical_flow, resized):
    """Crop optical flow to a given shape"""
    return optical_flow[..., resized[0][0]: resized[0][1], resized[1][0]: resized[1][1]].contiguous()


@iterate1
@iterate1
def crop_intrinsics(intrinsics, resized):
    """Crop intrinsics matrix to a given shape"""
    intrinsics = intrinsics.clone()
    intrinsics[:, 0, 2] -= resized[1][0]
    intrinsics[:, 1, 2] -= resized[0][0]
    return intrinsics


@iterate1
@iterate1
def crop_camera(camera, resized):
    """Crop camera to a given shape"""
    camera = camera.offset_start([r[0] for r in resized])
    camera.hw = (resized[0][1] - resized[0][0], resized[1][1] - resized[1][0])
    return camera


def crop_batch(batch, params):
    """Crop batch to a given shape"""
    if len(params) == 3 and random.random() > torch.rand(1):
        return batch

    batch_key = list(batch.keys())[0]
    shape = batch['rgb'][batch_key].shape[-2:]
    params = parse_crop(shape, params)

    for key in ['rgb']:
        batch[key] = crop_rgb(batch[key], resized=params)
    for key in ['intrinsics']:
        batch[key] = crop_intrinsics(batch[key], resized=params)
    for key in ['depth']:
        batch[key] = crop_depth(batch[key], resized=params)
    for key in ['optical_flow']:
        batch[key] = crop_optical_flow(batch[key], resized=params)

    return batch


def crop_define(data, params):
    """Crop DeFiNe data to a given shape"""
    if len(params) == 3 and random.random() > torch.rand(1):
        return data

    data_key = list(data.keys())[0]
    shape = data[data_key]['gt']['rgb'].shape[-2:]
    params_key = parse_crop(shape, params)

    for data_key in data.keys():
        for key in ['rgb']:
            data[data_key]['gt'][key] = crop_rgb(data[data_key]['gt'][key], resized=params_key)
        for key in ['depth']:
            data[data_key]['gt'][key] = crop_depth(data[data_key]['gt'][key], resized=params_key)
        for key in ['cam']:
            data[data_key][key] = crop_camera(data[data_key][key], resized=params_key)
    return data


def clip_depth_batch(batch, val):
    """Clip depth values on a batch to a maximum range"""
    for key in ['depth']:
        for tgt in batch[key].keys():
            batch[key][tgt][batch[key][tgt] > val] = val
    return batch

def scale_batch(batch, val):
    """Scale batch data to a given value"""
    if len(val) == 1:
        scale = val[0]
    elif len(val) == 2:
        scale = val[0] + (val[1] - val[0]) * random.random()

    for key in ['depth']:
        for tgt in batch[key].keys():
            batch[key][tgt] *= scale
    for key in ['cams']:
        tgt0 = list(batch[key].keys())[0]
        base = batch[key][tgt0].Twc.T
        base_inv = invert_pose(base)
        for tgt in batch[key].keys():
            pose = batch[key][tgt].Twc.T @ base_inv
            pose[:, :3, -1] *= scale
            batch[key][tgt].Twc.T = pose @ base
    return batch

def check_mode(key, is_training):
    """Check if a given key is for training or validation"""
    return (key.endswith('train') and is_training) or \
           (key.endswith('val') and not is_training) or \
           (not key.endswith('train') and not key.endswith('val'))

def augment_batch1(batch, cfg, is_training):
    """Augment batch data before preparation"""
    if cfg is None:
        return batch
    for key, val in cfg.dict.items():
        if key.startswith('resize') and check_mode(key, is_training):
            batch = resize_batch(batch, val)
        elif key.startswith('crop') and check_mode(key, is_training):
            batch = crop_batch(batch, val)
    return batch


def augment_define(batch, cfg):
    """Augment DeFiNe batch data"""
    if cfg is None:
        return batch
    for key, val in cfg.dict.items():
        if key.startswith('resize'):
            batch = resize_define(batch, val)
        elif key.startswith('crop'):
            batch = crop_define(batch, val)
        elif key.startswith('augment_canonical') and val is True:
            batch = augment_canonical(batch)
        elif key.startswith('jitter_camera_origin'):
            batch = jitter_camera_origin(batch, val)
    return batch


def augment_batch2(batch, cfg, is_training):
    """Augment batch data after preparation"""
    if cfg is None:
        return batch
    for key, val in cfg.dict.items():
        if key.startswith('clip_depth') and check_mode(key, is_training):
            batch = clip_depth_batch(batch, val)
        elif key.startswith('scale') and check_mode(key, is_training):
            batch = scale_batch(batch, val)
    return batch


def crop_stack_batch(batch, window):
    """Break batch information into smaller windows and stack it"""

    params = []
    key0 = list(batch['rgb'].keys())[0]
    b, hw = batch['rgb'][key0].shape[0], batch['rgb'][key0].shape[-2:]

    for h in range(0, hw[0], window[0]):
        for w in range(0, hw[1], window[1]):
            if window[0] >= hw[0]:
                hi, hf = 0, hw[0]
            else:
                hi, hf = h, h + window[0]
                if hf > hw[0]:
                    off = hf - hw[0]
                    hi, hf = hi - off, hf - off
            if window[1] >= hw[1]:
                wi, wf = 0, hw[1]
            else:
                wi, wf = w, w + window[1]
                if wf > hw[1]:
                    off = wf - hw[1]
                    wi, wf = wi - off, wf - off
            params.append([[hi, hf], [wi, wf]])

    if 'rgb' in batch:
        rgb = [crop_rgb(batch['rgb'], param) for param in params]
        batch['rgb'] = {key: torch.cat([val[key] for val in rgb], 0) for key in batch['rgb']}
    if 'intrinsics' in batch:
        intrinsics = [crop_intrinsics(batch['intrinsics'], param) for param in params]
        batch['intrinsics'] = {key: torch.cat([val[key] for val in intrinsics], 0) for key in batch['intrinsics']}
    if 'pose' in batch:
        pose = [batch['pose'] for _ in params]
        batch['pose'] = {key: torch.cat([val[key] for val in pose], 0) for key in batch['pose']}

    batch['hw'] = hw
    return batch

def merge_stack_predictions(batch, predictions, task):
    """Merge broken down cropped predictions into a single one"""

    key0 = list(predictions[task].keys())[0]
    shape = batch['hw']

    data = predictions[task]
    for key in predictions[task]:
        for idx in range(len(data[key])):

            hw = data[key0][idx].shape[-2:]
            h = int(np.ceil(shape[0] / hw[0]))
            w = int(np.ceil(shape[1] / hw[1]))
            n = h * w

            hw = data[key0][idx].shape[-2:]
            b, c = data[key0][idx].shape[:2]
            bn = b // n

            full = torch.zeros(
                (bn, c, *shape), device=data[key][idx].device, dtype=data[key][idx].dtype)
            sums = torch.zeros_like(full)

            for i in range(bn):

                cnt = 0
                for j in range(0, shape[0], hw[0]):
                    for k in range(0, shape[1], hw[1]):

                        hi, hf = j, j + hw[0]
                        wi, wf = k, k + hw[1]

                        if hf > shape[0]:
                            off = hf - shape[0]
                            hi, hf = hi - off, hf - off
                        if wf > shape[1]:
                            off = wf - shape[1]
                            wi, wf = wi - off, wf - off

                        full[i, :, hi:hf, wi:wf] += data[key][idx][i + bn * cnt]
                        sums[i, :, hi:hf, wi:wf] += 1.0
                        cnt += 1

            predictions[task][key][idx] = full / sums

    return predictions


def augment_canonical(batch):
    """Augment batch data by randomly selecting a base pose and transforming all other poses to it"""
    keys = list(batch.keys())
    base_key = random.choice(keys)

    base_pose = batch[base_key]['cam'].Tcw
    for key in keys:
        batch[key]['cam'].Twc = batch[key]['cam'].Twc * base_pose

    return batch


def jitter_camera_origin(batch, params):
    """Jitter camera origin by a random amount"""
    keys = list(batch.keys())
    base_pose = batch[keys[0]]['cam'].Twc
    base_pose_inv = batch[keys[0]]['cam'].Tcw

    for key in keys:
        batch[key]['cam'].Twc = batch[key]['cam'].Twc * base_pose_inv

    jittered = []
    for i in range(len(base_pose)):
        jitter = base_pose[i].clone()
        jitter.translateUp(params[0] * get_random())
        jitter.translateLeft(params[0] * get_random())
        jitter.translateForward(params[0] * get_random())
        jitter.rotateRoll(torch.pi * params[1] * get_random())
        jitter.rotatePitch(torch.pi * params[1] * get_random())
        jitter.rotateYaw(torch.pi * params[1] * get_random())
        jittered.append(jitter.T)
    jittered = torch.cat(jittered, 0)

    for key in keys:
        batch[key]['cam'].Twc.T = batch[key]['cam'].Twc.T @ jittered

    return batch
