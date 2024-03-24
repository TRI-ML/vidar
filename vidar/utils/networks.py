# Copyright 2023 Toyota Research Institute.  All rights reserved.

import math

import torch
import torch.nn as nn

from knk_vision.vidar.vidar.utils.distributed import print0, rank, dist_mode
from knk_vision.vidar.vidar.utils.logging import pcolor
from knk_vision.vidar.vidar.utils.tensor import same_shape
from knk_vision.vidar.vidar.utils.types import is_list


def freeze_layers(network, layers=('ALL',), flag_freeze=True):
    """Freeze layers of a network"""
    if len(layers) > 0:
        for name, parameters in network.named_parameters():
            for layer in layers:
                if layer in name or layer == 'ALL':
                    parameters.requires_grad_(not flag_freeze)


def freeze_norms(network, layers=('ALL',), flag_freeze=True):
    """Freeze normalization layers of a network"""
    if len(layers) > 0:
        for name, module in network.named_modules():
            for layer in layers:
                if layer in name or layer == 'ALL':
                    if isinstance(module, nn.BatchNorm2d) or \
                            isinstance(module, nn.LayerNorm):
                        if hasattr(module, 'weight'):
                            module.weight.requires_grad_(not flag_freeze)
                        if hasattr(module, 'bias'):
                            module.bias.requires_grad_(not flag_freeze)
                        if flag_freeze:
                            module.eval()
                        else:
                            module.train()


def freeze_layers_and_norms(network, layers=('ALL',), flag_freeze=True):
    """Freeze layers and normalizations of a network"""
    freeze_layers(network, layers, flag_freeze)
    freeze_norms(network, layers, flag_freeze)


def make_val_fit(model, key, val, updated_state_dict, mode='copy', strict=False):
    """Modify a value in a state dict to fit the model"""
    fit = 0
    val_new = model.state_dict()[key]
    if same_shape(val.shape, val_new.shape):
        updated_state_dict[key] = val
        fit += 1
    elif not strict:
        for i in range(val.dim()):
            if val.shape[i] != val_new.shape[i]:
                if val_new.shape[i] > val.shape[i]:
                    if mode == 'copy':
                        ratio = math.ceil(val_new.shape[i] / val.shape[i])
                        val = torch.cat([val] * ratio, i)
                        if val.shape[i] != val_new.shape[i]:
                            if i == 0: val = val[:val_new.shape[0], ...]
                            elif i == 1: val = val[:, :val_new.shape[1], ...]
                            elif i == 2: val = val[:, :, val_new.shape[2], ...]
                    elif mode in ['zeros', 'small']:
                        shape = list(val.shape)
                        shape[i] = val_new.shape[i]
                        k = {'zeros': 0, 'small': 1e-6}[mode]
                        tensor_new = k * torch.zeros(shape, dtype=val.dtype)
                        if i == 0: tensor_new[:val.shape[0], ...] = val
                        elif i == 1: tensor_new[:, :val.shape[1], ...] = val
                        elif i == 2: tensor_new[:, :, val.shape[2], ...] = val
                        val = tensor_new
                    if same_shape(val.shape, val_new.shape):
                        updated_state_dict[key] = val
                        fit += 1
                elif val_new.shape[i] < val.shape[i]:
                    if i == 0: val = val[:val_new.shape[0], ...]
                    elif i == 1: val = val[:, :val_new.shape[1], ...]
                    elif i == 2: val = val[:, :, val_new.shape[2], ...]
                    if same_shape(val.shape, val_new.shape):
                        updated_state_dict[key] = val
                        fit += 1
    assert fit <= 1  # Each tensor cannot fit 2 or more times
    return fit


def load_checkpoint(model, checkpoint, mode='copy', strict=False, verbose=False, prefix=None,
                    remove_prefixes=('model.', 'module.'), replaces=()):
    """Load a checkpoint to a model"""
    if is_list(checkpoint):
        for ckpt in checkpoint:
            load_checkpoint(model, ckpt, mode, strict, verbose, prefix, remove_prefixes, replaces)
        return model

    font1 = {'color': 'magenta', 'attrs': ('bold', 'dark')}
    font2 = {'color': 'magenta', 'attrs': ('bold',)}

    if verbose:
        print0(pcolor('#' * 60, **font1))
        print0(pcolor('###### Loading from checkpoint: ', **font1) +
               pcolor('{}'.format(checkpoint), **font2))

    state_dict = torch.load(
        checkpoint, map_location='cpu' if dist_mode() == 'cpu' else 'cuda:{}'.format(rank())
    )['state_dict']
    updated_state_dict = {}

    prefix_state_dict = {}
    for key1, val1 in state_dict.items():
        for key2, val2 in model.state_dict().items():
            if key1.endswith(key2):
                prefix_state_dict[key2] = val1
    state_dict = prefix_state_dict

    total, fit = len(model.state_dict()), 0
    for key, val in state_dict.items():

        for replace in replaces:
            key = key.replace(replace[0], replace[1])

        if key in model.state_dict().keys():
            fit += make_val_fit(
                model, key, val, updated_state_dict,
                mode=mode, strict=strict
            )

    model.load_state_dict(updated_state_dict, strict=strict)

    if verbose:
        color = 'red' if fit == 0 else 'yellow' if fit < total else 'green'
        print0(pcolor('###### Loaded ', **font1) + \
               pcolor('{}/{}'.format(fit,total), color=color, attrs=('bold',)) + \
               pcolor(' tensors', **font1))
        print0(pcolor('#' * 60, **font1))

    return model


def save_checkpoint(filename, wrapper, epoch):
    """Save a checkpoint"""
    torch.save({
        'config': wrapper.cfg, 'epoch': epoch,
        'state_dict': wrapper.arch.state_dict(),
    }, filename)
