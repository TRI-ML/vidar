# Copyright 2023 Toyota Research Institute.  All rights reserved.

import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from vidar.datasets.utils.transforms import get_transforms
from vidar.metrics.depth import DepthEvaluation
from vidar.metrics.extrinsics import ExtrinsicsEvaluation
from vidar.metrics.optical_flow import OpticalFlowEvaluation
from vidar.metrics.rgb import ImageEvaluation
from vidar.utils.config import Config, get_folder_name, load_class, \
    recursive_assignment, cfg_has, cfg_add_to_dict, get_from_cfg_list
from vidar.utils.config import merge_dict, to_namespace
from vidar.utils.data import flatten, keys_in
from vidar.utils.decorators import iterate12
from vidar.utils.distributed import print0, rank, world_size, dist_mode
from vidar.utils.logging import pcolor
from vidar.utils.networks import load_checkpoint
from vidar.utils.types import is_namespace, is_list


def setup_arch(cfg, checkpoint=None, verbose=False):
    """Setup architecture from config file

    Parameters
    ----------
    cfg : Config
        Configuration file with architecture information
    checkpoint : str, optional
        Checkpoint file, by default None
    verbose : bool, optional
        True if information is displayed on screen, by default False

    Returns
    -------
    torch.Module
        Generated architecture
    """

    font = {'color': 'green'}

    if verbose:
        print0(pcolor('#' * 60, **font))
        print0(pcolor('### Preparing Architecture', **font))
        print0(pcolor('#' * 60, **font))

    font1 = {'color': 'yellow', 'attrs': ('dark',)}
    font2 = {'color': 'yellow', 'attrs': ('dark', 'bold')}

    folder, name = get_folder_name(cfg.model.file, 'models')
    model = load_class(name, folder)(cfg)
    if verbose:
        print0(pcolor('###### Model:', **font2))
        print0(pcolor('######### %s' % model.__class__.__name__, **font1))

    if cfg_has(cfg, 'networks'):
        if verbose:
            print0(pcolor('###### Networks:', **font2))
        recursive_assignment(model.networks, cfg.networks, 'networks', verbose=verbose)

    if cfg_has(cfg, 'losses'):
        if verbose:
            print0(pcolor('###### Losses:', **font2))
        recursive_assignment(model.losses, cfg.losses, 'losses', verbose=verbose)

    if checkpoint is not None:
        model = load_checkpoint(model, checkpoint,
                                strict=True, verbose=verbose)
    elif cfg_has(cfg.model, 'checkpoint'):
        model = load_checkpoint(
            model, cfg.model.checkpoint,
            mode=cfg.model.has('checkpoint_mode', 'copy'),
            strict=cfg.model.has('checkpoint_strict', False),
            verbose=verbose,
        )

    return model


def setup_network(cfg):
    """Setup network from config file"""
    folder, name = get_folder_name(cfg.file, 'networks')
    return load_class(name, folder)(cfg)


def setup_dataset(cfg, verbose=False, no_transform=False):
    """Setup dataset from config file"""

    if cfg.has('external') and not cfg.external[0]:
        root = 'vidar/datasets'
    else:
        root = 'externals/efm_datasets/efm_datasets/dataloaders'

    shared_keys = ['context', 'labels', 'labels_context']

    num_datasets = 0
    for key, val in cfg.dict.items():
        if key not in shared_keys and not is_namespace(val):
            num_datasets = max(num_datasets, len(val))

    datasets, datasets_cfg = [], []
    for i in range(num_datasets):

        args = {}
        for key, val in cfg.dict.items():
            if is_namespace(val):
                cfg_add_to_dict(args, cfg, key)
            else:
                cfg_add_to_dict(args, cfg, key, i if key not in shared_keys else None)

        args['data_transform'] = get_transforms('train', cfg.augmentation) \
            if cfg_has(cfg, 'augmentation') else get_transforms('none') if not no_transform else None

        name = get_from_cfg_list(cfg, 'name', i)
        repeat = get_from_cfg_list(cfg, 'repeat', i)
        cameras = get_from_cfg_list(cfg, 'cameras', i)

        context = cfg.has('context', [])
        labels = cfg.has('labels', [])

        dataset = load_class(name + 'Dataset', root)(**args)

        if cfg_has(cfg, 'repeat') and repeat > 1:
            dataset = ConcatDataset([dataset for _ in range(repeat)])

        if verbose:
            string = f'######### {name}: {len(dataset)} samples'
            if cfg_has(cfg, 'repeat'):
                string += f' (x{repeat})'
            if cfg_has(cfg, 'context'):
                string += f' | context {context}'.replace(', ', ',')
            if cfg_has(cfg, 'cameras'):
                string += f' | cameras {cameras}'.replace(', ', ',').replace("'", "")
            if cfg_has(cfg, 'labels'):
                string += f' | labels {labels}'.replace(', ', ',').replace('\'', '')
            print0(pcolor(string, color='yellow', attrs=('dark',)))

        datasets.append(dataset)
        datasets_cfg.append(Config(**args))

    return datasets, datasets_cfg


def setup_datasets(cfg, verbose=False, concat_modes=('train', 'mixed'), stack=True, no_transform=False):
    """Setup multiple datasets from configuration file

    Parameters
    ----------
    cfg : Config
        Configurafile with dataset information
    verbose : bool, optional
        True if information is displayed on screen, by default False
    concat_modes : tuple, optional
        Which datasets are concatenated after generated, by default ('train', 'mixed')
    stack : bool, optional
        Stack generated datasets, by default True
    no_transform : bool, optional
        If True skip data transformations, by default False

    Returns
    -------
    dict
        Generated datasets
    """

    if verbose:
        print0(pcolor('#' * 60, 'green'))
        print0(pcolor('### Preparing Datasets', 'green'))
        print0(pcolor('#' * 60, 'green'))

    font = {'color': 'yellow', 'attrs': ('bold', 'dark')}

    modes = ['train', 'mixed', 'validation']

    datasets_cfg = {}
    for key in cfg.dict.keys():
        datasets_cfg[key] = cfg.dict[key]
        for mode in modes:
            if key.startswith(mode) and key != mode and mode in cfg.dict.keys():
                datasets_cfg[key] = to_namespace(merge_dict(deepcopy(
                    cfg.dict[mode].dict), cfg.dict[key].dict))

    for key in list(datasets_cfg.keys()):
        if datasets_cfg[key].has('datasets') and \
                key not in datasets_cfg[key].datasets and \
                key not in modes:
            datasets_cfg.pop(key)
    for key in list(datasets_cfg.keys()):
        if datasets_cfg[key].has('datasets'):
            val = datasets_cfg[key].dict
            val.pop('datasets')
            datasets_cfg[key] = to_namespace(val)

    datasets = {}
    for key, val in list(datasets_cfg.items()):
        if 'name' in val.dict.keys():
            if verbose:
                print0(pcolor('###### {}'.format(key), **font))
            datasets[key], datasets_cfg[key] = setup_dataset(
                val, verbose=verbose, no_transform=no_transform)
            for mode in concat_modes:
                if key.startswith(mode):
                    if len(datasets[key]) > 1:
                        datasets[key] = ConcatDataset(datasets[key])
                    if is_list(datasets[key]):
                        datasets[key] = datasets[key][0]
        else:
            datasets_cfg.pop(key)

    if stack:
        datasets = stack_datasets(datasets)

    modes = ['train', 'mixed', 'validation', 'test']
    reduced_datasets_cfg = {key: [] for key in modes}
    for key, val in datasets_cfg.items():
        for mode in modes:
            if key.startswith(mode):
                reduced_datasets_cfg[mode].append(val)
    for key in list(reduced_datasets_cfg.keys()):
        reduced_datasets_cfg[key] = flatten(reduced_datasets_cfg[key])
        if len(reduced_datasets_cfg[key]) == 0:
            reduced_datasets_cfg.pop(key)
    datasets_cfg = reduced_datasets_cfg

    if 'train' in datasets_cfg:
        datasets_cfg['train'] = datasets_cfg['train'][0]

    return datasets, datasets_cfg


def setup_metrics(cfg):
    """Setup metrics from configuration file"""

    methods = {
        'rgb': ImageEvaluation,
        'depth': DepthEvaluation,
        'optical_flow': OpticalFlowEvaluation,
        'extrinsics': ExtrinsicsEvaluation,
    }

    available_tasks = [key for key in cfg.dict.keys() if key != 'tasks']
    requested_tasks = cfg_has(cfg, 'tasks', available_tasks)
    tasks = [task for task in available_tasks if task in requested_tasks and task.split('|')[0] in methods]

    return {task: methods[task.split('|')[0]](cfg.dict[task]) for task in tasks}


def worker_init_fn(worker_id):
    """Function to initialize workers"""
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)


def get_datasampler(dataset, shuffle):
    """Distributed data sampler"""
    return torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=shuffle,
        num_replicas=world_size(), rank=rank())


def no_collate(batch):
    """Dummy function to avoid collate_fn in dataloader"""
    return batch


@iterate12
def setup_dataloader(dataset, cfg, mode):
    """
    Create a dataloader class

    Parameters
    ----------
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the dataloader
    dataset : Dataset
        List of datasets from which to create dataloaders
    cfg : CfgNode
        Model configuration (cf. configs/default_config.py)

    Returns
    -------
    dataloaders : list of Dataloader
        List of created dataloaders for each input dataset
    """
    cfg = cfg.dataloader
    ddp = dist_mode() == 'ddp'
    shuffle = 'train' in mode
    return DataLoader(dataset,
        batch_size=cfg_has(cfg, 'batch_size', 1),
        pin_memory=cfg_has(cfg, 'pin_memory', True),
        num_workers=cfg_has(cfg, 'num_workers', 8),
        worker_init_fn=worker_init_fn,
        shuffle=False if ddp else shuffle,
        sampler=get_datasampler(dataset, shuffle=shuffle) if ddp else None,
        collate_fn=None if cfg_has(cfg, 'collate', True) else no_collate,
    )


def reduce(data, modes, train_modes):
    """ Reduce data to a single value per mode"""
    reduced = {
        mode: flatten([val for key, val in data.items() if mode in key])
        for mode in modes
    }
    for key, val in list(reduced.items()):
        if len(val) == 0:
            reduced.pop(key)
    for mode in keys_in(reduced, train_modes):
        reduced[mode] = reduced[mode][0]
    return reduced


def stack_datasets(datasets):
    """Stack datasets together"""

    all_modes = ['train', 'mixed', 'validation', 'test']
    train_modes = ['train', 'mixed']

    stacked_datasets = OrderedDict()

    for mode in all_modes:
        stacked_datasets[mode] = []
        for key, val in datasets.items():
            if mode in key:
                stacked_datasets[mode].append(val)
        stacked_datasets[mode] = flatten(stacked_datasets[mode])

    for mode in train_modes:
        length = len(stacked_datasets[mode])
        if length == 1:
            stacked_datasets[mode] = stacked_datasets[mode][0]
        elif length > 1:
            stacked_datasets[mode] = ConcatDataset(stacked_datasets[mode])
        for key in list(datasets.keys()):
            if key.startswith(mode) and key != mode:
                datasets.pop(key)

    return stacked_datasets
