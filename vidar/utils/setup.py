# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from vidar.datasets.utils.transforms import get_transforms
from vidar.metrics.depth import DepthEvaluation
from vidar.utils.config import get_folder_name, load_class, \
    recursive_assignment, cfg_has, cfg_add_to_dict, get_from_cfg_list
from vidar.utils.config import merge_dict, to_namespace
from vidar.utils.data import flatten, keys_in
from vidar.utils.decorators import iterate1
from vidar.utils.distributed import print0, rank, world_size, dist_mode
from vidar.utils.logging import pcolor
from vidar.utils.networks import load_checkpoint, save_checkpoint
from vidar.utils.types import is_namespace


def setup_arch(cfg, checkpoint=None, verbose=False):
    """
    Set architecture up for training/inference

    Parameters
    ----------
    cfg : Config
        Configuration file
    checkpoint : String
        Checkpoint to be loaded
    verbose : Bool
        Print information on screen

    Returns
    -------
    model: nn.Module
        Model ready to go
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

    if cfg_has(cfg, 'model'):
        if verbose:
            print0(pcolor('###### Model:', **font2))
            print0(pcolor('######### %s' % model.__class__.__name__, **font1))
        recursive_assignment(model, cfg.model, 'models', verbose=verbose)

    if cfg_has(cfg, 'networks'):
        if verbose:
            print0(pcolor('###### Networks:', **font2))
        recursive_assignment(model, cfg.networks, 'networks', verbose=verbose)

    if cfg_has(cfg, 'losses'):
        if verbose:
            print0(pcolor('###### Losses:', **font2))
        recursive_assignment(model, cfg.losses, 'losses', verbose=verbose)

    if checkpoint is not None:
        model = load_checkpoint(model, checkpoint,
                                strict=True, verbose=verbose)
    elif cfg_has(cfg.model, 'checkpoint'):
        model = load_checkpoint(model, cfg.model.checkpoint,
                                strict=cfg.model.has('checkpoint_strict', False), verbose=verbose)

    if cfg.model.has('checkpoint_save'):
        save_checkpoint(cfg.model.checkpoint_save, model)

    return model


def setup_dataset(cfg, root='vidar/datasets', verbose=False):
    """
    Set dataset up for training/inference

    Parameters
    ----------
    cfg : Config
        Configuration file
    root : String
        Where the dataset is located
    verbose : Bool
        Print information on screen

    Returns
    -------
    dataset : Dataset
        Dataset ready to go
    """
    shared_keys = ['context', 'labels', 'labels_context']

    num_datasets = 0
    for key, val in cfg.__dict__.items():
        if key not in shared_keys and not is_namespace(val):
            num_datasets = max(num_datasets, len(val))

    datasets = []
    for i in range(num_datasets):
        args = {}
        for key, val in cfg.__dict__.items():
            if not is_namespace(val):
                cfg_add_to_dict(args, cfg, key, i if key not in shared_keys else None)

        args['data_transform'] = get_transforms('train', cfg.augmentation) \
            if cfg_has(cfg, 'augmentation') else get_transforms('none')

        name = get_from_cfg_list(cfg, 'name', i)
        repeat = get_from_cfg_list(cfg, 'repeat', i)
        cameras = get_from_cfg_list(cfg, 'cameras', i)

        context = cfg.context
        labels = cfg.labels

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
                string += f' | cameras {cameras}'.replace(', ', ',')
            if cfg_has(cfg, 'labels'):
                string += f' | labels {labels}'.replace(', ', ',')
            print0(pcolor(string , color='yellow', attrs=('dark',)))

        datasets.append(dataset)

    return datasets


def setup_datasets(cfg, verbose=False, concat_modes=('train', 'mixed'), stack=True):
    """
    Set multiple datasets up for training/inference

    Parameters
    ----------
    cfg : Config
        Configuration file
    verbose : Bool
        Print information on screen
    concat_modes : String
        Which dataset modes are going to be concatenated into a single one
    stack : Bool
        Whether datasets are stacked together

    Returns
    -------
    datasets : Dict
        Datasets ready to go
    datasets_cfg : Dict
        Dataset configurations
    """
    if verbose:
        print0(pcolor('#' * 60, 'green'))
        print0(pcolor('### Preparing Datasets', 'green'))
        print0(pcolor('#' * 60, 'green'))

    font = {'color': 'yellow', 'attrs': ('bold', 'dark')}

    datasets_cfg = {}
    for key in cfg.__dict__.keys():
        datasets_cfg[key] = cfg.__dict__[key]
        for mode in ['train', 'validation']:
            if key.startswith(mode) and key != mode and mode in cfg.__dict__.keys():
                datasets_cfg[key] = to_namespace(merge_dict(deepcopy(
                    cfg.__dict__[mode].__dict__), cfg.__dict__[key].__dict__))

    datasets = {}
    for key, val in list(datasets_cfg.items()):
        if 'name' in val.__dict__.keys():
            if verbose:
                print0(pcolor('###### {}'.format(key), **font))
            datasets[key] = setup_dataset(val, verbose=verbose)
            datasets_cfg[key] = [datasets_cfg[key]] * len(datasets[key])
            for mode in concat_modes:
                if key.startswith(mode) and len(datasets[key]) > 1:
                    datasets[key] = ConcatDataset(datasets[key])
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
    """
    Set metrics up for evaluation

    Parameters
    ----------
    cfg : Config
        Configuration file

    Returns
    -------
    tasks : Dict
        Dictionary containing metric classes for requested tasks
    """

    methods = {
        'depth': DepthEvaluation,
    }

    available_tasks = [key for key in cfg.__dict__.keys() if key is not 'tasks']
    requested_tasks = cfg_has(cfg, 'tasks', available_tasks)
    tasks = [task for task in available_tasks if task in requested_tasks and task in methods]

    return {task: methods[task](cfg.__dict__[task]) for task in tasks}


def worker_init_fn(worker_id):
    """Function to initialize workers"""
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)


def get_datasampler(dataset, shuffle):
    """Return distributed data sampler"""
    return torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=shuffle,
        num_replicas=world_size(), rank=rank())


def no_collate(batch):
    """Dummy function to use when dataset is not to be collated"""
    return batch


@iterate1
def setup_dataloader(dataset, cfg, mode):
    """
    Create a dataloader class

    Parameters
    ----------
    mode : String {'train', 'validation', 'test'}
        Mode from which we want the dataloader
    dataset : Dataset
        List of datasets from which to create dataloaders
    cfg : Config
        Model configuration (cf. configs/default_config.py)

    Returns
    -------
    dataloaders : list[Dataloader]
        List of created dataloaders for each input dataset
    """
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
    """
    Reduce dictionary values

    Parameters
    ----------
    data : Dict
        Dictionary with data for reduction
    modes : String
        Data mode ('train', 'validation', 'test')
    train_modes : list[String]
        Which modes are training modes

    Returns
    -------
    reduced : Dict
        Dictionary with reduced information
    """
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
    """
    Stack datasets together for training/validation

    Parameters
    ----------
    datasets : Dict
        Dictionary containing datasets

    Returns
    -------
    stacked_datasets: : Dict
        Dictionary containing stacked datasets
    """
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
