# Copyright 2023 Toyota Research Institute.  All rights reserved.

import argparse
import importlib
import os
from argparse import Namespace

import torch
import yaml

from vidar.utils.data import make_list, num_trainable_params
from vidar.utils.distributed import print0
from vidar.utils.logging import pcolor
from vidar.utils.networks import load_checkpoint
from vidar.utils.types import is_dict, is_list, is_namespace, is_module_dict


def cfg_has(*args):
    """Check if key is in configuration"""
    if len(args) == 2:
        cfg, name = args
        if not is_list(name):
            return name in cfg.__dict__.keys()
        else:
            return all([n in cfg.__dict__.keys() for n in name])
    elif len(args) == 3:
        cfg, name, default = args
        has = name in cfg.__dict__.keys()
        return cfg.__dict__[name] if has else default
    else:
        raise ValueError('Wrong number of arguments for cfg_has')


def cfg_add_to_dict(dic, cfg, key, i=None):
    """Add configuration key to dictionary"""
    if cfg_has(cfg, key):
        dic[key] = cfg.__dict__[key] if i is None \
            else cfg.__dict__[key][0] if len(cfg.__dict__[key]) == 1 \
            else cfg.__dict__[key][i]


def cfg_from_dict(dic):
    """Generate configuration from dictionary"""
    for key, val in dic.items():
        if is_dict(val):
            dic[key] = cfg_from_dict(val)
    return Config(**dic)


def update_cfg(cfg):
    """Update configuration with new information"""
    if not torch.cuda.is_available():
        cfg.setup.grad_scaler = False
    return cfg


def to_namespace(data):
    """Convert to namespace"""
    for key in data.keys():
        if is_dict(data[key]):
            data[key] = to_namespace(data[key])
    return Config(**data)


def merge_dict(default, config):
    """Merge two dictionaries"""
    if is_namespace(default):
        default = default.dict
    if is_namespace(config):
        config = config.dict
    for key in config.keys():
        if key not in default.keys():
            default[key] = {}
        if is_namespace(config[key]) or is_dict(config[key]):
            default[key] = merge_dict(default[key], config[key])
        else:
            default[key] = config[key]
    return default


def update_from_kwargs(cfg, **kwargs):
    """Update configuration from kwargs"""
    if kwargs is not None:
        for key, val in kwargs.items():
            key_split = key.split('.')
            dic = cfg.__dict__
            for k in key_split[:-1]:
                dic = dic[k].__dict__
            dic[key_split[-1]] = val
    return cfg


def recursive_recipe(cfg, super_key=None):
    """Create configuration from recipe recursively"""
    for key in list(cfg.keys()):
        if is_dict(cfg[key]):
            cfg[key] = recursive_recipe(cfg[key], super_key=key)
        elif key.startswith('recipe'):
            recipe = 'configs/recipes/' + cfg.pop(key)
            if '|' in recipe:
                recipe, super_key = recipe.split('|')
            recipe = read_config(recipe + '.yaml')
            while '.' in super_key:
                split = super_key.split('.')
                recipe = recipe.__dict__[split[0]]
                super_key = '.'.join(split[1:])
            recipe = recipe.__dict__[super_key].__dict__
            cfg = merge_dict(recipe, cfg)
    return cfg


def read_config(path, **kwargs):
    """Read configuration from file"""
    with open(path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    config = recursive_recipe(config)
    cfg = to_namespace(config)
    if kwargs is not None:
        cfg = update_from_kwargs(cfg, **kwargs)
    return cfg


def is_recursive(val):
    """Check if configuration entry is recursive"""
    return 'file' in val.__dict__.keys()


def get_folder_name(path, mode, root='vidar/arch'):
    """Get folder and name from configuration path"""
    folder, name = os.path.dirname(path), os.path.basename(path)
    folder = os.path.join(root, mode, folder)
    if folder.endswith('/'):
        folder = folder[:-1]
    return folder, name


def recursive_assignment(model, cfg, mode, n=10, verbose=True):
    """Recursively assign information from a configuration"""
    font = {'color': 'yellow', 'attrs': ('dark',)}
    nested = is_module_dict(model)
    for key, val in cfg.__dict__.items():
        cls = cfg.__dict__[key]
        if is_namespace(cls):
            if is_recursive(val):
                folder, name = get_folder_name(val.file, mode)
                if nested:
                    model[key] = load_class(name, folder)(cls)
                    model_key = model[key]
                else:
                    getattr(model, mode)[key] = load_class(name, folder)(cls)
                    model_key = getattr(model, mode)[key]
                if verbose:
                    string = '#' * n + ' {}'.format(model_key.__class__.__name__)
                    num_params = num_trainable_params(model_key)
                    if num_params > 0:
                        string += f' ({num_params:,} parameters)'
                    print0(pcolor(string, **font))
                if cfg_has(val, 'checkpoint'):
                    load_checkpoint(model_key, val.checkpoint, strict=False, verbose=verbose, prefix=key)
                recursive_assignment(model_key, cls, mode, n=n+5, verbose=verbose)
            elif nested:
                model[key] = torch.nn.ModuleDict()
                recursive_assignment(model[key], cls, mode, n=n, verbose=verbose)


def load_class(filename, paths, concat=True, methodname=None):
    """
    Look for a file in different locations and return its method with the same name
    Optionally, you can use concat to search in path.filename instead

    Parameters
    ----------
    filename : str
        Name of the file we are searching for
    paths : str or list of str
        Folders in which the file will be searched
    concat : bool
        Flag to concatenate filename to each path during the search
    methodname : str or list of str
        Method name (If None, use filename
                     If it's a string, use it as the methodname
                     If it's a list, use the first methodname found)

    Returns
    -------
    method : Function
        Loaded method
    """
    # If method name is not given, use filename
    methodname = make_list(filename if methodname is None else methodname)
    # for each path in paths
    for path in make_list(paths):
        # Create full path
        path = path.replace('/', '.').replace("\\",'.')
        full_path = '{}.{}'.format(path, filename) if concat else path
        # Get module
        module = importlib.import_module(full_path)
        # Try all method names
        for name in methodname:
            method = getattr(module, name, None)
            # Return if found
            if method is not None:
                return method
    # Didn't find anything
    raise ValueError('Unknown class {}'.format(filename))


def get_from_cfg_list(cfg, key, idx):
    """Get configuration entry from list"""
    if key not in cfg.__dict__.keys():
        return None
    data = cfg.__dict__[key]
    return data if not is_list(data) else data[idx] if len(data) > 1 else data[0]


def dataset_prefix(cfg, idx=0):
    """Creates dataset prefix from configuration"""
    # Dataset path is always available
    # prefix = cfg.name[idx]
    prefix = '{}'.format(os.path.splitext(get_from_cfg_list(cfg, 'path', idx).split('/')[-1])[0])
    # If split is available
    val = cfg.has('split', None)
    if val is not None:
        prefix += '-{}'.format(os.path.splitext(os.path.basename(val))[0])
    # If input depth type is available
    val = cfg.has('input_depth_type', None)
    if val is not None and val not in [None, '']:
        prefix += '-+{}'.format(val)
    # If depth type is available
    val = cfg.has('depth_type', None)
    if val is not None and val not in [None, '']:
        prefix += '-{}'.format(val)
    # If there is camera information
    val = cfg.has('cameras', None)
    if val is not None and is_list(val) and len(val) > 0:
        prefix += '-cam{}'.format(val[0])
    if cfg.has('evaluation') and cfg.evaluation.has('suffix'):
        prefix += '-{}'.format(cfg.evaluation.suffix)
    # Return prefix
    return prefix


class Config(Namespace):
    """Configuration class, used to store and access parameters"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_file(file):
        return read_config(file)

    @property
    def dict(self):
        return self.__dict__

    def keys(self):
        return self.dict.keys()

    def items(self):
        return self.dict.items()

    def values(self):
        return self.dict.values()

    def has(self, *args):
        return cfg_has(self, *args)

    def get(self, key, default):
        return self.dict[key] if key in self.keys() else default

    def pop(self, key, default):
        return self.get(key, default)

    def __getitem__(self, key):
        return self.dict[key]
