# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import importlib
import os
from argparse import Namespace

import torch
import yaml

from vidar.utils.data import make_list, num_trainable_params
from vidar.utils.distributed import print0
from vidar.utils.logging import pcolor
from vidar.utils.networks import load_checkpoint
from vidar.utils.types import is_dict, is_list, is_namespace


def cfg_has(*args):
    """
    Check if a key is in configuration

    Parameters
    ----------
    args : Tuple (Config, String, Value)
        Inputs:
            length 2 = configuration/name,
            length 3 = configuration/name/default

    Returns
    -------
    Flag : Bool or Value
        True/False if key is in configuration, key value/default if default is provided
    """
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
    """
    Add configuration key to dictionary

    Parameters
    ----------
    dic : Dict
        Input dictionary
    cfg : Config
        Input configuration
    key : String
        Input key
    i : Int
        Optional list index
    """
    if cfg_has(cfg, key):
        dic[key] = cfg.__dict__[key] if i is None \
            else cfg.__dict__[key][0] if len(cfg.__dict__[key]) == 1 \
            else cfg.__dict__[key][i]


def cfg_from_dict(dic):
    """
    Create configuration from dictionary

    Parameters
    ----------
    dic : Dict
        Input dictionary

    Returns
    -------
    cfg : Config
        Output configuration
    """
    for key, val in dic.items():
        if is_dict(val):
            dic[key] = cfg_from_dict(val)
    return Config(**dic)


def update_cfg(cfg):
    """
    Update configuration with hard-coded information

    Parameters
    ----------
    cfg : Config
        Input configuration

    Returns
    -------
    cfg : Config
        Updated configuration
    """
    if not torch.cuda.is_available():
        cfg.setup.grad_scaler = False
    return cfg


def to_namespace(data):
    """
    Convert dictionary to namespace

    Parameters
    ----------
    data : Dict or Config
        Input dictionary

    Returns
    -------
    cfg : Config
        Output configuration
    """
    for key in data.keys():
        if is_dict(data[key]):
            data[key] = to_namespace(data[key])
    return Config(**data)


def merge_dict(default, config):
    """
    Merge two dictionaries

    Parameters
    ----------
    default : Dict
        Dictionary with default values
    config : Dict
        Dictionary with values to update

    Returns
    -------
    cfg : Dict
        Updated dictionary
    """
    if is_namespace(default):
        default = default.__dict__
    for key in config.keys():
        if key not in default.keys():
            default[key] = {}
        if not is_dict(config[key]):
            default[key] = config[key]
        else:
            default[key] = merge_dict(default[key], config[key])
    return default


def update_from_kwargs(cfg, **kwargs):
    """
    Update configuration based on keyword arguments

    Parameters
    ----------
    cfg : Config
        Input configuration
    kwargs : Dict
        Keyword arguments

    Returns
    -------
    cfg : Config
        Updated configuration
    """
    if kwargs is not None:
        for key, val in kwargs.items():
            key_split = key.split('.')
            dic = cfg.__dict__
            for k in key_split[:-1]:
                dic = dic[k].__dict__
            dic[key_split[-1]] = val
    return cfg


def recursive_recipe(cfg, super_key=None):
    """
    Add recipe parameters to configuration

    Parameters
    ----------
    cfg : Config
        Input configuration
    super_key : String
        Which recipe entry to use

    Returns
    -------
    cfg : Config
        Updated configuration
    """
    for key in list(cfg.keys()):
        if is_dict(cfg[key]):
            cfg[key] = recursive_recipe(cfg[key], super_key=key)
        elif key == 'recipe':
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
    """
    Create configuration from file

    Parameters
    ----------
    path : String
        Configuration path
    kwargs : Dict
        Keyword arguments to update configuration

    Returns
    -------
    cfg : Config
        Output configuration
    """
    """Read configuration from file"""
    with open(path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    config = recursive_recipe(config)
    cfg = to_namespace(config)
    if kwargs is not None:
        cfg = update_from_kwargs(cfg, **kwargs)
    return cfg


def is_recursive(val):
    """
    Check if configuration entry is recursive

    Parameters
    ----------
    val : Config
        Input Configuration

    Returns
    -------
    Flag : Bool
        True/False if is recursive or not
    """
    return 'file' in val.__dict__.keys()


def get_folder_name(path, mode, root='vidar/arch'):
    """
    Get folder and name from configuration path

    Parameters
    ----------
    path : String
        Input path
    mode : String
        Which mode to use (e.g., models, networks, losses)
    root : String
        Which folder to use

    Returns
    -------
    folder : String
        Output folder
    name : String
        Output name
    """
    """Get folder and name from configuration path"""
    folder, name = os.path.dirname(path), os.path.basename(path)
    folder = os.path.join(root, mode, folder)
    if folder.endswith('/'):
        folder = folder[:-1]
    return folder, name


def recursive_assignment(model, cfg, mode, verbose=True):
    """
    Recursively assign information from a configuration

    Parameters
    ----------
    model : torch.nn.Module
        Which network we are using
    cfg : Config
        Input Configuration
    mode : String
        Which mode we are using (e.g., models, networks, losses)
    verbose : Bool
        Print information on screen
    """
    font = {'color': 'yellow', 'attrs': ('dark',)}
    for key, val in cfg.__dict__.items():
        cls = cfg.__dict__[key]
        if is_namespace(cls):
            if is_recursive(val):
                folder, name = get_folder_name(val.file, mode)
                getattr(model, mode)[key] = load_class(name, folder)(cls)
                if verbose:
                    string = '######### {}'.format(getattr(model, mode)[key].__class__.__name__)
                    num_params = num_trainable_params(getattr(model, mode)[key])
                    if num_params > 0:
                        string += f' ({num_params:,} parameters)'
                    print0(pcolor(string, **font))
                if cfg_has(val, 'checkpoint'):
                    model_attr = getattr(model, mode)[key]
                    load_checkpoint(model_attr, val.checkpoint, strict=False, verbose=verbose, prefix=key)
                recursive_assignment(getattr(model, mode)[key], cls, mode, verbose=verbose)
            if key == 'blocks':
                for key2, val2 in cfg.__dict__[key].__dict__.items():
                    cls2 = cfg.__dict__[key].__dict__[key2]
                    if is_recursive(val2):
                        folder, name = get_folder_name(val2.file, 'blocks')
                        model.blocks[key2] = load_class(name, folder)(cls2)
                        recursive_assignment(model.blocks[key2], cls2, 'blocks', verbose=verbose)


def load_class(filename, paths, concat=True, methodname=None):
    """
    Look for a file in different locations and return its method with the same name
    Optionally, you can use concat to search in path.filename instead

    Parameters
    ----------
    filename : String
        Name of the file we are searching for
    paths : String or list[String]
        Folders in which the file will be searched
    concat : Bol
        Flag to concatenate filename to each path during the search
    methodname : String or list[String]
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
        path = path.replace('/', '.')
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
    """
    Get configuration value from a list

    Parameters
    ----------
    cfg : Config
        Input configuration
    key : String
        Input configuration key
    idx : Int
        List index

    Returns
    -------
    data : Value
        Key value at that index if it's a list, otherwise return the key value directly
    """
    if key not in cfg.__dict__.keys():
        return None
    data = cfg.__dict__[key]
    return data if not is_list(data) else data[idx] if len(data) > 1 else data[0]


def dataset_prefix(cfg, idx):
    """
    Create dataset prefix based on configuration information

    Parameters
    ----------
    cfg : Config
        Input configuration
    idx : Int
        Input index for information retrieval

    Returns
    -------
    prefix : String
        Dataset prefix
    """
    # Dataset path is always available
    # prefix = cfg.name[idx]
    prefix = '{}'.format(os.path.splitext(get_from_cfg_list(cfg, 'path', idx).split('/')[-1])[0])
    # If split is available
    val = get_from_cfg_list(cfg, 'split', idx)
    if val is not None:
        prefix += '-{}'.format(os.path.splitext(os.path.basename(val))[0])
    # If input depth type is available
    val = get_from_cfg_list(cfg, 'input_depth_type', idx)
    if val is not None and val not in [None, '']:
        prefix += '-+{}'.format(val)
    # If depth type is available
    val = get_from_cfg_list(cfg, 'depth_type', idx)
    if val is not None and val not in [None, '']:
        prefix += '-{}'.format(val)
    # If there is camera information
    val = get_from_cfg_list(cfg, 'cameras', idx)
    if val is not None and is_list(val) and len(val) > 0:
        prefix += '-cam{}'.format(val[0])
    # Return prefix
    return prefix


class Config(Namespace):
    """
    Configuration class for passing arguments between other classes

    Parameters
    ----------
    kwargs: Dict
        Arguments to create configuration
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_file(file):
        """Read configuration from file"""
        return read_config(file)

    @property
    def dict(self):
        """Return configuration as dictionary"""
        return self.__dict__

    def keys(self):
        """Return dictionary keys of configuration"""
        return self.dict.keys()

    def items(self):
        """Return dictionary items of configuration"""
        return self.dict.items()

    def values(self):
        """Return dictionary values of configuration"""
        return self.dict.values()

    def has(self, *args):
        """Check if configuration has certain parameters"""
        return cfg_has(self, *args)

