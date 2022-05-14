# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import os

import torch.distributed as dist


def dist_mode():
    return os.getenv('DIST_MODE')


def rank():
    """Returns process rank"""
    if dist_mode() in ['cpu', 'gpu', None]:
        return 0
    elif dist_mode() == 'ddp':
        return int(os.environ['RANK'])
    else:
        raise ValueError('Wrong distributed mode {}'.format(dist_mode))


def world_size():
    """Returns world size"""
    if dist_mode() in ['cpu', 'gpu', None]:
        return 1
    elif dist_mode() == 'ddp':
        return int(os.environ['WORLD_SIZE'])
    else:
        raise ValueError('Wrong distributed mode {}'.format(dist_mode))


def on_rank_0(func):
    """Decorator to run function only on rank 0"""
    def wrapper(*args, **kwargs):
        if rank() == 0:
            return func(*args, **kwargs)
    return wrapper


@on_rank_0
def print0(string='\n'):
    """Function to print only on rank 0"""
    print(string)


def reduce_value(value, average, name):
    """
    Reduce the mean value of a tensor from all GPUs

    Parameters
    ----------
    value : torch.Tensor
        Value to be reduced
    average : Bool
        Whether values will be averaged or not
    name : String
        Value name

    Returns
    -------
    value : torch.Tensor
        reduced value
    """
    if dist_mode() == 'cpu':
        return value
    elif dist_mode() == 'gpu':
        return value
    elif dist_mode() == 'ddp':
        dist.all_reduce(tensor=value, op=dist.ReduceOp.SUM, async_op=False)
        if average:
            value /= world_size()
        return value
    else:
        raise ValueError('Wrong distributed mode {}'.format(dist_mode))
