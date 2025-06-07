# Copyright 2023 Toyota Research Institute.  All rights reserved.

import os
from termcolor import colored
import argparse

from knk_vision.vidar.vidar.utils.distributed import on_rank_0
from functools import partial


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


@on_rank_0
def print_config(config):
    """
    Prints header for model configuration

    Parameters
    ----------
    config : CfgNode
        Model configuration
    """
    header_colors = {
        0: ('red', ('bold', 'dark')),
        1: ('cyan', ('bold','dark')),
        2: ('green', ('bold', 'dark')),
        3: ('green', ('bold', 'dark')),
    }
    line_colors = ('blue', ())

    # Recursive print function
    def print_recursive(rec_args, pad=3, level=0):
        # if level == 0:
        #     print(pcolor('config:',
        #                  color=header_colors[level][0],
        #                  attrs=header_colors[level][1]))
        for key, val in rec_args.__dict__.items():
            if isinstance(val, argparse.Namespace):
                print(pcolor('{} {}:'.format('-' * pad, key),
                             color=header_colors[level][0],
                             attrs=header_colors[level][1]))
                print_recursive(val, pad + 2, level + 1)
            else:
                print('{}: {}'.format(pcolor('{} {}'.format('-' * pad, key),
                                             color=line_colors[0],
                                             attrs=line_colors[1]), val))
    # Print header, config and header again
    print()
    # print(header)
    print_recursive(config)
    # print(header)
    print()


def set_debug(debug):
    """
    Enable or disable debug terminal logging

    Parameters
    ----------
    debug : bool
        Debugging flag (True to enable)
    """
    # Disable logging if requested
    if not debug:
        os.environ['NCCL_DEBUG'] = ''
        os.environ['WANDB_SILENT'] = 'true'
        # warnings.filterwarnings("ignore")
        # logging.disable(logging.CRITICAL)


class AvgMeter:
    """Average meter for logging"""
    def __init__(self, n_max=100):
        self.n_max = n_max
        self.values = []

    def __call__(self, value):
        """Append new value and returns average"""
        if value is not None:
            self.values.append(value)
            if len(self.values) > self.n_max:
                self.values.pop(0)
        return self.get()

    def get(self):
        """Get current average"""
        return 0.0 if len(self.values) == 0 else sum(self.values) / len(self.values)

    def reset(self):
        """Reset meter"""
        self.values.clear()

    def get_and_reset(self):
        """Get current average and reset"""
        average = self.get()
        self.reset()
        return average
