# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import argparse
import os
from functools import partial

from termcolor import colored

from vidar.utils.distributed import on_rank_0


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : String
        String that will be colored
    color : String
        Color to use
    on_color : String
        Background color to use
    attrs : list[String]
        Different attributes for the string

    Returns
    -------
    string: String
        Colored string
    """
    return colored(string, color, on_color, attrs)


@on_rank_0
def print_config(config):
    """
    Prints header for model configuration

    Parameters
    ----------
    config : Config
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

    # Color partial functions
    pcolor1 = partial(pcolor, color='blue', attrs=('bold', 'dark'))
    pcolor2 = partial(pcolor, color='blue', attrs=('bold',))
    # Config and name
    line = pcolor1('#' * 120)
    # if 'default' in config.__dict__.keys():
    #     path = pcolor1('### Config: ') + \
    #            pcolor2('{}'.format(config.default.replace('/', '.'))) + \
    #            pcolor1(' -> ') + \
    #            pcolor2('{}'.format(config.config.replace('/', '.')))
    # if 'name' in config.__dict__.keys():
    #     name = pcolor1('### Name: ') + \
    #            pcolor2('{}'.format(config.name))
    #     # Add wandb link if available
    #     if not config.wandb.dry_run:
    #         name += pcolor1(' -> ') + \
    #                 pcolor2('{}'.format(config.wandb.url))
    #     # Add s3 link if available
    #     if config.checkpoint.s3_path is not '':
    #         name += pcolor1('\n### s3:') + \
    #                 pcolor2(' {}'.format(config.checkpoint.s3_url))
    #     # # Create header string
    #     # header = '%s\n%s\n%s\n%s' % (line, path, name, line)

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
    debug : Bool
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
        self.values.append(value)
        if len(self.values) > self.n_max:
            self.values.pop(0)
        return self.get()

    def get(self):
        """Get current average"""
        return sum(self.values) / len(self.values)

    def reset(self):
        """Reset meter"""
        self.values.clear()

    def get_and_reset(self):
        """Get current average and reset"""
        average = self.get()
        self.reset()
        return average
