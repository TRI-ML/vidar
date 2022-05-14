# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import os
from datetime import datetime

import numpy as np
import torch

from vidar.utils.config import cfg_has
from vidar.utils.logging import pcolor


class ModelCheckpoint:
    """
    Class for model checkpointing

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    verbose : Bool
        Print information on screen if enabled
    """
    def __init__(self, cfg, verbose=False):
        super().__init__()

        # Create checkpoint folder
        self.folder = cfg_has(cfg, 'folder', None)
        self.name = cfg_has(cfg, 'name', datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss"))
        if self.folder:
            self.path = os.path.join(self.folder, self.name)
            os.makedirs(self.path, exist_ok=True)
        else:
            self.path = None

        # Exclude folders
        self.excludes = ['sandbox']

        # If there is no folder, only track metrics
        self.tracking_only = self.path is None

        # Store arguments
        self.keep_top = cfg_has(cfg, 'keep_top', -1)
        self.dataset = cfg_has(cfg, 'dataset', [])
        self.monitor = cfg_has(cfg, 'monitor', [])
        self.mode = cfg_has(cfg, 'mode', [])

        # Number of metrics to track
        self.num_tracking = len(self.mode)

        # Prepare s3 bucket
        if cfg_has(cfg, 's3_bucket'):
            self.s3_path = f's3://{cfg.s3_bucket}/{self.name}'
            self.s3_url = f'https://s3.console.aws.amazon.com/s3/buckets/{self.s3_path[5:]}'
        else:
            self.s3_path = self.s3_url = None

        # Get starting information
        self.torch_inf = torch.tensor(np.Inf)
        mode_dict = {
            'min': (self.torch_inf, 'min'),
            'max': (-self.torch_inf, 'max'),
            'auto': (-self.torch_inf, 'max') if \
                'acc' in self.monitor or \
                'a1' in self.monitor or \
                'fmeasure' in self.monitor \
                else (self.torch_inf, 'min'),
        }

        if self.mode:
            self.top = [[] for _ in self.mode]
            self.store_val = [[] for _ in self.mode]
            self.previous = [0 for _ in self.mode]
            self.best = [mode_dict[m][0] for m in self.mode]
            self.mode = [mode_dict[m][1] for m in self.mode]
        else:
            self.top = []

        # Print if requested
        if verbose:
            self.print()

        # Save if requested
        if cfg_has(cfg, 'save_code', False):
            self.save_code()
            if self.s3_url:
                self.sync_s3(verbose=False)

    def print(self):
        """Print information on screen"""
        font_base = {'color': 'red', 'attrs': ('bold', 'dark')}
        font_name = {'color': 'red', 'attrs': ('bold',)}
        font_underline = {'color': 'red', 'attrs': ('underline',)}

        print(pcolor('#' * 60, **font_base))
        if self.path:
            print(pcolor('### Checkpoint: ', **font_base) + \
                  pcolor('{}/{}'.format(self.folder, self.name), **font_name))
            if self.s3_url:
                print(pcolor('### ', **font_base) + \
                      pcolor('{}'.format(self.s3_url), **font_underline))
        else:
            print(pcolor('### Checkpoint: ', **font_base) + \
                  pcolor('Tracking only', **font_name))
        print(pcolor('#' * 60, **font_base))

    @staticmethod
    def save_model(wrapper, name, epoch):
        """Save model"""
        torch.save({
            'config': wrapper.cfg, 'epoch': epoch,
            'state_dict': wrapper.arch.state_dict(),
        }, name)

    @staticmethod
    def del_model(name):
        """Delete model"""
        if os.path.isfile(name):
            os.remove(name)

    def save_code(self):
        """Save code in the models folder"""
        excludes = ' '.join([f'--exclude {exclude}' for exclude in self.excludes])
        os.system(f"tar cfz {self.path}/{self.name}.tar.gz {excludes} *")

    def sync_s3(self, verbose=True):
        """Sync saved models with the s3 bucket"""

        font_base = {'color': 'magenta', 'attrs': ('bold', 'dark')}
        font_name = {'color': 'magenta', 'attrs': ('bold',)}

        if verbose:
            print(pcolor('Syncing ', **font_base) +
                  pcolor('{}'.format(self.path), **font_name) +
                  pcolor(' -> ', **font_base) +
                  pcolor('{}'.format(self.s3_path), **font_name))

        command = f'aws s3 sync {self.path} {self.s3_path} ' \
                  f'--acl bucket-owner-full-control --quiet --delete'
        os.system(command)

    def print_improvements(self, key, value, idx, is_best):
        """Print color-coded changes in tracked metrics"""

        font1 = {'color': 'cyan', 'attrs':('dark', 'bold')}
        font2 = {'color': 'cyan', 'attrs': ('bold',)}
        font3 = {'color': 'yellow', 'attrs': ('bold',)}
        font4 = {'color': 'green', 'attrs': ('bold',)}
        font5 = {'color': 'red', 'attrs': ('bold',)}

        current_inf = self.best[idx] == self.torch_inf or \
                      self.best[idx] == -self.torch_inf

        print(
            pcolor(f'{key}', **font2) + \
            pcolor(f' ({self.mode[idx]}) : ', **font1) + \
            ('' if current_inf else
             pcolor('%3.6f' % self.previous[idx], **font3) +
             pcolor(f' -> ', **font1)) + \
            (pcolor('%3.6f' % value, **font4) if is_best else
             pcolor('%3.6f' % value, **font5)) +
            ('' if current_inf else
             pcolor(' (%3.6f)' % self.best[idx], **font2))
        )

    def save(self, wrapper, epoch, verbose=True):
        """Save model"""
        # Do nothing if no path is provided
        if self.path:

            name = '%03d.ckpt' % epoch
            folder = os.path.join(self.path, 'models')

            os.makedirs(folder, exist_ok=True)
            folder_name = os.path.join(folder, name)
            self.save_model(wrapper, folder_name, epoch)
            self.top.append(folder_name)
            if 0 < self.keep_top < len(self.top):
                self.del_model(self.top.pop(0))
            if self.s3_url:
                self.sync_s3(verbose=False)

        if verbose:
            print()

    def check_and_save(self, wrapper, metrics, prefixes, epoch, verbose=True):
        """Check if model should be saved and maybe save it"""
        # Not tracking any metric, save every iteration
        if self.num_tracking == 0:
            # Do nothing if no path is provided
            if self.path:

                name = '%03d.ckpt' % epoch
                folder = os.path.join(self.path, 'models')

                os.makedirs(folder, exist_ok=True)
                folder_name = os.path.join(folder, name)
                self.save_model(wrapper, folder_name, epoch)
                self.top.append(folder_name)
                if 0 < self.keep_top < len(self.top):
                    self.del_model(self.top.pop(0))
                if self.s3_url:
                    self.sync_s3(verbose=False)

        # Check if saving for every metric
        else:

            for idx in range(self.num_tracking):

                key = '{}-{}'.format(prefixes[self.dataset[idx]], self.monitor[idx])
                value = metrics[key]

                if self.mode[idx] == 'min':
                    is_best = value < self.best[idx]
                    will_store = len(self.store_val[idx]) < self.keep_top or \
                                 value < np.max(self.store_val[idx])
                    store_idx = 0 if len(self.store_val[idx]) == 0 else int(np.argmax(self.store_val[idx]))
                else:
                    is_best = value > self.best[idx]
                    will_store = len(self.store_val[idx]) < self.keep_top or \
                                 value > np.min(self.store_val[idx])
                    store_idx = 0 if len(self.store_val[idx]) == 0 else int(np.argmin(self.store_val[idx]))

                if verbose:
                    self.print_improvements(key, value, idx, is_best)

                self.previous[idx] = value

                if is_best:
                    self.best[idx] = value

                if is_best or will_store:

                    if self.path:

                        name = '%03d_%3.6f.ckpt' % (epoch, value)
                        folder = os.path.join(self.path, key)

                        os.makedirs(folder, exist_ok=True)
                        folder_name = os.path.join(folder, name)
                        self.save_model(wrapper, folder_name, epoch)
                        self.top[idx].append(folder_name)
                        self.store_val[idx].append(value)
                        if 0 < self.keep_top < len(self.top[idx]):
                            self.del_model(self.top[idx].pop(store_idx))
                            self.store_val[idx].pop(store_idx)
                        if self.s3_url:
                            self.sync_s3(verbose=False)

        if verbose:
            print()
