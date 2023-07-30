# Copyright 2023 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict
from functools import partial

import numpy as np
import torch

from vidar.utils.distributed import reduce_value
from vidar.utils.tensor import same_shape, interpolate
from vidar.utils.data import make_list


class BaseEvaluation:
    def __init__(self, cfg, name, task, metrics):
        self.name = name
        self.task = task
        self.width = 42 + 11 * len(metrics)
        self.metrics = metrics
        self.modes = ['']

        self.font1 = {'color': 'magenta', 'attrs': ('bold',)}
        self.font2 = {'color': 'cyan', 'attrs': ()}

        self.nearest = partial(interpolate, scale_factor=None, mode='nearest')
        self.bilinear = partial(interpolate, scale_factor=None, mode='bilinear')

        self.only_first = cfg.has('only_first', False)
        self.strict = cfg.has('strict', False)

    @property
    def horz_line(self):
        """Prints a horizontal line of asterisks"""
        return '|{:<}|'.format('*' * self.width)

    @property
    def metr_line(self):
        """Prints a line with metric names"""
        return '| {:^40} |' + ' {:^8} |' * len(self.metrics)

    @property
    def outp_line(self):
        """Prints a line with metric values"""
        return '{:<40}' + ' | {:^8.3f}' * len(self.metrics)

    @staticmethod
    def wrap(string):
        """Wraps a string in vertical bars"""
        return '| {} |'.format(string)

    def check_name(self, key):
        """Checks if a key belongs to this evaluation"""
        return key.startswith(self.name) or \
               key.startswith('fwd_' + self.name) or \
               key.startswith('bwd_' + self.name)

    def reduce_fn(self, *args, **kwargs):
        """Reduce function for distributed evaluation"""
        raise NotImplementedError('reduce_fn not implemented for {}'.format(self.__name__))

    def populate_metrics_dict(self, *args, **kwargs):
        """Populates a dictionary with metrics"""
        raise NotImplementedError('create_dict_key not implemented for {}'.format(self.__name__))

    def print(self, *args, **kwargs):
        """Prints evaluation results"""
        raise NotImplementedError('print not implemented for {}'.format(self.__name__))

    @staticmethod
    def interp(dst, src, fn):
        """Helper function to interpolate src to dst shape"""
        if dst is None:
            return dst
        assert dst.dim() == src.dim()
        if dst.dim() == 4 and not same_shape(dst.shape, src.shape):
            dst = fn(dst, size=src)
        return dst

    def interp_bilinear(self, dst, src):
        """Helper function for bilinear interpolation"""
        return self.interp(dst, src, self.bilinear)

    def interp_nearest(self, dst, src):
        """Helper function for nearest neighbor interpolation"""
        return self.interp(dst, src, self.nearest)

    @staticmethod
    def ctx_str(ctx):
        """Prints a context as a string"""
        replaces = [[' ', ''], ['(', ''], [')', ''], [',', '_']]
        ctx = str(ctx)
        for replace in replaces:
            ctx = ctx.replace(replace[0], replace[1])
        return ctx

    def suffix_single(self, tgt, i):
        """Parse a suffix for a single target"""
        return '(%s)' % self.ctx_str(tgt) + ('_%d' % i if not self.only_first else '')

    def suffix_single2(self, tgt, ctx, i):
        """Parse a suffix for a single target and context"""
        return '(%s)_(%s)' % (self.ctx_str(tgt), self.ctx_str(ctx)) + ('_%d' % i if not self.only_first else '')

    def suffix_multi(self, ctx, i, j):
        """Parse a suffix for multiple targets"""
        return '(%s_%d)' % (self.ctx_str(ctx), j) + ('_%d' % i if not self.only_first else '')

    def suffix_ctx(self, tgt, ctx, i):
        """Parse a suffix for a single target and context"""
        return '(%s)_(%s)' % (self.ctx_str(tgt), self.ctx_str(ctx)) + ('_%d' % i if not self.only_first else '')

    def reduce(self, output, dataloaders, prefixes, task, verbose=True):
        """Reduce metrics for all datasets"""
        reduced_data = self.reduce_metrics(output, dataloaders)
        metrics_dict = self.create_metrics_dict(reduced_data, prefixes)
        if verbose:
            self.print(reduced_data, prefixes, task)
        return metrics_dict

    def create_metrics_dict(self, reduced_data, prefixes):
        """Create a dictionary with metrics from reduced data"""
        # Create metrics dictionary
        metrics_dict = {}
        # For all datasets
        for n, metrics in enumerate(reduced_data):
            if metrics:  # If there are calculated metrics
                self.populate_metrics_dict(metrics, metrics_dict, prefixes[n])
        # Return metrics dictionary
        return metrics_dict

    def reduce_metrics(self, dataset_outputs, datasets, ontology=None):
        # List storing metrics for all datasets
        all_metrics_dict = []
        # Loop over all datasets and all batches
        for batch_outputs, dataset in zip(make_list(dataset_outputs), datasets):
            # Initialize metrics dictionary
            metrics_dict = OrderedDict()
            # Get length, names and dimensions
            length = len(dataset)
            names, dims = [], []
            for batch_output in batch_outputs:
                for key in batch_output.keys():
                    if key not in names:
                        names.append(key)
                        dims.append(tuple(batch_output[key].size()))
            # Get data device
            device = batch_outputs[0]['idx'].device
            # Reduce relevant metrics
            for name, dim in zip(names, dims):
                if name == 'idx':
                    continue

                metrics = torch.zeros([length] + list(dim), device=device)

                # Count how many times each sample was seen
                seen = torch.zeros(length, device=device)
                for output in batch_outputs:
                    if name in output:
                        seen[output['idx']] += 1
                seen = reduce_value(seen, average=False, name='idx')
                if self.strict:
                    assert not np.any(seen.cpu().numpy() == 0), \
                        'Not all samples were seen during evaluation'

                for output in batch_outputs:
                    if name in output:
                        metrics[output['idx']] = output[name]
                metrics = reduce_value(metrics, average=False, name=name)
                metrics_dict[name] = [
                    self.reduce_fn(metrics, seen), (seen > 0).sum(), len(seen)
                ]
            # Append metrics dictionary to the list
            all_metrics_dict.append(metrics_dict)

        # Return list of metrics dictionary
        return all_metrics_dict

    def get_valid_keys_and_metrics(self, metrics, task):
        """Parse valid keys and metrics from a dictionary"""
        valid_keys_and_metrics, seen = {}, [0, 0]
        for key, metric in sorted(metrics.items()):
            if key.startswith(self.name) and task.count('|') == key.count('|') - 1 and task.split('|')[-1] in key:
                valid_keys_and_metrics[key] = metric[0]
                seen = metric[1:3]
        return valid_keys_and_metrics, seen
