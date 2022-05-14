# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict
from functools import partial

import numpy as np
import torch

from vidar.utils.distributed import reduce_value
from vidar.utils.tensor import same_shape, interpolate


class BaseEvaluation:
    """
    Base class for evaluation metrics

    Parameters
    ----------
    cfg : Config
        Configuration file
    name : String
        Evaluation name
    task : String
        Task referent to the evaluation
    metrics : String
        Metrics name
    """
    def __init__(self, cfg, name, task, metrics):
        self.name = name
        self.task = task
        self.width = 32 + 11 * len(metrics)
        self.metrics = metrics
        self.modes = ['']

        self.font1 = {'color': 'magenta', 'attrs': ('bold',)}
        self.font2 = {'color': 'cyan', 'attrs': ()}

        self.nearest = partial(interpolate, scale_factor=None, mode='nearest', align_corners=None)
        self.bilinear = partial(interpolate, scale_factor=None, mode='bilinear', align_corners=True)

        self.only_first = cfg.has('only_first', False)

    @property
    def horz_line(self):
        """Print horizontal line"""
        return '|{:<}|'.format('*' * self.width)

    @property
    def metr_line(self):
        """Print metrics line"""
        return '| {:^30} |' + ' {:^8} |' * len(self.metrics)

    @property
    def outp_line(self):
        """Print output line"""
        return '{:<30}' + ' | {:^8.3f}' * len(self.metrics)

    @staticmethod
    def wrap(string):
        """Wrap line around vertical bars"""
        return '| {} |'.format(string)

    def check_name(self, key):
        """Check name for prefixes"""
        return key.startswith(self.name) or \
               key.startswith('fwd_' + self.name) or \
               key.startswith('bwd_' + self.name)

    def reduce_fn(self, *args, **kwargs):
        """Reduce function"""
        raise NotImplementedError('reduce_fn not implemented for {}'.format(self.__name__))

    def populate_metrics_dict(self, *args, **kwargs):
        """Populate metrics function"""
        raise NotImplementedError('create_dict_key not implemented for {}'.format(self.__name__))

    def print(self, *args, **kwargs):
        """Print function"""
        raise NotImplementedError('print not implemented for {}'.format(self.__name__))

    @staticmethod
    def interp(dst, src, fn):
        """Interpolate dst to be the size of src using the interpolation function fn"""
        if dst is None:
            return dst
        assert dst.dim() == src.dim()
        if dst.dim() == 4 and not same_shape(dst.shape, src.shape):
            dst = fn(dst, size=src)
        return dst

    def interp_bilinear(self, dst, src):
        """Bilinear interpolation"""
        return self.interp(dst, src, self.bilinear)

    def interp_nearest(self, dst, src):
        """Nearest interpolation"""
        return self.interp(dst, src, self.nearest)

    def reduce(self, output, dataloaders, prefixes, verbose=True):
        """Reduce function"""
        reduced_data = self.reduce_metrics(output, dataloaders)
        metrics_dict = self.create_metrics_dict(reduced_data, prefixes)
        if verbose:
            self.print(reduced_data, prefixes)
        return metrics_dict

    def create_metrics_dict(self, reduced_data, prefixes):
        """Create metrics dictionary"""
        metrics_dict = {}
        # For all datasets
        for n, metrics in enumerate(reduced_data):
            if metrics:  # If there are calculated metrics
                self.populate_metrics_dict(metrics, metrics_dict, prefixes[n])
        # Return metrics dictionary
        return metrics_dict

    def reduce_metrics(self, dataset_outputs, datasets, ontology=None, strict=True):
        """Reduce metrics"""
        # If there is only one dataset, wrap in a list
        if isinstance(dataset_outputs[0], dict):
            dataset_outputs = [dataset_outputs]
        # List storing metrics for all datasets
        all_metrics_dict = []
        # Loop over all datasets and all batches
        for batch_outputs, dataset in zip(dataset_outputs, datasets):
            # Initialize metrics dictionary
            metrics_dict = OrderedDict()
            # Get length, names and dimensions
            length = len(dataset)
            names = [key for key in list(batch_outputs[0].keys()) if self.check_name(key)]
            dims = [tuple(batch_outputs[0][name].size()) for name in names]
            # Get data device
            device = batch_outputs[0]['idx'].device
            # Count how many times each sample was seen
            if strict:
                seen = torch.zeros(length, device=device)
                for output in batch_outputs:
                    seen[output['idx']] += 1
                seen = reduce_value(seen, average=False, name='idx')
                assert not np.any(seen.cpu().numpy() == 0), \
                    'Not all samples were seen during evaluation'
            # Reduce relevant metrics
            for name, dim in zip(names, dims):
                metrics = torch.zeros([length] + list(dim), device=device)

                # Count how many times each sample was seen
                if not strict:
                    seen = torch.zeros(length, device=device)
                    for output in batch_outputs:
                        if name in output:
                            seen[output['idx']] += 1
                    seen = reduce_value(seen, average=False, name='idx')

                for output in batch_outputs:
                    if name in output:
                        metrics[output['idx']] = output[name]
                metrics = reduce_value(metrics, average=False, name=name)
                metrics_dict[name] = self.reduce_fn(metrics, seen)
            # Append metrics dictionary to the list
            all_metrics_dict.append(metrics_dict)
        # Return list of metrics dictionary
        return all_metrics_dict

