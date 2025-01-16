# Copyright 2023 Toyota Research Institute.  All rights reserved.

import os
import random
from abc import ABC
from collections import OrderedDict

import torch

from knk_vision.vidar.vidar.utils.config import cfg_has, read_config
from knk_vision.vidar.vidar.utils.data import set_random_seed, get_from_dict, make_list
from knk_vision.vidar.vidar.utils.distributed import rank, world_size
from knk_vision.vidar.vidar.utils.flip import flip_batch, flip_output
from knk_vision.vidar.vidar.utils.logging import pcolor, set_debug
from knk_vision.vidar.vidar.utils.networks import load_checkpoint, save_checkpoint, freeze_layers_and_norms
from knk_vision.vidar.vidar.utils.setup import setup_arch, setup_datasets, setup_metrics
from knk_vision.vidar.vidar.utils.types import is_str
from knk_vision.vidar.vidar.utils.optimizers import get_step_schedule_with_warmup, get_linear_schedule_with_warmup
from knk_vision.vidar.vidar.utils.augmentations import crop_stack_batch, merge_stack_predictions


class Wrapper(torch.nn.Module, ABC):

    def __init__(self, cfg, ckpt=None, verbose=False):
        """Wrapper manager class.

        Parameters
        ----------
        cfg : Config
            Configuration file with wrapper parameters
        ckpt : str, optional
            Path to checkpoint model, by default None
        verbose : bool, optional
            True if information is displayed on screen, by default False
        """
        super().__init__()

        if verbose and rank() == 0:
            font = {'color': 'cyan', 'attrs': ('bold', 'dark')}
            print(pcolor('#' * 100, **font))
            print(pcolor('#' * 42 + ' VIDAR WRAPPER ' + '#' * 43, **font))
            print(pcolor('#' * 100, **font))

        # Get configuration
        cfg = read_config(cfg) if is_str(cfg) else cfg
        self.set_environment(cfg)
        self.cfg = cfg

        # Data augmentations
        self.flip_lr_prob = cfg_has(cfg.wrapper, 'flip_lr_prob', 0.0)
        self.validate_flipped = cfg_has(cfg.wrapper, 'validate_flipped', False)

        # Set random seed
        set_random_seed(cfg.wrapper.seed + rank())
        set_debug(cfg_has(cfg.wrapper, 'debug', False))

        # Setup architecture, datasets and tasks
        self.arch = setup_arch(cfg.arch, checkpoint=ckpt, verbose=verbose) if cfg_has(cfg, 'arch') else None
        self.datasets, self.datasets_cfg = setup_datasets(
            cfg.datasets, verbose=verbose) if cfg_has(cfg, 'datasets') else (None, None)
        self.metrics = setup_metrics(cfg.evaluation) if cfg_has(cfg, 'evaluation') else {}

        sync_batch_norm = cfg_has(cfg.wrapper, 'sync_batch_norm', False)
        if self.arch is not None and sync_batch_norm and os.environ['DIST_MODE'] == 'ddp':
            self.arch = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.arch)

        self.mixed_precision = cfg_has(cfg.wrapper, 'mixed_precision', False)
        self.crop_stack = cfg.wrapper.has('crop_stack', None)
        self.broken = cfg.wrapper.has('broken', False)

        self.update_schedulers = None

        self.samples_per_validation = cfg.wrapper.has('samples_per_validation', None)
        self.epochs_per_validation = cfg.wrapper.has('epochs_per_validation', 1)
        if self.samples_per_validation is not None:
            self.epochs_per_validation = None
        if self.epochs_per_validation is not None:
            self.samples_per_validation = None

    @staticmethod
    def set_environment(cfg):
        """Set environment variables for wrapper"""
        if cfg.wrapper.has('align_corners'):
            os.environ['ALIGN_CORNERS'] = str(cfg.wrapper.align_corners)
        elif os.getenv('ALIGN_CORNERS') is not None:
            del os.environ['ALIGN_CORNERS']

    def save(self, filename, epoch=None):
        """Save wrapper to checkpoint"""
        save_checkpoint(filename, self, epoch=epoch)

    def load(self, checkpoint, mode='copy', strict=True, verbose=False):
        """Load wrapper from checkpoint"""
        load_checkpoint(self, checkpoint, mode=mode, strict=strict, verbose=verbose)

    def train_custom(self, in_optimizers, out_optimizers):
        """Custom training function for wrapper"""
        if self.arch is None:
            return
        self.arch.train()
        for key in in_optimizers.keys():
            arch = self.arch.module if hasattr(self.arch, 'module') else self.arch
            freeze_layers_and_norms(arch.networks[key], ['ALL'], flag_freeze=False)
        for key in out_optimizers.keys():
            arch = self.arch.module if hasattr(self.arch, 'module') else self.arch
            freeze_layers_and_norms(arch.networks[key], ['ALL'], flag_freeze=True)

    def eval_custom(self):
        """Custom evaluation function for wrapper"""
        if self.arch is None:
            return
        self.arch.eval()

    def configure_optimizers_and_schedulers(self):
        """Configure depth and pose optimizers and the corresponding scheduler"""

        if not cfg_has(self.cfg, 'optimizers'):
            return None, None
        if 'train' not in self.datasets_cfg.keys():
            return None, None

        optimizers = OrderedDict()
        schedulers = OrderedDict()

        for key, val in self.cfg.optimizers.__dict__.items():
            if key not in self.arch.networks:
                continue
            args = {
                    'lr': val.lr,
                    'weight_decay': val.has('weight_decay', 0.0),
                    'params': self.arch.networks[key].parameters(),
                }
            if 'Adam' in val.name:
                args.update({'eps': val.has('eps', 1.0e-8)})
            optimizers[key] = {
                'optimizer': getattr(torch.optim, val.name)(**args),
                'settings': {} if not cfg_has(val, 'settings') else val.settings.__dict__
            }
            if cfg_has(val, 'scheduler'):
                if val.scheduler.name == 'CosineAnnealingWarmUpRestarts':
                    from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
                    epoch = float(len(self.datasets['train']) / (
                            world_size() * self.datasets_cfg['train'].dataloader.batch_size * self.datasets_cfg['train'].repeat[0]))
                    schedulers[key] = CosineAnnealingWarmupRestarts(**{
                        'optimizer': optimizers[key]['optimizer'],
                        'first_cycle_steps': int(val.scheduler.first_cycle_steps * epoch),
                        'cycle_mult': val.scheduler.cycle_mult,
                        'min_lr': val.scheduler.min_lr,
                        'max_lr': val.scheduler.max_lr,
                        'warmup_steps': int(val.scheduler.warmup_steps * epoch),
                        'gamma': val.scheduler.gamma,
                    })
                    self.update_schedulers = 'step'
                elif val.scheduler.name == 'LinearWarmUp':
                    schedulers[key] = get_linear_schedule_with_warmup(**{
                        'optimizer': optimizers[key]['optimizer'],
                        'num_warmup_steps': val.scheduler.num_warmup_steps,
                        'num_training_steps': val.scheduler.num_training_steps,
                    })
                    self.update_schedulers = 'step'
                elif val.scheduler.name == 'StepWarmUp':
                    epoch_size = float(len(self.datasets['train']) / (
                            world_size() * self.datasets_cfg['train'].dataloader.batch_size))
                    schedulers[key] = get_step_schedule_with_warmup(**{
                        'optimizer': optimizers[key]['optimizer'],
                        'warmup_epochs': int(val.scheduler.warmup_epochs * epoch_size),
                        'lr_start': val.scheduler.lr_start,
                        'epoch_size': epoch_size,
                        'step_size': val.scheduler.step_size,
                        'gamma': val.scheduler.gamma,
                    })
                    self.update_schedulers = 'step'
                else:
                    schedulers[key] = getattr(torch.optim.lr_scheduler, val.scheduler.name)(**{
                        'optimizer': optimizers[key]['optimizer'],
                        'step_size': val.scheduler.step_size,
                        'gamma': val.scheduler.gamma,
                    })
                    self.update_schedulers = 'epoch'

        # Return optimizer and scheduler
        return optimizers, schedulers

    def run_arch(self, batch, epoch, flip, unflip):
        """Run wrapper for a given batch"""
        if self.arch is None:
            return {}

        batch = flip_batch(batch) if flip else batch
        output = self.arch(batch, epoch=epoch)
        return flip_output(output) if flip and unflip else output

    def training_step(self, batch, epoch):
        """Processes a training batch"""
        flip_lr = False if self.flip_lr_prob == 0 else \
            random.random() < self.flip_lr_prob

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            output = self.run_arch(batch, epoch=epoch, flip=flip_lr, unflip=False)

        losses = {key: val for key, val in output.items() if key.startswith('loss')}

        return {
            **losses,
            'metrics': get_from_dict(output, 'metrics'),
            'predictions': get_from_dict(output, 'predictions'),
        }

    def validation_step(self, batch, dataset_idx, epoch):
        """Processes a validation batch"""

        if self.crop_stack is not None:
            batch = crop_stack_batch(batch, window=self.crop_stack)

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            output = self.run_arch(batch, epoch=epoch, flip=False, unflip=False)
            flipped_output = None if not self.validate_flipped else \
                self.run_arch(batch, epoch=epoch, flip=True, unflip=True)

        if 'batch' in output:
            batch = output['batch']

        if self.crop_stack is not None:
            for key in output['predictions']:
                for task in ['depth']:
                    if key.startswith(task):
                        output['predictions'] = merge_stack_predictions(
                            batch, output['predictions'], task=key)
                        if flipped_output is not None:
                            flipped_output['predictions'] = merge_stack_predictions(
                                batch, flipped_output['predictions'], task=key)

        results = self.evaluate(batch, output, flipped_output, dataset_idx=dataset_idx)
        results = [{
            'idx': batch['idx'][i],
            **{key: val[i] for key, val in results['metrics'].items() if val[i].mean() != -1}
        } for i in range(len(batch['idx']))]

        return output, results

    @staticmethod
    def training_epoch_end():
        """Finishes a training epoch (do nothing for now)"""
        return {}

    def validation_epoch_end(self, output, prefixes):
        """Finishes a validation epoch"""
        if isinstance(output[0], dict):
            output = [output]

        metrics_dict = {}
        datasets = self.datasets['validation']
        for task in self.metrics:
            metrics_dict.update(
                self.metrics[task].reduce(output, datasets, prefixes, task))

        return metrics_dict

    def evaluate(self, batch, output, flipped_output=None, dataset_idx=None):
        """Evaluate batch to produce predictions and metrics for different tasks"""
        # Evaluate different tasks
        metrics, predictions = OrderedDict(), OrderedDict()
        for task in self.metrics:
            # Check if evaluation is to happen
            dataset_cfg = self.datasets_cfg['validation'][dataset_idx]
            evaluate = not dataset_cfg.has('evaluation') or not dataset_cfg.evaluation.has('metrics') or \
                       (dataset_cfg.has('evaluation') and dataset_cfg.evaluation.has('metrics') and
                        task in make_list(dataset_cfg.evaluation.metrics))
            # Evaluate if requested
            if evaluate:
                task_metrics, task_predictions = self.metrics[task].evaluate(
                    batch, output['predictions'], task,
                    flipped_output['predictions'] if flipped_output else None
                )
                metrics.update(task_metrics)
                predictions.update(task_predictions)
        # Crate results dictionary with metrics and predictions
        results = {'metrics': metrics, 'predictions': predictions}
        # Return final results
        return results



