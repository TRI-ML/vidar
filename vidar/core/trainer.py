# Copyright 2023 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

from knk_vision.vidar.vidar.core.checkpoint import ModelCheckpoint
from knk_vision.vidar.vidar.core.logger import WandbLogger
from knk_vision.vidar.vidar.core.saver import Saver
from knk_vision.vidar.vidar.utils.config import cfg_has, dataset_prefix
from knk_vision.vidar.vidar.utils.data import make_list, prepare_batch, get_from_dict
from knk_vision.vidar.vidar.utils.distributed import on_rank_0, rank, world_size, print0, dist_mode
from knk_vision.vidar.vidar.utils.logging import pcolor, AvgMeter
from knk_vision.vidar.vidar.utils.setup import setup_dataloader, reduce
from knk_vision.vidar.vidar.utils.types import is_dict, is_seq, is_numpy, is_tensor, is_list, is_namespace


def sample_to_cuda(sample, proc_rank, dtype=None):
    """Sends sample to cuda device"""
    # Do nothing if cuda is not available
    if not torch.cuda.is_available():
        return sample
    # If it's a sequence (list or tuple)
    if is_seq(sample):
        return [sample_to_cuda(val, proc_rank, dtype) for val in sample]
    # If it's a dictionary
    elif is_dict(sample):
        return {key: sample_to_cuda(sample[key], proc_rank, dtype) for key in sample.keys()}
    # If it's a torch tensor
    elif is_tensor(sample):
        dtype = dtype if torch.is_floating_point(sample) else None
        return sample.to(f'cuda:{proc_rank}', dtype=dtype)
    # If it's a numpy array
    elif is_numpy(sample):
        tensor_data = torch.Tensor(sample)
        dtype = dtype if torch.is_floating_point(tensor_data) else None
        return tensor_data.to(f'cuda:{proc_rank}', dtype=dtype)
    # Otherwise, do nothing
    else:
        return sample


class Trainer:
    def __init__(self, cfg, ckpt=None):
        """Trainer manager class.

        Parameters
        ----------
        cfg : Config
            Configuration file with trainer parameters
        ckpt : str, optional
            Checkpoint model used to initialize parameters, by default None
        """
        super().__init__()

        # self.avg_losses = {'loss': AvgMeter(50)}
        self.avg_losses = {}
        self.min_epochs = cfg.wrapper.has('min_epochs', 0)
        self.max_epochs = cfg.wrapper.has('max_epochs', 100)
        self.debug = cfg.wrapper.has('debug', False)

        self.validate_first = cfg.wrapper.has('validate_first', False)
        self.find_unused_parameters = cfg.wrapper.has('find_unused_parameters', False)
        self.clip_grad_norm = cfg.wrapper.has('clip_grad_norm', None)
        self.clip_grad_value = cfg.wrapper.has('clip_grad_value', None)
        self.save_first = cfg.wrapper.has('save_first', None)

        self.grad_scaler = cfg.wrapper.has('grad_scaler', None)
        self.min_grad_scale = None if self.grad_scaler is None else cfg.wrapper.has('min_scale', 128.0)

        self.saver = self.logger = self.checkpoint = None
        if not self.debug:
            self.prep_logger_and_checkpoint(cfg)

        self.prep_saver(cfg, ckpt)

        self.all_modes = ['train', 'mixed', 'validation', 'test']
        self.train_modes = ['train', 'mixed']

        self.current_epoch = 0
        self.samples_processed, self.samples_accumulated = 0, 0
        self.progress_metrics = cfg_has(cfg.wrapper, 'progress_metrics', [])

        if cfg.wrapper.has('use_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = cfg.wrapper.use_tf32 is True or 'matmul' in cfg.wrapper.use_tf32
            torch.backends.cudnn.allow_tf32 = cfg.wrapper.use_tf32 is True or 'cudnn' in cfg.wrapper.use_tf32

    @property
    def progress(self):
        """Returns current progress percentage"""
        return self.current_epoch / self.max_epochs

    @property
    def proc_rank(self):
        """Returns process rank"""
        return rank()

    @property
    def world_size(self):
        """Returns world size"""
        return world_size()

    @property
    def is_rank_0(self):
        """Returns True if process rank is 0"""
        return self.proc_rank == 0

    @staticmethod
    def param_logs(optimizers):
        """Returns various logs for tracking"""
        params = OrderedDict()
        for key, val in optimizers.items():
            params[f'{key}_learning_rate'] = val['optimizer'].param_groups[0]['lr']
            params[f'{key}_weight_decay'] = val['optimizer'].param_groups[0]['weight_decay']
        return {
            **params,
        }

    @on_rank_0
    def prep_logger_and_checkpoint(self, cfg):
        """Initialize logger and checkpoint objects"""

        add_logger = cfg_has(cfg, 'wandb')
        add_checkpoint = cfg_has(cfg, 'checkpoint')

        if add_logger:
            self.logger = WandbLogger(cfg.wandb, verbose=True)
            if add_checkpoint and not cfg_has(cfg.checkpoint, 'name'):
                cfg.checkpoint.name = self.logger.run_name
        else:
            self.logger = None

        if add_checkpoint:
            self.checkpoint = ModelCheckpoint(cfg.checkpoint, verbose=True)
        else:
            self.checkpoint = None

        if add_logger:
            self.logger.log_config(cfg)

    def prep_saver(self, cfg, ckpt=None):
        """Initialize saver object"""
        if not cfg.has('arch'):
            return

        ckpt = ckpt if ckpt is not None else cfg.arch.model.has('checkpoint', None)
        add_saver = cfg_has(cfg, 'save')

        if add_saver:
            print0(pcolor('#' * 60, color='red', attrs=('dark',)))
            print0(pcolor('### Saving data to: %s' % cfg.save.folder, color='red'))
            print0(pcolor('#' * 60, color='red', attrs=('dark',)))
            self.saver = Saver(cfg.save, ckpt)

    @on_rank_0
    def check_and_save(self, wrapper, output, prefixes, epoch_perc):
        """Checks and saves model and optimizer states"""
        if self.checkpoint is not None:
            self.checkpoint.check_and_save(
                wrapper, output, prefixes,
                epoch=self.current_epoch + epoch_perc,
                samples=self.samples_processed
            )

    @on_rank_0
    def log_losses_and_metrics(self, metrics=None, optimizers=None, epoch_perc=0.0):
        """Logs losses and metrics to logger"""
        if self.logger is not None:
            self.logger.log_metrics({
                '{}'.format(key): val.get() for key, val in self.avg_losses.items()
            })
            if optimizers is not None:
                self.logger.log_metrics(self.param_logs(optimizers))
            if metrics is not None:
                self.logger.log_metrics({
                    **metrics,
                    'epochs': self.current_epoch + epoch_perc,
                    'samples': self.samples_processed,
                })

    @on_rank_0
    def print_logger_and_checkpoint(self):
        """Prints logger and checkpoint urls on console"""

        font_base = {'color': 'red', 'attrs': ('bold', 'dark')}
        font_name = {'color': 'red', 'attrs': ('bold',)}
        font_underline = {'color': 'red', 'attrs': ('underline',)}

        if self.logger or self.checkpoint:
            print(pcolor('#' * 120, **font_base))
        if self.logger:
            print(pcolor('### WandB: ', **font_base) + \
                  pcolor('{}'.format(self.logger.run_name), **font_name) + \
                  pcolor(' - ', **font_base) + \
                  pcolor('{}'.format(self.logger.run_url), **font_underline))
        if self.checkpoint and self.checkpoint.s3_url is not None:
            print(pcolor('### Checkpoint: ', **font_base) + \
                  pcolor('{}'.format(self.checkpoint.s3_url), **font_underline))
        if self.logger or self.checkpoint:
            print(pcolor('#' * 120 + '\n', **font_base))

    @on_rank_0
    def print_skip(self):
        """Dummy class to skip printing when not requested"""
        print()

    @on_rank_0
    def update_train_progress_bar(self, progress_bar, scaler=None):
        """Update grain progress bar with current losses and metrics"""
        string = ''
        if 'loss' in self.avg_losses.keys():
            string += '| {} | Loss {:.3f}'.format(
                int(self.current_epoch), self.avg_losses['loss'].get())
        if scaler is not None and scaler.get_scale() != 1.0:
            string += ' | {}'.format(scaler.get_scale())
        for pair in self.progress_metrics:
            name, abbrv = pair if is_list(pair) else (pair, ''.join(p[0] for p in pair.split('_')))
            if name in self.avg_losses.keys():
                string += ' | {} {:.3f}'.format(abbrv, self.avg_losses[name].get())
        progress_bar.set_description(string + ' |')

    @on_rank_0
    def update_averages(self, output):
        """Update running averages with output information"""
        averages = {}
        if get_from_dict(output, 'loss') is not None:
            averages['loss'] = output['loss']
        if get_from_dict(output, 'metrics') is not None:
            averages.update(**output['metrics'])
        for key in averages.keys():
            if key not in self.avg_losses.keys():
                self.avg_losses[key] = AvgMeter(50)
            self.avg_losses[key](averages[key].item() if is_tensor(averages[key]) else averages[key])

    def train_progress_bar(self, dataloader, ncols=None, aux_dataloader=None):
        """Create progress bar for training loop"""
        full_dataloader = dataloader if aux_dataloader is None else zip(dataloader, aux_dataloader)
        return tqdm(enumerate(full_dataloader, 0),
                    unit='im', unit_scale=self.world_size * dataloader.batch_size,
                    total=len(dataloader), smoothing=0,
                    disable=not self.is_rank_0, ncols=ncols)

    def val_progress_bar(self, dataloader, prefix, ncols=None):
        """Create progress bar for validation loop"""
        return tqdm(enumerate(dataloader, 0),
                    unit='im', unit_scale=self.world_size * dataloader.batch_size,
                    total=len(dataloader), smoothing=0,
                    disable=not self.is_rank_0, ncols=ncols,
                    desc=prefix)

    def prepare_distributed_model(self, wrapper):
        """Prepare distributed model based on distributed mode"""
        if dist_mode() == 'cpu':
            wrapper.arch = wrapper.arch
        elif dist_mode() == 'gpu':
            wrapper = wrapper.cuda(self.proc_rank)
            wrapper.arch = wrapper.arch
        elif dist_mode() == 'ddp':
            wrapper = wrapper.cuda(self.proc_rank)
            wrapper.arch = torch.nn.parallel.DistributedDataParallel(
                wrapper.arch, device_ids=[self.proc_rank],
                find_unused_parameters=self.find_unused_parameters,
                broadcast_buffers=True)
        else:
            raise ValueError('Wrong distributed mode {}'.format(dist_mode))
        return wrapper

    def prepare_dataloaders(self, wrapper):
        """Prepare dataloaders for training and validation"""

        font1 = {'color': 'blue', 'attrs': ('dark', 'bold')}
        font2 = {'color': 'blue', 'attrs': ('bold',)}

        print0(pcolor('#' * 60, **font1))

        if dist_mode() == 'cpu':
            print0(pcolor(f'### ', **font1) +
                   pcolor(f'CPU Training', **font2))
        elif dist_mode() == 'gpu':
            print0(pcolor(f'### ', **font1) +
                   pcolor(f'GPU Training', **font2))
        elif dist_mode() == 'ddp':
            print0(pcolor(f'### ', **font1) +
                   pcolor(f'DDP Training ', **font2) +
                   pcolor(f'with ', **font1) +
                   pcolor(f'{self.world_size}', **font2) +
                   pcolor(f' GPUs', **font1))

        # Send wrapper to GPU
        if wrapper.arch is not None:
            wrapper = self.prepare_distributed_model(wrapper)

        for key in wrapper.datasets_cfg.keys():
            wrapper.datasets_cfg[key] = make_list(wrapper.datasets_cfg[key])

        # Prepare dataloaders
        dataloaders = {
            key: setup_dataloader(make_list(val), wrapper.datasets_cfg[key], key)
            for key, val in wrapper.datasets.items() if key in wrapper.datasets_cfg.keys()
        }

        # Prepare prefixes
        prefixes = {
            key: [dataset_prefix(wrapper.datasets_cfg[key][n]) for n in range(len(val))]
            for key, val in wrapper.datasets_cfg.items() if 'name' in wrapper.datasets_cfg[key][0].__dict__.keys()
        }

        # Reduce information
        reduced_dataloaders = reduce(dataloaders, self.all_modes, self.train_modes)
        reduced_prefixes = reduce(prefixes, self.all_modes, self.train_modes)

        print0(pcolor('#' * 60, **font1))

        return reduced_dataloaders, reduced_prefixes

    def filter_optimizers(self, optimizers):
        """Filter optimizers based on current information"""
        in_optimizers, out_optimizers = {}, {}
        if optimizers is not None:
            for key, val in optimizers.items():
                if 'stop_epoch' not in val['settings'] or \
                        val['settings']['stop_epoch'] >= self.current_epoch:
                    in_optimizers[key] = val['optimizer']
                else:
                    out_optimizers[key] = val['optimizer']

        if rank() == 0:

            string = pcolor('Optimizing: ', color='yellow')
            for key, val in in_optimizers.items():
                string += pcolor('{}'.format(key), color='green', attrs=('bold', 'dark'))
                string += pcolor(' ({}) '.format(val.param_groups[0]['lr']),
                                 color='green', attrs=('dark',))
            for key, val in out_optimizers.items():
                string += pcolor('{}'.format(key), color='cyan', attrs=('bold', 'dark'))
                string += pcolor(' ({}) '.format(val.param_groups[0]['lr']),
                                 color='cyan', attrs=('dark',))

            print(pcolor('#' * 120, color='yellow', attrs=('dark',)))
            print(string)
            print(pcolor('#' * 120, color='yellow', attrs=('dark',)))
            print()

        return in_optimizers, out_optimizers

    def learn(self, wrapper):
        """Learning loop for training and validation"""

        # Get optimizers and schedulers
        optimizers, schedulers = wrapper.configure_optimizers_and_schedulers()

        # Get gradient scaler if requested
        scaler = torch.cuda.amp.GradScaler(
            enabled=self.grad_scaler.has('enabled', True),
            init_scale=2.0 ** self.grad_scaler.has('init_scale', 10),
            growth_factor=self.grad_scaler.has('growth_factor', 2.0),
            backoff_factor=self.grad_scaler.has('backoff_factor', 0.5),
            growth_interval=2 ** self.grad_scaler.has('growth_interval', 6),
        ) if is_namespace(self.grad_scaler) else torch.cuda.amp.GradScaler(enabled=self.grad_scaler is True)

        # Get learn information
        dataloaders, prefixes = self.prepare_dataloaders(wrapper)

        # Check for train and validation dataloaders
        has_train_dataloader = 'train' in dataloaders
        has_validation_dataloader = 'validation' in dataloaders

        # Save model first if requested
        if self.save_first is not None:
            wrapper.save(self.save_first)

        # Validate before training if requested
        self.current_epoch = 0
        if self.validate_first and has_validation_dataloader:
            validation_output = self.validate('validation', dataloaders, prefixes, wrapper)
            self.post_validation(validation_output, optimizers, prefixes['validation'], wrapper)

        # Epoch loop
        self.current_epoch = 1
        if has_train_dataloader:
            for epoch in range(self.current_epoch, self.max_epochs + 1):

                # Train and log
                self.train(dataloaders, optimizers, schedulers, prefixes, wrapper, scaler=scaler)

                # Validate, save and log
                if has_validation_dataloader and wrapper.epochs_per_validation is not None:
                    if wrapper.epochs_per_validation <= 1 or epoch % wrapper.epochs_per_validation == 0:
                        validation_output = self.validate('validation', dataloaders, prefixes, wrapper)
                        self.post_validation(validation_output, optimizers, prefixes['validation'], wrapper)
                else:
                    self.print_skip()
                self.current_epoch += 1

                # Take a scheduler step
                if wrapper.update_schedulers == 'epoch':
                    for scheduler in schedulers.values():
                        scheduler.step()

        # Finish logger if available
        if self.logger:
            self.logger.finish()

    def train(self, dataloaders, optimizers, schedulers, prefixes, wrapper, scaler=None):
        """Training epoch loop"""

        train_dataloader = get_from_dict(dataloaders, 'train')
        mixed_dataloader = get_from_dict(dataloaders, 'mixed')
        validation_dataloader = get_from_dict(dataloaders, 'validation')

        # Choose which optimizers to use
        in_optimizers, out_optimizers = self.filter_optimizers(optimizers)
        
        # Set wrapper to train
        wrapper.train_custom(in_optimizers, out_optimizers)

        # Shuffle dataloader sampler
        if train_dataloader is not None:
            if hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(self.current_epoch)

        # Shuffle auxiliary dataloader sampler
        if mixed_dataloader is not None:
            if hasattr(mixed_dataloader.sampler, "set_epoch"):
                mixed_dataloader.sampler.set_epoch(self.current_epoch)

        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            train_dataloader, aux_dataloader=mixed_dataloader, ncols=120)

        # Zero gradients for the first iteration
        for optimizer in in_optimizers.values():
            optimizer.zero_grad()

        step_size = world_size() * train_dataloader.batch_size
        samples_per_validation = None if wrapper.samples_per_validation is None else \
            int(np.ceil(wrapper.samples_per_validation / step_size) * step_size)
        epochs_per_validation = wrapper.epochs_per_validation
        if epochs_per_validation is not None:
            epochs_per_validation = None if wrapper.epochs_per_validation >= 1 else int(1. / wrapper.epochs_per_validation)
        if epochs_per_validation is not None:
            validation_splits = np.linspace(0, len(progress_bar), epochs_per_validation + 1).astype(int).tolist()[1:-1]
        else:
            validation_splits = []

        # Loop through all batches
        for i, batch in progress_bar:

            # Send samples to GPU and take a training step
            batch = sample_to_cuda(batch, self.proc_rank)
            if wrapper.broken:
                batch = prepare_batch(batch, break_flow=True)
            output = wrapper.training_step(batch, epoch=self.current_epoch)

            # Backprop through loss
            if get_from_dict(output, 'loss') is not None and is_tensor(output['loss']):
                scaler.scale(output['loss']).backward()
                for optimizer in in_optimizers.values():
                    scaler.unscale_(optimizer)
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(wrapper.arch.parameters(), self.clip_grad_norm)
                    if self.clip_grad_value is not None:
                        torch.nn.utils.clip_grad_value_(wrapper.arch.parameters(), self.clip_grad_value)
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                scaler.update()

                if self.min_grad_scale is not None and scaler._scale < self.min_grad_scale:
                    scaler._scale = torch.tensor(self.min_grad_scale).to(scaler._scale)

            self.update_averages(output)
            self.update_train_progress_bar(progress_bar, scaler=scaler)

            # Step optimizer
            if wrapper.update_schedulers == 'step':
                for scheduler in schedulers.values():
                    scheduler.step()

            # Validate before training if requested
            self.samples_processed += step_size
            self.samples_accumulated += step_size
            if validation_dataloader is not None:
                if samples_per_validation is not None and self.samples_accumulated >= samples_per_validation:
                    self.samples_accumulated -= samples_per_validation
                    epoch_perc = int(float(100 * (i + 1)) / float(len(progress_bar))) / 100
                    validation_output = self.validate('validation', dataloaders, prefixes, wrapper)
                    self.post_validation(validation_output, optimizers, prefixes['validation'], wrapper, epoch_perc=epoch_perc)
                    wrapper.train_custom(in_optimizers, out_optimizers)
                if len(validation_splits) > 0 and i >= validation_splits[0]:
                    validation_splits = validation_splits[1:]
                    epoch_perc = int(float(100 * i) / float(len(progress_bar))) / 100
                    validation_output = self.validate('validation', dataloaders, prefixes, wrapper)
                    self.post_validation(validation_output, optimizers, prefixes['validation'], wrapper, epoch_perc=epoch_perc),
                    wrapper.train_custom(in_optimizers, out_optimizers)

        # Return outputs for epoch end
        return wrapper.training_epoch_end()

    @torch.no_grad()
    def validate(self, mode, dataloaders, prefixes, wrapper):
        """Validation loop"""

        # Set wrapper to eval
        wrapper.eval_custom()
        # For all validation datasets
        dataset_outputs = []
        for dataset_idx, (dataset, dataloader, prefix) in \
                enumerate(zip(wrapper.datasets[mode], dataloaders[mode], prefixes[mode])):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(dataloader, prefix, ncols=120)
            # For all batches
            batch_outputs = []
            for batch_idx, batch in progress_bar:
                # Send batch to GPU and take a validation step
                batch = sample_to_cuda(batch, self.proc_rank)
                if wrapper.broken:
                    batch = prepare_batch(batch, break_flow=True)
                output, results = wrapper.validation_step(
                    batch, dataset_idx=dataset_idx, epoch=self.current_epoch)
                if 'batch' in output:
                    batch = output['batch']
                batch_outputs += results
                if self.logger:
                    self.logger.log_data('val', batch, output, dataset, prefix)
                if self.saver:
                    self.saver.save_data(batch, output, prefix)
            # Append dataset outputs to list of all outputs
            dataset_outputs.append(batch_outputs)
        # Get results from validation epoch end
        return wrapper.validation_epoch_end(dataset_outputs, prefixes[mode])

    def post_validation(self, output, optimizers, prefixes, wrapper, epoch_perc=0.0):
        """Post-process validation results"""
        if epoch_perc > 0.0:
            epoch_perc = epoch_perc - 1
        self.check_and_save(wrapper, output, prefixes, epoch_perc)
        self.log_losses_and_metrics(output, optimizers, epoch_perc)
        self.print_logger_and_checkpoint()

    def test(self, wrapper):
        """Test loop"""
        dataloaders, prefixes = self.prepare_dataloaders(wrapper)
        self.validate('validation', dataloaders, prefixes, wrapper)

