# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict

import torch
from tqdm import tqdm

from vidar.core.checkpoint import ModelCheckpoint
from vidar.core.logger import WandbLogger
from vidar.core.saver import Saver
from vidar.utils.config import cfg_has, dataset_prefix
from vidar.utils.data import make_list, keys_in
from vidar.utils.distributed import on_rank_0, rank, world_size, print0, dist_mode
from vidar.utils.logging import pcolor, AvgMeter
from vidar.utils.setup import setup_dataloader, reduce
from vidar.utils.types import is_dict, is_seq, is_numpy, is_tensor, is_list


def sample_to_cuda(sample, proc_rank, dtype=None):
    """
    Copy sample to GPU

    Parameters
    ----------
    sample : Dict
        Dictionary with sample information
    proc_rank : Int
        Process rank
    dtype : torch.Type
        Data type for conversion

    Returns
    -------
    sample : Dict
        Dictionary with sample on the GPU
    """
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
    """
    Trainer class for model optimization and inference

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    ckpt : String
        Name of the model checkpoint to start from
    """
    def __init__(self, cfg, ckpt=None):
        super().__init__()

        self.avg_losses = {}
        self.min_epochs = cfg_has(cfg.wrapper, 'min_epochs', 0)
        self.max_epochs = cfg_has(cfg.wrapper, 'max_epochs', 100)

        self.validate_first = cfg_has(cfg.wrapper, 'validate_first', False)
        self.find_unused_parameters = cfg_has(cfg.wrapper, 'find_unused_parameters', False)
        self.grad_scaler = cfg_has(cfg.wrapper, 'grad_scaler', False) and torch.cuda.is_available()

        self.saver = self.logger = self.checkpoint = None
        self.prep_logger_and_checkpoint(cfg)

        self.prep_saver(cfg, ckpt)

        self.all_modes = ['train', 'mixed', 'validation', 'test']
        self.train_modes = ['train', 'mixed']

        self.current_epoch = 0
        self.training_bar_metrics = cfg_has(cfg.wrapper, 'training_bar_metrics', [])

    @property
    def progress(self):
        """Current epoch progress (percentage)"""
        return self.current_epoch / self.max_epochs

    @property
    def proc_rank(self):
        """Process rank"""
        return rank()

    @property
    def world_size(self):
        """World size"""
        return world_size()

    @property
    def is_rank_0(self):
        """True if worker is on rank 0"""
        return self.proc_rank == 0

    def param_logs(self, optimizers):
        """Returns various logs for tracking"""
        params = OrderedDict()
        for key, val in optimizers.items():
            params[f'{key}_learning_rate'] = val['optimizer'].param_groups[0]['lr']
            params[f'{key}_weight_decay'] = val['optimizer'].param_groups[0]['weight_decay']
        params['progress'] = self.progress
        return {
            **params,
        }

    @on_rank_0
    def prep_logger_and_checkpoint(self, cfg):
        """Prepare logger and checkpoint class if requested"""
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
        """Prepare saver class if requested"""
        ckpt = ckpt if ckpt is not None else cfg.arch.model.has('checkpoint', None)
        add_saver = cfg_has(cfg, 'save')

        if add_saver:
            print0(pcolor('#' * 60, color='red', attrs=('dark',)))
            print0(pcolor('### Saving data to: %s' % cfg.save.folder, color='red'))
            print0(pcolor('#' * 60, color='red', attrs=('dark',)))
            self.saver = Saver(cfg.save, ckpt)

    @on_rank_0
    def check_and_save(self, wrapper, output, prefixes):
        """Check for conditions and save if it's time"""
        if self.checkpoint is not None:
            self.checkpoint.check_and_save(
                wrapper, output, prefixes, epoch=self.current_epoch)

    @on_rank_0
    def log_losses_and_metrics(self, metrics=None, optimizers=None):
        """Log losses and metrics on wandb"""
        if self.logger is not None:
            self.logger.log_metrics({
                '{}'.format(key): val.get() for key, val in self.avg_losses.items()
            })
            if optimizers is not None:
                self.logger.log_metrics(self.param_logs(optimizers))
            if metrics is not None:
                self.logger.log_metrics({
                    **metrics, 'epochs': self.current_epoch,
                })

    @on_rank_0
    def print_logger_and_checkpoint(self):
        """Print logger and checkpoint information"""
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
    def update_train_progress_bar(self, progress_bar):
        """Update training progress bar on screen"""
        string = '| {} | Loss {:.3f}'.format(
            self.current_epoch, self.avg_losses['loss'].get())
        bar_keys = self.training_bar_metrics
        for key in keys_in(self.avg_losses, bar_keys):
            name, abbrv = (key[0], key[1]) if is_list(key) else (key, key)
            string += ' | {} {:.2f}'.format(abbrv, self.avg_losses[name].get())
        progress_bar.set_description(string)

    @on_rank_0
    def update_averages(self, output):
        """Update loss averages"""
        averages = {'loss': output['loss'], **output['metrics']}
        for key in averages.keys():
            if key not in self.avg_losses.keys():
                self.avg_losses[key] = AvgMeter(50)
            self.avg_losses[key](averages[key].item() if is_tensor(averages[key]) else averages[key])

    def train_progress_bar(self, dataloader, ncols=None, aux_dataloader=None):
        """Print training progress bar on screen"""
        full_dataloader = dataloader if aux_dataloader is None else zip(dataloader, aux_dataloader)
        return tqdm(enumerate(full_dataloader, 0),
                    unit='im', unit_scale=self.world_size * dataloader.batch_size,
                    total=len(dataloader), smoothing=0,
                    disable=not self.is_rank_0, ncols=ncols)

    def val_progress_bar(self, dataloader, prefix, ncols=None):
        """Print validation progress bar on screen"""
        return tqdm(enumerate(dataloader, 0),
                    unit='im', unit_scale=self.world_size * dataloader.batch_size,
                    total=len(dataloader), smoothing=0,
                    disable=not self.is_rank_0, ncols=ncols,
                    desc=prefix)

    def prepare_distributed_model(self, wrapper):
        """Prepare model for distributed training or not (CPU/GPU/DDP)"""
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
        """Prepare dataloaders for training and inference"""
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
        wrapper = self.prepare_distributed_model(wrapper)

        for key in wrapper.datasets_cfg.keys():
            wrapper.datasets_cfg[key] = make_list(wrapper.datasets_cfg[key])

        # Prepare dataloaders
        dataloaders = {
            key: setup_dataloader(val, wrapper.datasets_cfg[key][0].dataloader, key)
            for key, val in wrapper.datasets.items() if key in wrapper.datasets_cfg.keys()
        }

        # Prepare prefixes

        prefixes = {
            key: [dataset_prefix(wrapper.datasets_cfg[key][n], n) for n in range(len(val))]
            for key, val in wrapper.datasets_cfg.items() if 'name' in wrapper.datasets_cfg[key][0].__dict__.keys()
        }

        # Reduce information
        reduced_dataloaders = reduce(dataloaders, self.all_modes, self.train_modes)
        reduced_prefixes = reduce(prefixes, self.all_modes, self.train_modes)

        print0(pcolor('#' * 60, **font1))

        return reduced_dataloaders, reduced_prefixes

    def filter_optimizers(self, optimizers):
        """Filter optimizers to find those being used at each epoch"""
        in_optimizers, out_optimizers = {}, {}
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
        """Entry-point class for training a model"""
        # Get optimizers and schedulers
        optimizers, schedulers = wrapper.configure_optimizers_and_schedulers()

        # Get gradient scaler if requested
        scaler = torch.cuda.amp.GradScaler() if self.grad_scaler else None

        # Get learn information
        dataloaders, prefixes = self.prepare_dataloaders(wrapper)
        aux_dataloader = None if 'mixed' not in dataloaders else dataloaders['mixed']

        # Check for train and validation dataloaders
        has_train_dataloader = 'train' in dataloaders
        has_validation_dataloader = 'validation' in dataloaders

        # Validate before training if requested
        if self.validate_first and has_validation_dataloader:
            validation_output = self.validate('validation', dataloaders, prefixes, wrapper)
            self.post_validation(validation_output, optimizers, prefixes['validation'], wrapper)
        else:
            self.current_epoch += 1

        # Epoch loop
        if has_train_dataloader:
            for epoch in range(self.current_epoch, self.max_epochs + 1):

                # Train and log
                self.train(dataloaders['train'], optimizers, schedulers, wrapper, scaler=scaler,
                           aux_dataloader=aux_dataloader)

                # Validate, save and log
                if has_validation_dataloader:
                    validation_output = self.validate('validation', dataloaders, prefixes, wrapper)
                    self.post_validation(validation_output, optimizers, prefixes['validation'], wrapper)

                # Take a scheduler step
                if wrapper.update_schedulers == 'epoch':
                    for scheduler in schedulers.values():
                        scheduler.step()

        # Finish logger if available
        if self.logger:
            self.logger.finish()

    def train(self, dataloader, optimizers, schedulers, wrapper, scaler=None, aux_dataloader=None):
        """Training loop for each epoch"""
        # Choose which optimizers to use
        in_optimizers, out_optimizers = self.filter_optimizers(optimizers)
        
        # Set wrapper to train
        wrapper.train_custom(in_optimizers, out_optimizers)

        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(self.current_epoch)

        # Shuffle auxiliar dataloader sampler
        if aux_dataloader is not None:
            if hasattr(aux_dataloader.sampler, "set_epoch"):
                aux_dataloader.sampler.set_epoch(self.current_epoch)

        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, aux_dataloader=aux_dataloader, ncols=120)

        # Zero gradients for the first iteration
        for optimizer in in_optimizers.values():
            optimizer.zero_grad()

        # Loop through all batches
        for i, batch in progress_bar:

            # Send samples to GPU and take a training step
            batch = sample_to_cuda(batch, self.proc_rank)
            output = wrapper.training_step(batch, epoch=self.current_epoch)

            # Step optimizer
            if wrapper.update_schedulers == 'step':
                for scheduler in schedulers.values():
                    scheduler.step()

            # Backprop through loss
            if scaler is None:
                output['loss'].backward()
            else:
                scaler.scale(output['loss']).backward()

            for optimizer in in_optimizers.values():
                if not output['loss'].isnan().any():
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                else:
                    print('NAN DETECTED!', i, batch['idx'])
                optimizer.zero_grad()
            if scaler is not None:
                scaler.update()

            self.update_averages(output)
            self.update_train_progress_bar(progress_bar)

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
                output, results = wrapper.validation_step(batch, epoch=self.current_epoch)
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

    def post_validation(self, output, optimizers, prefixes, wrapper):
        """Post-processing steps for validation"""
        self.check_and_save(wrapper, output, prefixes)
        self.log_losses_and_metrics(output, optimizers)
        self.print_logger_and_checkpoint()
        self.current_epoch += 1

    def test(self, wrapper):
        """Test a model by running validation once"""
        dataloaders, prefixes = self.prepare_dataloaders(wrapper)
        self.validate('validation', dataloaders, prefixes, wrapper)

