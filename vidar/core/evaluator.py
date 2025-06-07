# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict

import torch
from tqdm import tqdm

from knk_vision.vidar.vidar.core.checkpoint import ModelCheckpoint
from knk_vision.vidar.vidar.utils.config import cfg_has, dataset_prefix
from knk_vision.vidar.vidar.utils.data import make_list, keys_in
from knk_vision.vidar.vidar.utils.setup import setup_dataloader, reduce
from knk_vision.vidar.vidar.utils.types import is_dict, is_seq, is_numpy, is_tensor, is_list


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


class Evaluator:
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

        self.all_modes = ['train', 'mixed', 'validation', 'test']
        self.train_modes = ['train', 'mixed']

        self.current_epoch = 0

    @property
    def progress(self):
        """Current epoch progress (percentage)"""
        return self.current_epoch / self.max_epochs

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

    def prep_logger_and_checkpoint(self, cfg):
        """Prepare logger and checkpoint class if requested"""
        add_checkpoint = cfg_has(cfg, 'checkpoint')

        if add_checkpoint:
            self.checkpoint = ModelCheckpoint(cfg.checkpoint, verbose=True)
        else:
            self.checkpoint = None

            
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

