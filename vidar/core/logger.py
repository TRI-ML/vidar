# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import wandb

from vidar.utils.config import cfg_has
from vidar.utils.distributed import world_size
from vidar.utils.logging import pcolor
from vidar.utils.types import is_dict, is_tensor, is_seq, is_namespace
from vidar.utils.viz import viz_depth, viz_inv_depth, viz_normals, viz_optical_flow, viz_camera


class WandbLogger:
    """
    Wandb logger class to monitor training

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    verbose : Bool
        Print information on screen if enabled
    """
    def __init__(self, cfg, verbose=False):
        super().__init__()

        self.num_logs = {
            'train': cfg_has(cfg, 'num_train_logs', 0),
            'val': cfg_has(cfg, 'num_validation_logs', 0),
            'test': cfg_has(cfg, 'num_test_logs', 0),
        }

        self._name = cfg.name if cfg_has(cfg, 'name') else None
        self._dir = cfg.folder
        self._entity = cfg.entity
        self._project = cfg.project

        self._tags = cfg_has(cfg, 'tags', '')
        self._notes = cfg_has(cfg, 'notes', '')

        self._id = None
        self._anonymous = None
        self._log_model = True

        self._experiment = self._create_experiment()
        self._metrics = OrderedDict()

        self.only_first = cfg_has(cfg, 'only_first', False)

        cfg.name = self.run_name
        cfg.url = self.run_url

        if verbose:
            self.print()

    @staticmethod
    def finish():
        """Finish wandb session"""
        wandb.finish()

    def print(self):
        """Print information on screen"""

        font_base = {'color': 'red', 'attrs': ('bold', 'dark')}
        font_name = {'color': 'red', 'attrs': ('bold',)}
        font_underline = {'color': 'red', 'attrs': ('underline',)}

        print(pcolor('#' * 60, **font_base))
        print(pcolor('### WandB: ', **font_base) + \
              pcolor('{}'.format(self.run_name), **font_name))
        print(pcolor('### ', **font_base) + \
              pcolor('{}'.format(self.run_url), **font_underline))
        print(pcolor('#' * 60, **font_base))

    def __getstate__(self):
        """Get the current logger state"""
        state = self.__dict__.copy()
        state['_id'] = self._experiment.id if self._experiment is not None else None
        state['_experiment'] = None
        return state

    def _create_experiment(self):
        """Creates and returns a new experiment"""
        experiment = wandb.init(
            name=self._name, dir=self._dir, project=self._project,
            anonymous=self._anonymous, reinit=True, id=self._id, notes=self._notes,
            resume='allow', tags=self._tags, entity=self._entity
        )
        wandb.run.save()
        return experiment

    def watch(self, model: nn.Module, log='gradients', log_freq=100):
        """Watch training parameters"""
        self.experiment.watch(model, log=log, log_freq=log_freq)

    @property
    def experiment(self):
        """Returns the experiment (creates a new if it doesn't exist)"""
        if self._experiment is None:
            self._experiment = self._create_experiment()
        return self._experiment

    @property
    def run_name(self):
        """Returns run name"""
        return wandb.run.name if self._experiment else None

    @property
    def run_url(self):
        """Returns run URL"""
        return f'https://app.wandb.ai/' \
               f'{wandb.run.entity}/' \
               f'{wandb.run.project}/runs/' \
               f'{wandb.run.id}' if self._experiment else None

    def log_config(self, cfg):
        """Log model configuration"""
        cfg = recursive_convert_config(deepcopy(cfg))
        self.experiment.config.update(cfg, allow_val_change=True)

    def log_metrics(self, metrics):
        """Log training metrics"""
        self._metrics.update(metrics)
        if 'epochs' in metrics or 'samples' in metrics:
            self.experiment.log(self._metrics)
            self._metrics.clear()

    def log_images(self, batch, output, prefix, ontology=None):
        """
        Log images depending on its nature

        Parameters
        ----------
        batch : Dict
            Dictionary containing batch information
        output : Dict
            Dictionary containing output information
        prefix : String
            Prefix string for the log name
        ontology : Dict
            Dictionary with ontology information
        """
        for data, suffix in zip([batch, output['predictions']], ['-gt', '-pred']):
            for key in data.keys():
                if key.startswith('rgb'):
                    self._metrics.update(log_rgb(
                        key, prefix + suffix, data, only_first=self.only_first))
                elif key.startswith('depth'):
                    self._metrics.update(log_depth(
                        key, prefix + suffix, data, only_first=self.only_first))
                elif key.startswith('inv_depth'):
                    self._metrics.update(log_inv_depth(
                        key, prefix + suffix, data, only_first=self.only_first))
                elif 'normals' in key:
                    self._metrics.update(log_normals(
                        key, prefix + suffix, data, only_first=self.only_first))
                elif key.startswith('stddev'):
                    self._metrics.update(log_stddev(
                        key, prefix + suffix, data, only_first=self.only_first))
                elif key.startswith('logvar'):
                    self._metrics.update(log_logvar(
                        key, prefix + suffix, data, only_first=self.only_first))
                elif 'optical_flow' in key:
                    self._metrics.update(log_optical_flow(
                        key, prefix + suffix, data, only_first=self.only_first))
                elif 'mask' in key or 'valid' in key:
                    self._metrics.update(log_rgb(
                        key, prefix, data, only_first=self.only_first))
                # elif 'camera' in key:
                #     self._metrics.update(log_camera(
                #         key, prefix + suffix, data, only_first=self.only_first))
                # elif 'uncertainty' in key:
                #     self._metrics.update(log_uncertainty(key, prefix, data))
                # elif 'semantic' in key and ontology is not None:
                #     self._metrics.update(log_semantic(key, prefix, data, ontology=ontology))
                # if 'scene_flow' in key:
                #     self._metrics.update(log_scene_flow(key, prefix_idx, data))
                # elif 'score' in key:
                #     # Log score as image heatmap
                #     self._metrics.update(log_keypoint_score(key, prefix, data))

    def log_data(self, mode, batch, output, dataset, prefix, ontology=None):
        """Helper function used to log images"""
        idx = batch['idx'][0]
        num_logs = self.num_logs[mode]
        if num_logs > 0:
            interval = (len(dataset) // world_size() // num_logs) * world_size()
            if interval == 0 or (idx % interval == 0 and idx < interval * num_logs):
                prefix = '{}-{}-{}'.format(mode, prefix, batch['idx'][0].item())
                # batch, output = prepare_logging(batch, output)
                self.log_images(batch, output, prefix, ontology=ontology)


def recursive_convert_config(cfg):
    """Convert configuration to dictionary recursively"""
    cfg = cfg.__dict__
    for key, val in cfg.items():
        if is_namespace(val):
            cfg[key] = recursive_convert_config(val)
    return cfg


def prep_image(key, prefix, image):
    """Prepare image for logging"""
    if is_tensor(image):
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.dim() == 4:
            image = image[0]
        image = image.detach().permute(1, 2, 0).cpu().numpy()
    prefix_key = '{}-{}'.format(prefix, key)
    return {prefix_key: wandb.Image(image, caption=key)}


def log_sequence(key, prefix, data, i, only_first, fn):
    """Logs a sequence of images (list, tuple or dict)"""
    log = {}
    if is_dict(data):
        for ctx, dict_val in data.items():
            if is_seq(dict_val):
                if only_first:
                    dict_val = dict_val[:1]
                for idx, list_val in enumerate(dict_val):
                    if list_val.dim() == 5:
                        for j in range(list_val.shape[1]):
                            log.update(fn('%s(%s_%d)_%d' % (key, str(ctx), j, idx), prefix, list_val[:, j], i))
                    else:
                        log.update(fn('%s(%s)_%d' % (key, str(ctx), idx), prefix, list_val, i))
            else:
                if dict_val.dim() == 5:
                    for j in range(dict_val.shape[1]):
                        log.update(fn('%s(%s_%d)' % (key, str(ctx), j), prefix, dict_val[:, j], i))
                else:
                    log.update(fn('%s(%s)' % (key, str(ctx)), prefix, dict_val, i))
    elif is_seq(data):
        if only_first:
            data = data[:1]
        for idx, list_val in enumerate(data):
            log.update(fn('%s_%d' % (key, idx), prefix, list_val, i))
    else:
        log.update(fn('%s' % key, prefix, data, i))
    return log


def log_rgb(key, prefix, batch, i=0, only_first=None):
    """Log RGB image"""
    rgb = batch[key] if is_dict(batch) else batch
    if is_seq(rgb) or is_dict(rgb):
        return log_sequence(key, prefix, rgb, i, only_first, log_rgb)
    return prep_image(key, prefix, rgb[i].clamp(min=0.0, max=1.0))


def log_depth(key, prefix, batch, i=0, only_first=None):
    """Log depth map"""
    depth = batch[key] if is_dict(batch) else batch
    if is_seq(depth) or is_dict(depth):
        return log_sequence(key, prefix, depth, i, only_first, log_depth)
    return prep_image(key, prefix, viz_depth(depth[i], filter_zeros=True))


def log_inv_depth(key, prefix, batch, i=0, only_first=None):
    """Log inverse depth map"""
    inv_depth = batch[key] if is_dict(batch) else batch
    if is_seq(inv_depth) or is_dict(inv_depth):
        return log_sequence(key, prefix, inv_depth, i, only_first, log_inv_depth)
    return prep_image(key, prefix, viz_inv_depth(inv_depth[i]))


def log_normals(key, prefix, batch, i=0, only_first=None):
    """Log normals"""
    normals = batch[key] if is_dict(batch) else batch
    if is_seq(normals) or is_dict(normals):
        return log_sequence(key, prefix, normals, i, only_first, log_normals)
    return prep_image(key, prefix, viz_normals(normals[i]))


def log_optical_flow(key, prefix, batch, i=0, only_first=None):
    """Log optical flow"""
    optical_flow = batch[key] if is_dict(batch) else batch
    if is_seq(optical_flow) or is_dict(optical_flow):
        return log_sequence(key, prefix, optical_flow, i, only_first, log_optical_flow)
    return prep_image(key, prefix, viz_optical_flow(optical_flow[i]))


def log_stddev(key, prefix, batch, i=0, only_first=None):
    """Log standard deviation"""
    stddev = batch[key] if is_dict(batch) else batch
    if is_seq(stddev) or is_dict(stddev):
        return log_sequence(key, prefix, stddev, i, only_first, log_stddev)
    return prep_image(key, prefix, viz_inv_depth(stddev[i], colormap='jet'))


def log_logvar(key, prefix, batch, i=0, only_first=None):
    """Log standard deviation"""
    logvar = batch[key] if is_dict(batch) else batch
    if is_seq(logvar) or is_dict(logvar):
        return log_sequence(key, prefix, logvar, i, only_first, log_logvar)
    return prep_image(key, prefix, viz_inv_depth(torch.exp(logvar[i]), colormap='jet'))


def log_camera(key, prefix, batch, i=0, only_first=None):
    """Log camera"""
    camera = batch[key] if is_dict(batch) else batch
    if is_seq(camera) or is_dict(camera):
        return log_sequence(key, prefix, camera, i, only_first, log_camera)
    return prep_image(key, prefix, viz_camera(camera[i]))

