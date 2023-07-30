# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

from vidar.arch.models.BaseModel import BaseModel
from vidar.utils.decorators import iterate1
from vidar.utils.tensor import interpolate_image


@iterate1
def make_rgb_scales(rgb, pyramid):
    return [interpolate_image(rgb, shape=pyr.shape[-2:]) for pyr in pyramid]


class SupervisedModel(BaseModel, ABC):
    """
    Supervised depth estimation model

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """

    required_networks = ('depth',)
    required_losses = ('supervision', 'smoothness')

    def __init__(self, cfg):
        super().__init__()

    def forward(self, batch, epoch):
        """Model forward pass, given a batch dictionary"""

        rgb = batch['rgb']

        depth_output = self.networks['depth'](rgb=rgb[0])
        depths = depth_output['depths']

        if not self.training:
            return {
                'predictions': {
                    'depth': {0: depths}
                },
            }

        losses = self.compute_losses(rgb, depths, batch['depth'])

        return {
            'loss': losses['loss'],
            'metrics': {
            },
            'predictions': {
                'depth': {0: depths}
            },
        }

    def compute_losses(self, rgb, depths, gt_depths):
        """
        Compute loss and metrics for training

        Parameters
        ----------
        rgb : Dict
            Dictionary with input images [B,3,H,W]
        depths : list[torch.Tensor]
            List with target depth maps in different scales [B,1,H,W]
        gt_depths : torch.Tensor
            Ground-truth depth map for supervised training

        Returns
        -------
        loss : torch.Tensor
            Training loss
        metrics : Dict
            Dictionary with training metrics
        """
        tgt = 0

        rgbs = make_rgb_scales(rgb, depths)
        rgb_tgt = [rgbs[tgt][i] for i in range(len(rgbs[tgt]))]

        supervision_output = self.losses['supervision'](depths, gt_depths[tgt])
        smoothness_output = self.losses['smoothness'](rgb_tgt, depths)

        loss = supervision_output['loss'] + \
               smoothness_output['loss']

        metrics = {
            **supervision_output['metrics'],
            **smoothness_output['metrics'],
        }

        return {
            'loss': loss,
            'metrics': metrics,
        }
