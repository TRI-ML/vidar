# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC
from functools import partial

import torch

from vidar.arch.losses.BaseLoss import BaseLoss
from vidar.utils.tensor import same_shape, interpolate
from vidar.utils.types import is_list


class SupervisedSemanticLoss(BaseLoss, ABC):
    """Supervised semantic loss class."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gamma = cfg.has('gamma', 2.0)
        self.alpha = cfg.has('alpha', 0.25)
        self.interpolate = partial(interpolate, scale_factor=None, mode='nearest')

        self.bootstrap_ratio = 0.3
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduce=False)

    def calculate(self, pred, gt):
        """Calculates loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction (BxCxHxW)
        gt : torch.Tensor
            Ground truth semantic map (BxHxW)

        Returns
        -------
        loss : torch.Tensor
            Loss value
        """
        # Interpolations
        pred = self.interp_nearest(pred, gt)

        # Flatten tensors
        pred, gt, mask, soft_mask = self.flatten(pred, gt)
        gt = gt.squeeze(1)

        # Cross-entropy loss
        loss_ce = self.cross_entropy_loss(pred, gt.to(torch.long))

        # Bootstrap loss
        num_bootstrapping = int(self.bootstrap_ratio * pred.shape[0])
        image_errors, _ = loss_ce.view(-1, ).sort()
        worst_errors = image_errors[-num_bootstrapping:]

        # Calculate loss
        return torch.mean(worst_errors)

    def forward(self, pred, gt, mask=None):
        """Forward pass for semantic loss.

        Parameters
        ----------
        pred : list of torch.Tensor
            List with predictions (BxCxHxW)
        gt : torch.Tensor
            Ground truth semantic maps (BxHxW)
        mask : torch.Tensor
            Valid mask (BxHxW)

        Returns
        -------
        dict
            Dictionary containing loss and metrics
        """
        scales = self.get_scales(pred)
        weights = self.get_weights(scales)

        losses, metrics = [], {}

        for i in range(scales):
            pred_i, gt_i = pred[i], gt[i] if is_list(gt) else gt

            loss_i = weights[i] * self.calculate(pred_i, gt_i)

            metrics[f'supervised_semantic_loss/{i}'] = loss_i.detach()
            losses.append(loss_i)

        loss = sum(losses) / len(losses)

        return {
            'loss': loss,
            'metrics': metrics,
        }



