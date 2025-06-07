# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch

from knk_vision.vidar.vidar.arch.losses.BaseLoss import BaseLoss
from knk_vision.vidar.vidar.utils.data import get_from_list, get_mask_from_list
from knk_vision.vidar.vidar.utils.depth import calculate_normals


def grad_mask(mask):
    """Calculates gradient mask for surface normals."""
    mask = mask[..., :-1, :-1] & mask[..., 1:, :-1] & mask[..., :-1, 1:] & mask[..., 1:, 1:]
    return torch.nn.functional.pad(mask, [0, 1, 0, 1], mode='constant', value=0)


class SupervisedNormalsLoss(BaseLoss, ABC):
    """Supervised surface normals loss class."""
    def __init__(self, cfg):
        super().__init__(cfg)
        # Cosine similarity criterion for loss calculation
        self.criterion = torch.nn.CosineSimilarity(dim=1)

    def calc(self, pred, gt):
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
        loss = self.criterion(pred, gt).unsqueeze(1)
        mask = ~torch.isnan(loss)
        return (1.0 - loss[mask].mean()) / 2

    def calculate(self, pred, gt, gt_depth, camera, mask):
        """Calculates loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction (BxCxHxW)
        gt : torch.Tensor
            Ground truth semantic map (BxHxW)
        gt_depth : torch.Tensor
            Ground truth depth map (BxHxW)
        camera : Camera
            Camera object
        mask : torch.Tensor
            Valid mask (BxHxW)

        Returns
        -------
        loss : torch.Tensor
            Loss value
        """
        # Transform depth maps in surface normals
        if camera is not None:
            pred = calculate_normals(pred, camera=camera, pad_last=True)
            gt = calculate_normals(gt, camera=camera, pad_last=True)

        # Interpolations
        gt = self.interp_nearest(gt, pred)
        mask = self.interp_nearest(mask, pred)

        # Flatten tensors
        # pred, gt, mask, soft_mask, logvar = self.flatten(pred, gt, mask)

        mask = self.mask_range(mask, gt_depth)
        mask = grad_mask(mask)

        pred = pred.permute(0, 2, 3, 1).reshape(-1, 3)
        gt = gt.permute(0, 2, 3, 1).reshape(-1, 3)
        mask = mask.permute(0, 2, 3, 1).reshape(-1)

        # Calculate loss
        return self.calc(pred[mask], gt[mask])

    def forward(self, pred, gt, gt_depth=None, camera=None, mask=None, epoch=None):
        """Forward pass for surface normals loss.

        Parameters
        ----------
        pred : list of torch.Tensor
            List with predictions (BxCxHxW)
        gt : torch.Tensor
            Ground truth normals (BxHxW)
        gt_depth : torch.Tensor
            Ground truth depth map (BxHxW)
        camera : Camera
            Camera object
        mask : torch.Tensor
            Valid mask (BxHxW)

        Returns
        -------
        dict
            Dictionary containing loss and metrics
        """
        scales = self.get_scales(pred)
        weights = self.get_weights(scales, epoch)

        losses, metrics = [], {}

        for i in range(scales):
            pred_i, gt_i = get_from_list(pred, i), get_from_list(gt, i)
            camera_i = get_from_list(camera, i)
            mask_i = get_mask_from_list(mask, i, return_ones=gt_i)
            gt_depth_i = get_from_list(gt_depth, i)

            loss_i = weights[i] * self.calculate(pred_i, gt_i, gt_depth_i, camera_i, mask_i)

            metrics[f'supervised_normals_loss/{i}'] = loss_i.detach()
            losses.append(loss_i)

        loss = sum(losses) / len(losses)

        return {
            'loss': loss,
            'metrics': metrics,
        }