# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import torch.nn as nn

from vidar.arch.losses.BaseLoss import BaseLoss
from vidar.utils.data import get_mask_from_list
from vidar.utils.depth import depth2index, get_depth_bins
from vidar.utils.depth import depth2inv
from vidar.utils.types import is_list


class BerHuLoss(nn.Module, ABC):
    """BerHu Loss"""
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, gt):
        huber_c = self.threshold * torch.max(pred - gt)
        diff = (pred - gt).abs()
        diff2 = diff[diff > huber_c] ** 2
        return torch.cat((diff, diff2))


class SilogLoss(nn.Module, ABC):
    """Scale Invariant Logarithmic Loss"""
    def __init__(self, ratio=10., var_ratio=0.85):
        super().__init__()
        self.ratio = ratio
        self.var_ratio = var_ratio

    def forward(self, pred, gt):
        log_diff = torch.log(pred) - torch.log(gt)
        silog1 = (log_diff ** 2).mean()
        silog2 = log_diff.mean() ** 2
        return torch.sqrt(silog1 - self.var_ratio * silog2) * self.ratio


class RMSELoss(nn.Module, ABC):
    """Root Mean Squared Error Loss"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, pred, gt):
        return torch.sqrt(self.criterion(pred, gt))


class L1LogLoss(nn.Module, ABC):
    """Root Mean Squared Error Loss"""
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss(reduction='none')

    def forward(self, pred, gt):
        return self.criterion(torch.log(pred), torch.log(gt))


class MixtureLoss(nn.Module, ABC):
    """Root Mean Squared Error Loss"""
    def __init__(self):
        super().__init__()

    @staticmethod
    def laplacian(mu, std, gt):
        std = std + 1e-12
        return 0.5 * torch.exp(-(torch.abs(mu - gt) / std)) / std

    def forward(self, pred, gt):
        mu0, mu1 = pred[:, [0]], pred[:, [1]]
        std0, std1 = pred[:, [2]], pred[:, [3]]
        w0 = pred[:, [4]]
        w1 = 1.0 - w0
        return (- torch.log(w0 * self.laplacian(mu0, std0, gt) +
                            w1 * self.laplacian(mu1, std1, gt))).mean()


class RootAbsRelLoss(nn.Module, ABC):
    """Root Mean Squared Error Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return torch.sqrt(torch.abs(pred - gt) / gt)


class SquareAbsRelLoss(nn.Module, ABC):
    """Root Mean Squared Error Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return (torch.abs(pred - gt) / gt) ** 2


class CrossEntropyLoss(nn.Module, ABC):
    """Supervised Loss"""
    def __init__(self):
        super().__init__()
        self.gamma = 2.0
        self.alpha = 0.25

        self.bootstrap_ratio = 0.3
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduce=False)

    def forward(self, pred, gt):
        min_depth, max_depth = 1.0, 100.0
        bins = get_depth_bins('linear', min_depth, max_depth, 100).to(pred.device)
        gt = depth2index(gt, bins).squeeze(1)

        loss_ce = self.cross_entropy_loss(pred, gt.to(torch.long))
        num_bootstrapping = int(self.bootstrap_ratio * pred.shape[0])
        image_errors, _ = loss_ce.view(-1, ).sort()
        worst_errors = image_errors[-num_bootstrapping:]
        return torch.mean(worst_errors)


def get_criterion(method):
    """Determines the supervised loss to be used"""
    if method == 'l1':
        return nn.L1Loss()
    elif method == 'l1log':
        return L1LogLoss()
    elif method == 'mse':
        return nn.MSELoss(reduction='none')
    elif method == 'rmse':
        return RMSELoss()
    elif method == 'huber':
        return nn.SmoothL1Loss(reduction='none')
    elif method == 'berhu':
        return BerHuLoss()
    elif method == 'silog':
        return SilogLoss()
    elif method == 'abs_rel':
        return lambda x, y: torch.abs(x - y) / x
    elif method == 'root_abs_rel':
        return RootAbsRelLoss()
    elif method == 'square_abs_rel':
        return SquareAbsRelLoss()
    elif method == 'mixture':
        return MixtureLoss()
    elif method == 'cross_entropy':
        return CrossEntropyLoss()
    else:
        raise ValueError('Unknown supervised loss {}'.format(method))


class LossWrapper(nn.Module):
    """
    Wrapper for supervised depth criteria

    Parameters
    ----------
    method : String
        Which supervised loss to use
    """
    def __init__(self, method):
        super().__init__()
        self.criterion = get_criterion(method)

    def forward(self, pred, gt, soft_mask=None):
        """
        Calculate supervised depth loss

        Parameters
        ----------
        pred : torch.Tensor
            Predicted depth [B,1,H,W]
        gt : torch.Tensor
            Ground-truth depth [B,1,H,W]
        soft_mask
            Mask for pixel weighting [B,1,H,W]

        Returns
        -------
        loss : torch.Tensor
            Supervised depth loss [1]
        """
        loss = self.criterion(pred, gt)
        if soft_mask is not None:
            loss = loss * soft_mask.detach().view(-1, 1)
        return loss.mean()


class SupervisedDepthLoss(BaseLoss, ABC):
    def __init__(self, cfg):
        """
        Supervised loss class

        Parameters
        ----------
        cfg : Config
            Configuration with parameters
        """
        super().__init__(cfg)
        self.criterion = LossWrapper(cfg.method)
        self.inverse = cfg.has('inverse', False)

    def calculate(self, pred, gt, mask=None, soft_mask=None):
        """
        Calculate supervised depth loss

        Parameters
        ----------
        pred : torch.Tensor
            Predicted depth [B,1,H,W]
        gt : torch.Tensor
            Ground-truth depth [B,1,H,W]
        mask : torch.Tensor
            Mask for pixel filtering [B,1,H,W]
        soft_mask
            Mask for pixel weighting [B,1,H,W]

        Returns
        -------
        loss : torch.Tensor
            Supervised depth loss [1]
        """
        # Interpolations
        pred = self.interp_nearest(pred, gt)
        mask = self.interp_nearest(mask, gt)
        soft_mask = self.interp_bilinear(soft_mask, gt)

        # Flatten tensors
        pred, gt, mask, soft_mask = self.flatten(pred, gt, mask, soft_mask)

        # Masks
        mask = self.mask_sparse(mask, gt)
        mask = self.mask_range(mask, gt)

        # Calculate loss
        return self.criterion(pred[mask], gt[mask],
                              soft_mask=soft_mask[mask] if soft_mask is not None else None)

    def forward(self, pred, gt, mask=None, soft_mask=None):
        """
        Supervised depth loss

        Parameters
        ----------
        pred : list[torch.Tensor]
            Predicted depths [B,1,H,W]
        gt : torch.Tensor
            Ground-truth depth [B,1,H,W]
        mask : torch.Tensor
            Mask for pixel filtering [B,1,H,W]
        soft_mask
            Mask for pixel weighting [B,1,H,W]

        Returns
        -------
        loss : torch.Tensor
            Supervised depth loss [1]
        """
        if self.inverse:
            pred, gt = depth2inv(pred), depth2inv(gt)

        scales = self.get_scales(pred)
        weights = self.get_weights(scales)

        losses, metrics = [], {}

        for i in range(scales):
            pred_i, gt_i = pred[i], gt[i] if is_list(gt) else gt
            mask_i = get_mask_from_list(mask, i, return_ones=gt_i)
            soft_mask_i = get_mask_from_list(soft_mask, i)

            loss_i = weights[i] * self.calculate(pred_i, gt_i, mask_i, soft_mask_i)

            metrics[f'supervised_depth_loss/{i}'] = loss_i.detach()
            losses.append(loss_i)

        loss = sum(losses) / len(losses)

        return {
            'loss': loss,
            'metrics': metrics,
        }
