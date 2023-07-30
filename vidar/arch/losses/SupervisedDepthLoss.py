# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import torch.nn as nn

from vidar.arch.losses.BaseLoss import BaseLoss
from vidar.utils.data import get_mask_from_list, get_from_list
from vidar.utils.depth import depth2index, get_depth_bins, depth2inv, scale_and_shift_pred, align


def grad(x):
    """ Calculate gradient of an image"""
    diff_x = x[..., 1:, :-1] - x[..., :-1, :-1]
    diff_y = x[..., :-1, 1:] - x[..., :-1, :-1]
    mag = torch.sqrt(diff_x ** 2 + diff_y ** 2)
    angle = torch.atan(diff_y / (diff_x + 1e-6))

    return mag, angle


def grad_mask(mask):
    """ Calculate gradient mask"""
    return mask[..., :-1, :-1] & mask[..., 1:, :-1] & mask[..., :-1, 1:] & mask[..., 1:, 1:]


class GradientLoss(nn.Module, ABC):
    """ Gradient loss class, to enforce gradient consistency in depth predictions."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        """ Forward pass for the gradient loss."""
        pred_grad, gt_grad = grad(pred), grad(gt)
        mask_grad = grad_mask(mask)

        loss_mag = (pred_grad[0][mask_grad] - gt_grad[0][mask_grad]).abs().sqrt()
        loss_ang = (pred_grad[1][mask_grad] - gt_grad[1][mask_grad]).abs()

        loss_mag = torch.clip(loss_mag, min=0.0, max=10.0)

        ratio = 0.10
        valid = int(ratio * loss_mag.shape[0])
        loss_mag = loss_mag.sort()[0][valid:-valid].mean()
        loss_ang = loss_ang.sort()[0][valid:-valid:].mean()

        return loss_mag + loss_ang


class ShiftScaleLoss(nn.Module, ABC):
    """ Scale and shift invariant depth loss class."""
    def __init__(self, cfg):
        super().__init__()
        self.shift_loss = cfg.has('shift_weight', 0.0)
        self.scale_loss = cfg.has('scale_weight', 0.0)

    def forward(self, pred, gt, mask):
        scale, shift, scaled_pred = scale_and_shift_pred(pred, gt, mask)
        loss = nn.functional.l1_loss(scaled_pred[mask], gt[mask])
        if self.shift_loss > 0.0:
            loss = loss + self.shift_loss * (1. - scale).abs()
        if self.scale_loss > 0.0:
            loss = loss + self.scale_loss * shift.abs()
        return loss


class MidasLoss(nn.Module, ABC):
    """Scale and shift invariant class. Taken from https://github.com/isl-org/MiDaS"""
    def __init__(self, cfg):
        super().__init__()
        self.shift_loss = cfg.has('shift_weight', 0.0)
        self.scale_loss = cfg.has('scale_weight', 0.0)
        self.grad_loss = cfg.has('grad_weight', 0.0)

    def forward(self, pred, gt, mask, eps=1e-6):
        pred_shift, pred_scale, pred_aligned = align(pred, mask)
        gt_shift, gt_scale, gt_aligned = align(gt, mask)
        loss = nn.functional.l1_loss(pred_aligned[mask], gt_aligned[mask])
        if self.shift_loss > 0.0:
            loss = loss + self.shift_loss * (pred_shift - gt_shift).abs().mean()
        if self.scale_loss > 0.0:
            loss = loss + self.scale_loss * (pred_scale - gt_scale).abs().mean()
        if self.grad_loss > 0.0:
            mask_grad = grad_mask(mask)
            pred_aligned_grad = grad(pred_aligned)
            gt_aligned_grad = grad(gt_aligned)
            loss_mag = nn.functional.l1_loss(pred_aligned_grad[0][mask_grad], gt_aligned_grad[0][mask_grad])
            loss_ang = nn.functional.l1_loss(pred_aligned_grad[1][mask_grad], gt_aligned_grad[1][mask_grad])
            loss = loss + self.grad_loss * (loss_mag + loss_ang)
        return loss

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


def get_criterion(cfg):
    """Determines the supervised loss to be used"""
    if cfg.method == 'l1':
        return nn.L1Loss()
    elif cfg.method == 'l1log':
        return L1LogLoss()
    elif cfg.method == 'mse':
        return nn.MSELoss(reduction='none')
    elif cfg.method == 'rmse':
        return RMSELoss()
    elif cfg.method == 'huber':
        return nn.SmoothL1Loss(reduction='none')
    elif cfg.method == 'berhu':
        return BerHuLoss()
    elif cfg.method == 'silog':
        return SilogLoss()
    elif cfg.method == 'abs_rel':
        return lambda x, y: torch.abs(x - y) / x
    elif cfg.method == 'root_abs_rel':
        return RootAbsRelLoss()
    elif cfg.method == 'square_abs_rel':
        return SquareAbsRelLoss()
    elif cfg.method == 'mixture':
        return MixtureLoss()
    elif cfg.method == 'shift_scale':
        return ShiftScaleLoss(cfg)
    elif cfg.method == 'midas':
        return MidasLoss(cfg)
    elif cfg.method == 'gradient':
        return GradientLoss()
    else:
        raise ValueError('Unknown supervised loss {}'.format(cfg.method))


class LossWrapper(nn.Module):
    """Wrapper for loss functions."""
    def __init__(self, cfg):
        super().__init__()
        self.criterion = get_criterion(cfg)
        self.method = cfg.method

    def forward(self, pred, gt, mask, soft_mask=None, logvar=None):
        """Forward pass for loss wrapper.

        Parameters
        ----------
        pred : list of torch.Tensor
            List with predictions (BxCxHxW)
        gt : torch.Tensor
            Ground truth image (BxHxW)
        soft_mask : list of torch.Tensor
            List with soft masks for per-pixel loss calculation (BxHxW)
        logvar : torch.Tensor
            Log-variance value for uncertainty weighting (BxHxW)

        Returns
        -------
        loss : torch.Tensor
            Loss value
        """        
        if self.method in ['shift_scale', 'midas', 'gradient']:
            loss = self.criterion(pred, gt, mask)
        else:
            pred = pred[mask] if mask is not None else pred
            gt = gt[mask] if mask is not None else gt
            loss = self.criterion(pred, gt)
        if soft_mask is not None:
            loss = loss * soft_mask.detach().view(-1, 1)
        if logvar is not None:
            loss = loss * torch.exp(-logvar)
        loss = loss.mean()
        if logvar is not None:
            loss = loss + logvar.mean()
        return loss


class SupervisedDepthLoss(BaseLoss, ABC):
    """Supervised Loss"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.criterion = LossWrapper(cfg)
        self.inverse = cfg.has('inverse', False)
        self.background = cfg.has('background', None)
        self.scale_shift_gt_from_pred = cfg.has('scale_shift_gt_from_pred', False)

    def calculate(self, pred, gt, mask=None, soft_mask=None, logvar=None):
        """Calculates loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction (BxCxHxW)
        gt : torch.Tensor
            Ground truth (BxHxW)
        mask : torch.Tensor
            Valid Mask (BxHxW)
        soft_mask : torch.Tensor
            Soft mask for per-pixel loss calculation (BxHxW)
        logvar : torch.Tensor
            Log-variance value for uncertainty weighting (BxHxW)

        Returns
        -------
        loss : torch.Tensor
            Loss value
        """
        # Interpolations
        pred = self.interp_nearest(pred, gt)
        mask = self.interp_nearest(mask, gt)
        soft_mask = self.interp_bilinear(soft_mask, gt)
        logvar = self.interp_bilinear(logvar, gt)

        # Masks
        mask = self.mask_sparse(mask, gt)
        mask = self.mask_range(mask, gt)

        # Calculate loss
        loss = self.criterion(
            pred, gt, mask, soft_mask=soft_mask, logvar=logvar,
        )

        ##
        if self.background is not None:
            mask_bkg = gt >= self.background[0]
            if mask_bkg.sum() > 0:
                loss_bkg = self.criterion(
                    pred[mask_bkg], gt[mask_bkg],
                    soft_mask=soft_mask[mask_bkg] if soft_mask is not None else None,
                    logvar=logvar[mask_bkg] if logvar is not None else None,
                )
                loss = loss + self.background[1] * loss_bkg
        ##

        return loss

    def forward(self, pred, gt, mask=None, soft_mask=None, logvar=None, epoch=None):
        """Forward pass for depth loss.

        Parameters
        ----------
        pred : list of torch.Tensor
            List with predictions (BxCxHxW)
        gt : torch.Tensor
            Ground truth depth map (BxHxW)
        mask : torch.Tensor
            Valid mask (BxHxW)
        soft_mask : list of torch.Tensor
            List with soft masks for per-pixel loss calculation (BxHxW)
        logvar : list of torch.Tensor
            List with log-variance value for uncertainty weighting (BxHxW)
        epoch : int
            Current epoch

        Returns
        -------
        dict
            Dictionary containing loss and metrics
        """
        if self.inverse:
            pred, gt = depth2inv(pred), depth2inv(gt)

        scales = self.get_scales(pred)
        weights = self.get_weights(scales, epoch)


        losses, metrics = [], {}

        for i in range(scales):
            pred_i, gt_i = get_from_list(pred, i), get_from_list(gt, i)
            mask_i = get_mask_from_list(mask, i, return_ones=gt_i)
            soft_mask_i = get_mask_from_list(soft_mask, i)
            logvar_i = get_from_list(logvar, i)

            if self.scale_shift_gt_from_pred:
                gt_i = scale_and_shift_pred(gt_i, pred_i)[-1]

            loss_i = weights[i] * self.calculate(pred_i, gt_i, mask_i, soft_mask_i, logvar_i)

            metrics[f'supervised_depth_loss/{i}'] = loss_i.detach()
            losses.append(loss_i)

        loss = sum(losses) / len(losses)

        return {
            'loss': loss,
            'metrics': metrics,
        }
