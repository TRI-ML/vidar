# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC
from functools import partial

import torch

from vidar.arch.losses.BaseLoss import BaseLoss
from vidar.utils.data import get_mask_from_list
from vidar.utils.tensor import interpolate, same_shape, multiply_mask, masked_average
from vidar.utils.types import is_list


class ConsistencyLoss(BaseLoss, ABC):
    """
    Consistency loss class

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.interpolate = partial(
            interpolate, mode='nearest', scale_factor=None, align_corners=None)

    def calculate(self, teacher, student, confidence_mask, valid_mask=None):
        """
        Calculate consistency loss

        Parameters
        ----------
        teacher : torch.Tensor
            Teacher depth predictions [B,1,H,W]
        student : torch.Tensor
            Student depth predictions [B,1,H,W]
        confidence_mask : torch.Tensor
            Confidence mask for pixel selection [B,1,H,W]
        valid_mask : torch.Tensor
            Valid mask for pixel selection [B,1,H,W]

        Returns
        -------
        loss : torch.Tensor
            Consistency loss [1]
        """
        if not same_shape(teacher.shape[-2:], student.shape[-2:]):
            teacher = self.interpolate(teacher, size=student.shape[-2:])
        if not same_shape(confidence_mask.shape, teacher.shape):
            confidence_mask = self.interpolate(confidence_mask, size=teacher.shape[-2:])
        if valid_mask is not None and not same_shape(valid_mask.shape, teacher.shape):
            valid_mask = self.interpolate(valid_mask, size=teacher.shape[-2:])
        non_confidence_mask = (1 - confidence_mask).float()
        consistency_loss = torch.abs(student - teacher.detach())
        return masked_average(consistency_loss,
                              multiply_mask(non_confidence_mask, valid_mask))

    def forward(self, teacher, student, confidence_mask, valid_mask=None):
        """
        Forward loop for loss calculation

        Parameters
        ----------
        teacher : list[torch.Tensor]
            Teacher depth predictions [B,1,H,W]
        student : list[torch.Tensor]
            Student depth predictions [B,1,H,W]
        confidence_mask : list[torch.Tensor]
            Confidence mask for pixel selection [B,1,H,W]
        valid_mask : list[torch.Tensor]
            Valid mask for pixel selection [B,1,H,W]

        Returns
        -------
        output : Dict
            Dictionary with loss and metrics
        """
        scales = self.get_scales(student)
        weights = self.get_weights(scales)

        losses, metrics = [], {}

        for i in range(scales):
            teacher_i = teacher[i] if is_list(teacher) else teacher
            student_i = student[i] if is_list(student) else student
            confidence_mask_i = get_mask_from_list(confidence_mask, i)
            valid_mask_i = get_mask_from_list(valid_mask, i)

            loss_i = weights[i] * self.calculate(
                teacher_i, student_i, confidence_mask_i, valid_mask_i)

            metrics[f'consistency_loss/{i}'] = loss_i.detach()
            losses.append(loss_i)

        loss = sum(losses) / len(losses)

        return {
            'loss': loss,
            'metrics': metrics,
        }
