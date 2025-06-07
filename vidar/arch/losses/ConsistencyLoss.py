# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC
from functools import partial

import torch

from knk_vision.vidar.vidar.arch.losses.BaseLoss import BaseLoss
from knk_vision.vidar.vidar.utils.data import get_mask_from_list
from knk_vision.vidar.vidar.utils.tensor import interpolate, same_shape, multiply_mask, masked_average
from knk_vision.vidar.vidar.utils.types import is_list


class ConsistencyLoss(BaseLoss, ABC):
    """Consistency loss, for enforcing consistency between teacher and student predictions."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.interpolate = partial(interpolate, mode='nearest', scale_factor=None)

    def calculate(self, teacher, student, confidence_mask, valid_mask=None):
        """Calculates the consistency loss between teacher and student predictions."""
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
        """Forward pass for the consistency loss."""
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
