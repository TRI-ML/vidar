# Copyright 2023 Toyota Research Institute.  All rights reserved.

from torch.optim.lr_scheduler import LambdaLR


def get_step_schedule_with_warmup(optimizer, lr_start, warmup_epochs, epoch_size, step_size, gamma):
    """Step schedule with warmup"""
    def lr_lambda(current_step: int):
        if current_step < warmup_epochs:
            return lr_start + (1.0 - lr_start) * (float(current_step) / float(max(1, warmup_epochs)))
        else:
            return 1.0 * gamma ** float((current_step - warmup_epochs) // (epoch_size * step_size))
    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """Linear schedule with warmup"""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)