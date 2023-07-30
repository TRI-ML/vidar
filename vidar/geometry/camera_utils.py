# Copyright 2023 Toyota Research Institute.  All rights reserved.

from vidar.utils.types import is_seq


def invert_intrinsics(K):
    """Invert camera intrinsics"""

    Kinv = K.clone()
    Kinv[:, 0, 0] = 1. / K[:, 0, 0]
    Kinv[:, 1, 1] = 1. / K[:, 1, 1]
    Kinv[:, 0, 2] = -1. * K[:, 0, 2] / K[:, 0, 0]
    Kinv[:, 1, 2] = -1. * K[:, 1, 2] / K[:, 1, 1]
    return Kinv


def scale_intrinsics(K, ratio):
    """Scale intrinsics given x_scale and y_scale factors"""

    if is_seq(ratio):
        ratio_h, ratio_w = ratio
    else:
        ratio_h = ratio_w = ratio

    K = K.clone()

    K[..., 0, 0] *= ratio_w
    K[..., 1, 1] *= ratio_h

    K[..., 0, 2] = K[..., 0, 2] * ratio_w
    K[..., 1, 2] = K[..., 1, 2] * ratio_h

    return K
