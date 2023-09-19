# Copyright 2023 Toyota Research Institute.  All rights reserved.

from typing import List
import numpy as np
import torch
import torch.nn as nn


def get_noised_init_vec(ang_representation: List[float],
                        translation: List[float],
                        std=0.,
                        dtype=torch.float
                        ) -> torch.Tensor:
    """
    Generate noised tensor

    Parameters
    ----------
    ang_representation : List[float]
        Flattened angular representation vector being used as `mean` in the initialization
    translation : List[float]
        Flattened values for translation vector
    std : float
        Noise to add, fed into this as the STD of Gaussian
    dtype : str
        Dtype of torch.Tensor

    Returns
    -------
    torch.Tensor
        nn.Parameter of pose vectors, which is the concatenation of (translation, euler_ang)
    """
    pose_vec = torch.concat([
        torch.tensor(np.random.normal(translation, std), dtype=dtype),
        torch.tensor(np.random.normal(ang_representation, std), dtype=dtype)],
        0)  # pose_vec == (translation, euler_ang)
    return nn.Parameter(pose_vec, requires_grad=True)
