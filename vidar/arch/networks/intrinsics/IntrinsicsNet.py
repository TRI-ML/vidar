# Copyright 2023 Toyota Research Institute.  All rights reserved.

from __future__ import absolute_import, division, print_function

from abc import ABC

import numpy as np
import torch
import torch.nn as nn

from knk_vision.vidar.vidar.arch.networks.BaseNet import BaseNet


class IntrinsicsNet(BaseNet, ABC):
    """
    Intrinsics networks for camera geometry.

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.has('shape')
        self.image_shape = cfg.shape
        self.geometry = cfg.has('camera_model', 'pinhole')
        self.sigmoid_init = nn.Parameter(
            torch.tensor(self.setup_sigmoid_init(cfg), dtype=torch.float), requires_grad=True)
        self.scale = nn.Parameter(
            torch.tensor(self.setup_scale(cfg), dtype=torch.float, requires_grad=False), requires_grad=False)
        self.offset = nn.Parameter(
            torch.tensor(self.setup_offset(cfg), dtype=torch.float, requires_grad=False), requires_grad=False)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def __len__(self):
        if self.geometry == 'pinhole':
            return 4
        elif self.geometry == 'ucm':
            return 5
        else:
            raise NotImplementedError('Invalid camera model')

    def setup_sigmoid_init(self, cfg):
        """ Setup initial sigmoid values for camera intrinsics"""
        if cfg.has('sigmoid_init'):
            assert len(cfg.sigmoid_init) == self.__len__()
            return np.array(cfg.sigmoid_init)
        else:
            if self.geometry == 'pinhole':
                return np.array([0.0, 0.0, 0.0, 0.0])
            elif self.geometry == 'ucm':
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                raise NotImplementedError('Invalid camera model')

    def setup_scale(self, cfg):
        """ Setup scale for camera intrinsics"""
        if cfg.has('scale'):
            assert len(cfg.scale) == self.__len__()
            return np.array(cfg.scale)
        else:
            h, w = self.image_shape
            fx_scale, fy_scale = (h + w), (h + w)
            cx_scale, cy_scale = w, h
            alpha_scale = 1.0

            if self.geometry == 'pinhole':
                return np.array([fx_scale, fy_scale, cx_scale, cy_scale])
            elif self.geometry == 'ucm':
                return np.array([fx_scale, fy_scale, cx_scale, cy_scale, alpha_scale])
            else:
                raise NotImplementedError('Invalid camera model')

    def setup_offset(self, cfg):
        """ Setup offset for camera intrinsics"""
        if cfg.has('offset'):
            assert len(cfg.offset) == self.__len__()
            return np.array(cfg.offset)
        else:
            if self.geometry == 'pinhole':
                return np.zeros(4)
            elif self.geometry == 'ucm':
                return np.zeros(5)
            else:
                raise NotImplementedError('Invalid camera model')

    def forward(self, rgb):
        """Network forward pass"""

        self.scale.requires_grad = False
        self.offset.requires_grad = False

        b = rgb.shape[0]
        K = self.sigmoid(self.sigmoid_init) * self.scale + self.offset

        return K.unsqueeze(0).repeat(b, 1)
