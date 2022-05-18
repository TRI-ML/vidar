from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

from abc import ABC

from vidar.arch.networks.BaseNet import BaseNet
from vidar.utils.config import cfg_has


class IntrinsicsNet(BaseNet, ABC):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg_has(cfg, 'shape')
        self.image_shape = cfg.shape
        self.camera_model = cfg_has(cfg, 'camera_model', 'UCM')
        self.sigmoid_init = nn.Parameter(torch.tensor(self.setup_sigmoid_init(cfg), dtype=torch.float), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(self.setup_scale(cfg), dtype=torch.float, requires_grad=False), requires_grad=False)
        self.offset = nn.Parameter(torch.tensor(self.setup_offset(cfg), dtype=torch.float, requires_grad=False), requires_grad=False)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def __len__(self):
        if self.camera_model == 'Pinhole':
            return 4
        elif self.camera_model == 'UCM':
            return 5
        elif self.camera_model == 'EUCM':
            return 6
        elif self.camera_model == 'DS':
            return 6
        else:
            raise NotImplementedError('Camera model {} is not implemented. Please choose from \{Pinhole,UCM, EUCM, DS\}.'.format(self.camera_model))

    def setup_sigmoid_init(self, cfg):
        if cfg_has(cfg, 'sigmoid_init'):
            assert len(cfg.sigmoid_init) == self.__len__()
            return np.array(cfg.sigmoid_init)
        else:
            if self.camera_model == 'Pinhole':
                return np.array([0.0, 0.0, 0.0, 0.0])
            elif self.camera_model == 'UCM':
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            elif self.camera_model == 'EUCM':
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif self.camera_model == 'DS':
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                raise NotImplementedError('Camera model {} is not implemented. Please choose from \{Pinhole,UCM, EUCM, DS\}.'.format(self.camera_model))

    def setup_scale(self, cfg):
        if cfg_has(cfg, 'scale'):
            assert len(cfg.scale) == self.__len__()
            return np.array(cfg.scale)
        else:
            h, w = self.image_shape
            fx_scale, fy_scale = (h + w), (h + w)
            cx_scale = w
            cy_scale = h
            alpha_scale = 1.0
            beta_scale = 2.0
            xi_scale = 2.0

            if self.camera_model == 'Pinhole':
                return np.array([fx_scale, fy_scale, cx_scale, cy_scale])
            elif self.camera_model == 'UCM':
                return np.array([fx_scale, fy_scale, cx_scale, cy_scale, alpha_scale])
            elif self.camera_model == 'EUCM':
                return np.array([fx_scale, fy_scale, cx_scale, cy_scale, alpha_scale, beta_scale])
            elif self.camera_model == 'DS':
                return np.array([fx_scale, fy_scale, cx_scale, cy_scale, xi_scale, alpha_scale])
            else:
                raise NotImplementedError('Camera model {} is not implemented. Please choose from \{Pinhole,UCM, EUCM, DS\}.'.format(self.camera_model))

    def setup_offset(self, cfg):
        if cfg_has(cfg, 'offset'):
            assert len(cfg.offset) == self.__len__()
            return np.array(cfg.offset)
        else:
            if self.camera_model == 'Pinhole':
                return np.zeros(4)
            elif self.camera_model == 'UCM':
                return np.zeros(5)
            elif self.camera_model == 'EUCM':
                return np.zeros(6)
            elif self.camera_model == 'DS':
                return np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
            else:
                raise NotImplementedError('Camera model {} is not implemented. Please choose from \{Pinhole,UCM, EUCM, DS\}.'.format(self.camera_model))


    def forward(self, rgb):
        B = rgb.shape[0]

        self.scale.requires_grad = False
        self.offset.requires_grad = False

        I = self.sigmoid(self.sigmoid_init) * self.scale + self.offset

        return I.unsqueeze(0).repeat(B,1)