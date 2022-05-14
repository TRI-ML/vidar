# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch

from vidar.arch.networks.BaseNet import BaseNet
from vidar.arch.networks.decoders.PoseDecoder import PoseDecoder
from vidar.arch.networks.encoders.ResNetEncoder import ResNetEncoder
from vidar.geometry.pose_utils import vec2mat


class PoseNet(BaseNet, ABC):
    """
    Pose Network

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.networks['pose_encoder'] = \
            ResNetEncoder(cfg)

        self.networks['pose'] = \
            PoseDecoder(
                num_ch_enc=self.networks['pose_encoder'].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2
            )

    def forward(self, rgb, invert):
        """Network forward pass"""

        rgb = torch.cat(rgb[::-1] if invert else rgb, 1)
        feats = self.networks['pose_encoder'](rgb)
        rotation, translation = self.networks['pose']([feats])
        transformation = vec2mat(
            rotation[:, 0], translation[:, 0], invert=invert)

        return {
            'rotation': rotation,
            'translation': translation,
            'transformation': transformation,
        }

