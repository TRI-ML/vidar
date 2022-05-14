# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

from vidar.arch.networks.BaseNet import BaseNet
from vidar.arch.networks.decoders.DepthDecoder import DepthDecoder
from vidar.arch.networks.encoders.ResNetEncoder import ResNetEncoder
from vidar.utils.config import cfg_has


class MonoDepthResNet(BaseNet, ABC):
    """
    Single-frame monocular depth network

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.num_scales = cfg_has(cfg, 'num_scales', 4)
        self.set_attr(cfg, 'scale_intrinsics', False)

        self.networks['mono_encoder'] = ResNetEncoder(cfg.encoder)
        cfg.decoder.num_ch_enc = self.networks['mono_encoder'].num_ch_enc
        self.networks['mono_depth'] = DepthDecoder(cfg.decoder)

    def forward(self, rgb, intrinsics=None):
        """Network forward pass"""

        features = self.networks['mono_encoder'](rgb)
        output = self.networks['mono_depth'](features)

        sigmoids = [output[('output', i)] for i in range(self.num_scales)]
        depths = self.sigmoid_to_depth(sigmoids)

        if intrinsics is not None and self.scale_intrinsics:
            depths = [d * intrinsics[:, 0, 0].view(
                rgb.shape[0], 1, 1, 1) for d in depths]

        return {
            'features': features,
            'depths': depths,
        }
