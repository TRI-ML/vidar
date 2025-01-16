# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

from knk_vision.vidar.vidar.arch.networks.BaseNet import BaseNet
from knk_vision.vidar.vidar.arch.networks.decoders.DepthDecoder import DepthDecoder
from knk_vision.vidar.vidar.arch.networks.encoders.MultiResNetEncoder import MultiResNetEncoder
# from knk_vision.vidar.vidar.arch.networks.encoders.MultiResNetEncoderStereoTwin import ResnetEncoderMatchingStereo
from knk_vision.vidar.vidar.utils.config import cfg_has


class MultiDepthNet(BaseNet, ABC):
    """
    Multi-frame monocular depth network

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.num_scales = cfg_has(cfg, 'num_scales', 4)
        self.set_attr(cfg, 'scale_intrinsics', False)

        self.networks['encoder'] = MultiResNetEncoder(cfg.encoder)
        cfg.decoder.num_ch_enc = self.networks['encoder'].num_ch_enc
        self.networks['depth'] = DepthDecoder(cfg.decoder)

    def forward(self, rgb, rgb_context, cams,
                intrinsics=None, networks=None):
        """Network forward pass"""

        encoder_output = self.networks['encoder'](
            rgb, rgb_context, cams, networks=networks)

        network_output = {
            **encoder_output,
        }

        output = self.networks['depth'](encoder_output['features'])
        sigmoids = [output[('output', i)] for i in range(self.num_scales)]
        network_output['depths'] = self.sigmoid_to_depth(sigmoids)

        if intrinsics is not None and self.scale_intrinsics:
            network_output['depths'] = [d * intrinsics[0][:, 0, 0].view(
                rgb.shape[0], 1, 1, 1) for d in network_output['depths']]

        return network_output
