# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

from vidar.arch.blocks.depth.SigmoidToInvDepth import SigmoidToInvDepth
from vidar.arch.networks.BaseNet import BaseNet
from vidar.arch.networks.decoders.DepthDecoder import DepthDecoder
from vidar.arch.networks.encoders.ResNetEncoder import ResNetEncoder as ResnetEncoder
from vidar.utils.depth import inv2depth, depth2inv


class FocalDepthResNet(BaseNet, ABC):
    """
    Depth network with focal length normalization

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.networks['encoder'] = ResnetEncoder(cfg.encoder)
        cfg.decoder.num_ch_enc = self.networks['encoder'].num_ch_enc
        self.networks['decoder'] = DepthDecoder(cfg.decoder)
        self.scale_inv_depth = SigmoidToInvDepth(
            min_depth=cfg.min_depth, max_depth=cfg.max_depth)

    def forward(self, rgb, intrinsics, **kwargs):
        """Network forward pass"""

        x = self.networks['encoder'](rgb)
        x = self.networks['decoder'](x)
        inv_depths = [x[('output', i)] for i in range(4)]

        if self.training:
            inv_depths = [self.scale_inv_depth(inv_depth)[0] for inv_depth in inv_depths]
        else:
            inv_depths = [self.scale_inv_depth(inv_depths[0])[0]]

        depths = inv2depth(inv_depths)
        depths = [d * intrinsics[:, 0, 0].view(rgb.shape[0], 1, 1, 1) for d in depths]
        inv_depths = depth2inv(depths)

        return {
            'inv_depths': inv_depths
        }
