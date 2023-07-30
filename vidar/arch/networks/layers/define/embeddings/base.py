# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from vidar.arch.networks.layers.define.embeddings.utils.fourier_position_encoding import \
    PerceiverFourierPositionEncoding
from vidar.utils.types import is_list, is_dict


class BaseEmbeddings(nn.Module):
    """
    Embeddings base class.

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()

        self.to_world = cfg.to_world
        self.downsample = cfg.has('downsample', 1.0)
        self.fourier_encoding_xyz = None if not cfg.has('xyz') else \
            PerceiverFourierPositionEncoding(cfg.xyz, 3)
        self.fourier_encoding_origin = None if not cfg.has('origin') else \
            PerceiverFourierPositionEncoding(cfg.origin, 3)
        self.fourier_encoding_rays = None if not cfg.has('rays') else \
            PerceiverFourierPositionEncoding(cfg.rays, 3)
        self.fourier_encoding_depth = None if not cfg.has('depth') else \
            PerceiverFourierPositionEncoding(cfg.depth, 1)
        self.fourier_encoding_time = None if not cfg.has('time') else \
            PerceiverFourierPositionEncoding(cfg.time, 1)
        self.fourier_encoding_reality = None if not cfg.has('reality') else \
            PerceiverFourierPositionEncoding(cfg.reality, 1)

    @property
    def channels(self):
        """ Returns the number of channels which depends on encoding type. """
        num_channels = 0
        if self.fourier_encoding_xyz is not None:
            num_channels += self.fourier_encoding_xyz.channels
        if self.fourier_encoding_origin is not None:
            num_channels += self.fourier_encoding_origin.channels
        if self.fourier_encoding_rays is not None:
            num_channels += self.fourier_encoding_rays.channels
        if self.fourier_encoding_depth is not None:
            num_channels += self.fourier_encoding_depth.channels
        if self.fourier_encoding_time is not None:
            num_channels += self.fourier_encoding_time.channels
        if self.fourier_encoding_reality is not None:
            num_channels += self.fourier_encoding_reality.channels
        return num_channels

