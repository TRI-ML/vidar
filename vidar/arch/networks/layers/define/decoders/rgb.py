# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch

from vidar.arch.networks.layers.define.decoders.utils.base_decoder import BaseDecoder


class RGBDecoder(BaseDecoder):
    """
    Perceiver IO image decoder

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sigmoid = torch.nn.Sigmoid()

    def process(self, pred, info, previous):
        """Process the output of the decoder"""
        return {
            'rgb': self.sigmoid(pred),
        }

