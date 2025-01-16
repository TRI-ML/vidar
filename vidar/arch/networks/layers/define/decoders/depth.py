# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch

from knk_vision.vidar.vidar.arch.blocks.depth.SigmoidToInvDepth import SigmoidToInvDepth
from knk_vision.vidar.vidar.arch.blocks.depth.SigmoidToLogDepth import SigmoidToLogDepth
from knk_vision.vidar.vidar.arch.networks.layers.define.decoders.utils.base_decoder import BaseDecoder
from knk_vision.vidar.vidar.utils.depth import get_depth_bins


class DepthDecoder(BaseDecoder):
    """
    Perceiver IO depth decoder

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.output_mode = cfg.output_mode
        self.sigmoid = torch.nn.Sigmoid()
        if self.output_mode == 'inv_depth':
            self.sigmoid_to_depth = SigmoidToInvDepth(
                min_depth=cfg.depth_range[0], max_depth=cfg.depth_range[1], return_depth=True)
        elif self.output_mode == 'inv_depth+logvar':
            self.sigmoid_to_depth = SigmoidToInvDepth(
                min_depth=cfg.depth_range[0], max_depth=cfg.depth_range[1], return_depth=True)
        elif self.output_mode == 'log_depth':
            self.sigmoid_to_log_depth = SigmoidToLogDepth()
        elif self.output_mode == 'mixture':
            self.sigmoid_to_depth = SigmoidToInvDepth(
                min_depth=cfg.depth_range[0], max_depth=cfg.depth_range[1], return_depth=True)
        elif self.output_mode == 'bins':
            self.return_training_depth = cfg.has('return_training_depth', False)
            self.sampling_type = cfg.has('sampling_type', 'linear')
            self.distribution = get_depth_bins(
                self.sampling_type, cfg.depth_range[0], cfg.depth_range[1], self.output_num_channels)
        else:
            raise ValueError('Invalid depth output mode')

    def process(self, pred, info, previous):
        """Process the output of the decoder"""
        if self.output_mode == 'inv_depth':
            pred = {
                'depth': self.sigmoid_to_depth(self.sigmoid(pred)),
            }
        elif self.output_mode == 'inv_depth+logvar':
            pred = {
                'depth': self.sigmoid_to_depth(self.sigmoid(pred[:, [0]])),
                'logvar': pred[:, [1]]
            }
        elif self.output_mode == 'log_depth':
            pred = {
                'depth': self.sigmoid_to_log_depth(self.sigmoid(pred))
            }
        elif self.output_mode == 'bins':
            b, c, h, w = pred.shape
            pred = {
                'bins': pred,
                'zvals': self.distribution.to(pred.device),
            }
            if not self.training or (self.training and self.return_training_depth):
                idx = torch.argmax(pred['bins'], dim=1, keepdim=True)
                bins = self.distribution.view(1, -1, 1, 1).repeat(b, 1, h, w).to(pred['bins'].device)
                pred['depth'] = torch.gather(bins, 1, idx)
        elif self.output_mode == 'mixture':
            pred[:, [0]] = self.sigmoid_to_depth(self.sigmoid(pred[:, [0]]))
            pred[:, [1]] = self.sigmoid_to_depth(self.sigmoid(pred[:, [1]]))
            pred[:, [2]] = 10 * self.sigmoid(pred[:, [2]])
            pred[:, [3]] = 10 * self.sigmoid(pred[:, [3]])
            pred[:, [4]] = self.sigmoid(pred[:, [4]])
        return pred

