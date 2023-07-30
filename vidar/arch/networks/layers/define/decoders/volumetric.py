# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch

from vidar.arch.networks.layers.define.decoders.utils.base_decoder import BaseDecoder
from vidar.utils.nerf import composite, composite_depth, composite_weights


class Exponential(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x)


class VolumetricDecoder(BaseDecoder):
    """
    Perceiver IO volumetric decoder

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """  
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rgb_activation = torch.nn.Sigmoid()
        self.sigma_activation = {
            'sigmoid': torch.nn.Sigmoid(),
            'relu': torch.nn.ReLU(),
            'exp': Exponential(),
        }[cfg.sigma_activation]
        self.white_background = cfg.has('white_background', False)
        self.task = cfg.has('task', 'rgb_depth')

    def composite_rgb_depth(self, raw, zvals, sigma_eps=1e-6):
        """Composite output into rgb and depth maps"""
        rgbs = self.rgb_activation(raw[..., :3])
        sigmas = self.sigma_activation(raw[..., [3]]) + sigma_eps
        return composite(rgbs, sigmas, zvals)

    def composite_depth(self, raw, zvals, sigma_eps=1e-6):
        """Composite output into depth maps"""
        sigmas = self.sigma_activation(raw[..., [0]]) + sigma_eps
        return composite_depth(sigmas, zvals)

    def composite_rgb(self, raw, weights):
        """Composite output into rgb"""
        rgbs = self.rgb_activation(raw[..., :3])
        rgb = composite_weights(rgbs, weights)
        return {'rgb': rgb, 'weights': weights}

    def composite_scnflow(self, raw, weights, key):
        """Composite output into scene flow"""
        scnflows = raw[..., :6]
        scnflow = composite_weights(scnflows, weights)
        return {'scnflow': {
            (key[0] - 1, key[1]): scnflow[..., :3],
            (key[0] + 1, key[1]): scnflow[..., 3:],
        }, 'weights': weights}

    def pre_process(self, raw, info, previous):
        """Process the output of the decoder"""
        if self.task == 'rgb_depth':
            zvals = info['zvals']
            output = self.composite_rgb_depth(raw, zvals)
        elif self.task == 'depth':
            zvals = info['zvals']
            output = self.composite_depth(raw, zvals)
        elif self.task == 'rgb':
            b, n, d, c = raw.shape
            weights = previous['weights'].view(b, d, 1, -1).permute(0, 3, 1, 2)
            output = self.composite_rgb(raw, weights)
        elif self.task == 'scnflow':
            b, n, d, c = raw.shape
            key = info['key']
            weights = previous['weights'].view(b, d, 1, -1).permute(0, 3, 1, 2)
            output = self.composite_scnflow(raw, weights, key)
        else:
            raise ValueError('Invalid volumetric task')
        return {**output, 'raw': raw}
