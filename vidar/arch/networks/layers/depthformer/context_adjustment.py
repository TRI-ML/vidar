# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
from torch import nn
from torch.nn.utils import weight_norm


class ContextAdjustmentLayer(nn.Module):
    """
    Context adjustment layer
    Base on https://github.com/mli0603/stereo-transformer/blob/main/module/context_adjustment_layer.py

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()

        num_blocks = cfg.num_blocks
        feature_dim = cfg.feat_dim
        expansion = cfg.expansion_ratio

        self.num_blocks = num_blocks

        self.in_conv = nn.Conv2d(4, feature_dim, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([ResBlock(feature_dim, expansion) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)

    def forward(self, depth_raw, img):
        """Network forward pass"""

        eps = 1e-6
        mean_depth_pred = depth_raw.mean()
        std_depth_pred = depth_raw.std() + eps
        depth_pred_normalized = (depth_raw - mean_depth_pred) / std_depth_pred

        feat = self.in_conv(torch.cat([depth_pred_normalized, img], dim=1))
        for layer in self.layers:
            feat = layer(feat, depth_pred_normalized)

        depth_res = self.out_conv(feat)
        depth_final = depth_pred_normalized + depth_res

        return depth_final * std_depth_pred + mean_depth_pred


class ResBlock(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0):
        """
        ResNet block

        Parameters
        ----------
        n_feats : Int
            Number of layer features
        expansion_ratio : Int
            Expansion ratio for middle layer
        res_scale : Float
            Scale ratio for residual connections
        """
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats + 1, n_feats * expansion_ratio, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x, depth):
        return x + self.module(torch.cat([depth, x], dim=1)) * self.res_scale
