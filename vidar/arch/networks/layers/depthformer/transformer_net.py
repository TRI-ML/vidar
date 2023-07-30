# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn

from vidar.arch.networks.layers.depthformer.feature_extraction import SppBackbone as Backbone
from vidar.arch.networks.layers.depthformer.regression import RegressionHead
from vidar.arch.networks.layers.depthformer.tokenizer import Tokenizer
from vidar.arch.networks.layers.depthformer.transformer import Transformer


def batched_index_select(source, dim, index):
    views = [source.shape[0]] + [1 if i != dim else -1 for i in range(1, len(source.shape))]
    expanse = list(source.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(source, dim, index)


class TransformerNet(nn.Module):
    """
    Transformer network class, to extract features and predict depth maps for DepthFormer
    
    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg, decoder_type='regression'):
        super().__init__()

        self.backbone = Backbone(cfg)
        self.tokenizer = Tokenizer(cfg)
        self.transformer = Transformer(cfg)

        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth
        self.num_bins = cfg.num_bins

        self.decoder_type = decoder_type

        self.regression_head = RegressionHead(cfg)

        self._reset_parameters()
        self._disable_batchnorm_tracking()
        self._relu_inplace()

    def _reset_parameters(self):
        """Initialize network weights"""
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def _disable_batchnorm_tracking(self):
        """Disable batchnorm tracking"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

    def _relu_inplace(self):
        """Set ReLU to inplace"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.inplace = True

    def fix_layers(self):
        """Fix layers (important to avoid batchnorm issues)"""
        def iterate(module):
            for key in module.keys():
                if type(module[key]) == nn.BatchNorm2d:
                    module[key] = nn.InstanceNorm2d(
                        module[key].num_features,
                        module[key].eps,
                        module[key].momentum,
                        module[key].affine,
                    )
                iterate(module[key]._modules)
        iterate(self._modules)

    def forward(self, target, context, sampled_rows, sampled_cols, cam=None):
        """ Forward method, taking target and context images and returning attention + depth maps"""

        bs, _, h, w = target.size()

        feat_all = self.backbone(target, context)
        feat = [feat_all[0]] + feat_all[2:]

        feat1 = [f[[0]] for f in feat]
        feat2 = [f[[1]] for f in feat]

        feat1 = self.tokenizer(feat1)
        feat2 = self.tokenizer(feat2)

        if sampled_cols is not None:
            feat1 = batched_index_select(feat1, 3, sampled_cols)
            feat2 = batched_index_select(feat2, 3, sampled_cols)
        if sampled_rows is not None:
            feat1 = batched_index_select(feat1, 2, sampled_rows)
            feat2 = batched_index_select(feat2, 2, sampled_rows)

        output_transformer = self.transformer(
            feat1, feat2, cam=cam, min_depth=self.min_depth, max_depth=self.max_depth, num_bins=self.num_bins)

        output_regression = self.regression_head(
            output_transformer['attn_weight'], target, context, sampled_rows, sampled_cols,
            min_depth=self.min_depth, max_depth=self.max_depth, num_bins=self.num_bins,
        )

        return {
            **output_transformer,
            **output_regression,
        }
