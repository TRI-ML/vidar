# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from vidar.utils.config import cfg_has
from vidar.utils.data import flatten
from vidar.utils.depth import depth2inv
from vidar.utils.tensor import grid_sample

RESNET_VERSIONS = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152
}


class MultiResNetEncoder(nn.Module, ABC):
    """
    Multi-frame depth encoder

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()

        self.adaptive_bins = cfg.adaptive_bins
        self.depth_binning = cfg.depth_binning
        self.set_missing_to_max = True

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        self.depth_range = cfg.depth_range
        self.min_depth_bin = cfg.depth_bin_range[0]
        self.max_depth_bin = cfg.depth_bin_range[1]
        self.num_depth_bins = cfg.num_depth_bins

        self.min_depth_bin = torch.nn.Parameter(torch.tensor(
            self.min_depth_bin), requires_grad=False)
        self.max_depth_bin = torch.nn.Parameter(torch.tensor(
            self.max_depth_bin), requires_grad=False)

        self.matching_height = cfg.input_shape[0] // 4
        self.matching_width = cfg.input_shape[1] // 4

        self.depth_bins = None
        self.warp_depths = None

        assert cfg.version in RESNET_VERSIONS, ValueError(
            '{} is not a valid number of resnet layers'.format(cfg.version))
        encoder = RESNET_VERSIONS[cfg.version](cfg.pretrained)

        self.layer0 = nn.Sequential(encoder.conv1,  encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool,  encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        if cfg.version > 34:
            self.num_ch_enc[1:] *= 4

        self.prematching_conv = nn.Sequential(
            nn.Conv2d(64, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.double_volume = cfg_has(cfg, 'double_volume', False)

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(self.num_ch_enc[1] + self.num_depth_bins * (2 if self.double_volume else 1),
                      out_channels=self.num_ch_enc[1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.grid_sample = partial(
            grid_sample, padding_mode='zeros', mode='bilinear', align_corners=True)

        self.volume_masking = cfg_has(cfg, 'volume_masking', False)

    def update_adaptive_depth_bins(self, depth):
        """Change depth bins based on predicted depth"""
        min_depth = depth.detach().min(-1)[0].min(-1)[0]
        max_depth = depth.detach().max(-1)[0].max(-1)[0]

        min_depth = min_depth.mean().cpu().item()
        max_depth = max_depth.mean().cpu().item()

        min_depth = max(self.depth_range[0], min_depth * 0.9)
        max_depth = min(self.depth_range[1], max_depth * 1.1)

        self.min_depth_bin = nn.Parameter(
            self.min_depth_bin * 0.99 + min_depth * 0.01, requires_grad=False)
        self.max_depth_bin = nn.Parameter(
            self.max_depth_bin * 0.99 + max_depth * 0.01, requires_grad=False)

    def compute_depth_bins(self, min_depth_bin, max_depth_bin, device):
        """Compute depth bins based on minimum and maximum values"""
        min_depth_bin = min_depth_bin.cpu()
        max_depth_bin = max_depth_bin.cpu()

        if self.depth_binning == 'inverse':
            self.depth_bins = 1. / np.linspace(
                1. / max_depth_bin, 1. / min_depth_bin, self.num_depth_bins)[::-1]
        elif self.depth_binning == 'linear':
            self.depth_bins = np.linspace(
                min_depth_bin, max_depth_bin, self.num_depth_bins)
        elif self.depth_binning == 'sid':
            self.depth_bins = np.array(
                [np.exp(np.log(min_depth_bin) + np.log(max_depth_bin / min_depth_bin) * i / (self.num_depth_bins - 1))
                 for i in range(self.num_depth_bins)])
        else:
            raise NotImplementedError
        self.depth_bins = torch.from_numpy(self.depth_bins).float().to(device)

        ones = torch.ones((1, self.matching_height, self.matching_width),
                          dtype=torch.float, device=device)
        return torch.stack([depth * ones for depth in self.depth_bins], 1)

    def feature_extraction(self, image, return_all_feats=False):
        """Extract features from input images"""
        image = (image - 0.45) / 0.225
        feats_0 = self.layer0(image)
        feats_1 = self.layer1(feats_0)
        return [feats_0, feats_1] if return_all_feats else feats_1

    def indices_to_inv_depth(self, indices):
        """Convert bin indices to inverse depth values"""
        batch, height, width = indices.shape
        depth = self.depth_bins[indices.reshape(-1)]
        return 1 / depth.reshape((batch, height, width))

    def compute_confidence_mask(self, cost_volume, num_bins_threshold=None):
        """Compute confidence mask based on cost volume"""
        if num_bins_threshold is None:
            num_bins_threshold = self.num_depth_bins
        return ((cost_volume > 0).sum(1) == num_bins_threshold).float()

    def forward(self, rgb, rgb_context=None,
                cams=None, mode='multi', networks=None):
        """Network forward pass"""

        feats = self.feature_extraction(rgb, return_all_feats=True)
        current_feats = feats[-1]

        if mode == 'mono':
            feats.append(self.layer2_mono(feats[-1]))
            feats.append(self.layer3_mono(feats[-1]))
            feats.append(self.layer4_mono(feats[-1]))
            return {
                'features': feats,
            }

        output = {}

        with torch.no_grad():
            if self.warp_depths is None or self.adaptive_bins:
                self.warp_depths = self.compute_depth_bins(
                    self.min_depth_bin, self.max_depth_bin, device=rgb.device)

        b, n, c, h, w = rgb_context.shape
        rgb_context = rgb_context.reshape(b * n, c, h, w)
        feats_context = self.feature_extraction(rgb_context, return_all_feats=True)

        output_transformer = networks['transformer'](rgb, rgb_context, rgb.device, cams[-1].scaled(1/4))
        output['depth_regr'] = [
            output_transformer['depth1_low'],
        ]
        output['depth_regr'] = flatten(output['depth_regr'])

        if 'ssim_lowest_cost' in output_transformer.keys():
            output['lowest_cost_ssim'] = output_transformer['ssim_lowest_cost']

        mask3d = output_transformer['warped_mask']

        mask2d = (mask3d.sum(0) == mask3d.shape[0]).float()
        mask2d[:, :2, :] = 0
        mask2d[:, -2:, :] = 0
        mask2d[:, :, :2] = 0
        mask2d[:, :,-2:] = 0

        output['confidence_mask_transformer'] = mask2d
        output['confidence_mask_transformer3d'] = mask3d
        output['lowest_cost_transformer1'] = depth2inv(output_transformer['depth1_low'])
        output['lowest_cost_transformer2'] = depth2inv(output_transformer['depth2_low'])
        output['cost_volume_transformer'] = output_transformer['attn_weight_softmax'][0].permute(0, 3, 1, 2)

        cost_volume = output['cost_volume_transformer']
        confidence_mask = output['confidence_mask_transformer']
        lowest_cost = output['lowest_cost_transformer1']

        if 'ssim_lowest_cost' in output_transformer:
            output['ssim_lowest_cost'] = output_transformer['ssim_lowest_cost']

        confidence_mask = self.compute_confidence_mask(
            cost_volume.detach() * confidence_mask.detach())
        cost_volume = cost_volume * confidence_mask.unsqueeze(1)

        post_matching_feats = self.reduce_conv(
            torch.cat([current_feats, cost_volume], 1))

        feats.append(self.layer2(post_matching_feats))
        feats.append(self.layer3(feats[-1]))
        feats.append(self.layer4(feats[-1]))

        output.update(**{
            'features': feats,
            'lowest_cost': lowest_cost,
            'confidence_mask': confidence_mask,
            'cost_volume': cost_volume,
        })

        return output
