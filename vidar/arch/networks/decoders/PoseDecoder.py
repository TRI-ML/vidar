# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import torch.nn as nn


class PoseDecoder(nn.Module, ABC):
    """
    Pose decoder network

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, num_ch_enc, num_input_features,
                 num_frames_to_predict_for=None,
                 stride=1, output_multiplier=0.01):
        super().__init__()

        self.num_encoder_channels = num_ch_enc
        self.num_input_features = num_input_features
        self.output_multiplier = output_multiplier

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_output_predictions = num_frames_to_predict_for

        self.convs = {
            'squeeze': nn.Conv2d(self.num_encoder_channels[-1], 256, 1),
            ('pose', 0): nn.Conv2d(num_input_features * 256, 256, 3, stride, 1),
            ('pose', 1): nn.Conv2d(256, 256, 3, stride, 1),
            ('pose', 2): nn.Conv2d(256, 6 * num_frames_to_predict_for, 1),
        }

        self.net = nn.ModuleList(list(self.convs.values()))
        self.relu = nn.ReLU()

    def forward(self, all_features):
        """Network forward pass"""

        last_features = [f[-1] for f in all_features]
        last_features = [self.relu(self.convs['squeeze'](f)) for f in last_features]
        cat_features = torch.cat(last_features, 1)

        for i in range(3):
            cat_features = self.convs[('pose', i)](cat_features)
            if i < 2:
                cat_features = self.relu(cat_features)

        output = self.output_multiplier * \
                 cat_features.mean(3).mean(2).view(-1, self.num_output_predictions, 1, 6)
        return torch.split(output, split_size_or_sections=3, dim=-1)
