# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

RESNET_VERSIONS = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152
}

RESNET_WEIGHTS = 'IMAGENET1K_V1'


class ResNetMultiInput(models.ResNet, ABC):
    def __init__(self, block_type, block_channels, num_input_rgb):
        """Multi-input ResNet model

        Parameters
        ----------
        block_type : nn.Module
            Residual block type
        block_channels : int
            Number of channels in each block
        num_input_rgb : int
            Number of input images
        """
        super().__init__(block_type, block_channels)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_rgb * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_type,  64, block_channels[0])
        self.layer2 = self._make_layer(block_type, 128, block_channels[1], stride=2)
        self.layer3 = self._make_layer(block_type, 256, block_channels[2], stride=2)
        self.layer4 = self._make_layer(block_type, 512, block_channels[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multi_input(num_layers, num_input_rgb, pretrained=True):
    """Generates a multi-input ResNet model

    Parameters
    ----------
    num_layers : int
        Number of layers in the ResNet
    num_input_rgb : int
        Number of input images
    pretrained : bool, optional
        True is pre-trained ImageNet weights are used, by default True

    Returns
    -------
    nn.Module
        Multi-input ResNet model
    """
    assert num_layers in [18, 50], 'Can only run with 18 or 50 layer resnet'

    block_channels = {
        18: [2, 2, 2, 2],
        50: [3, 4, 6, 3]
    }[num_layers]

    block_type = {
        18: models.resnet.BasicBlock,
        50: models.resnet.Bottleneck
    }[num_layers]

    model = ResNetMultiInput(block_type, block_channels, num_input_rgb)

    if pretrained:
        loaded = RESNET_VERSIONS[num_layers](weights=RESNET_WEIGHTS).state_dict()
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_rgb, 1) / num_input_rgb
        model.load_state_dict(loaded)

    return model


class ResNetEncoder(nn.Module, ABC):
    """
    Single-frame depth encoder

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.features = []

        assert cfg.version in RESNET_VERSIONS, f'Invalid ResNet version: {cfg.version}'

        if cfg.num_rgb_in > 1:
            self.encoder = resnet_multi_input(
                cfg.version, cfg.num_rgb_in, cfg.pretrained)
        else:
            self.encoder = RESNET_VERSIONS[cfg.version](
                weights=None if not cfg.pretrained else RESNET_WEIGHTS)

        if cfg.version > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        """Network forward pass"""

        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        self.features.clear()
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
