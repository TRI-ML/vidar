# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """Creates a ResNet encoder with different parameters"""
    def __init__(self, name):
        super().__init__()
        if name == 'densenet121':
            self.base_model = models.densenet121(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif name == 'densenet161':
            self.base_model = models.densenet161(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif name == 'resnet50':
            self.base_model = models.resnet50(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif name == 'resnet101':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif name == 'resnext50':
            self.base_model = models.resnext50_32x4d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif name == 'resnext101':
            self.base_model = models.resnext101_32x8d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError('Not supported encoder: {}'.format(name))

    def forward(self, x):
        features, skips = [x], [x]
        for key, val in self.base_model._modules.items():
            if not any(x in key for x in ['fc', 'avgpool']):
                feature = val(features[-1])
                features.append(feature)
                if any(x in key for x in self.feat_names):
                    skips.append(feature)
        return skips
