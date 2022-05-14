# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch

from vidar.arch.networks.depth.MonoDepthResNet import MonoDepthResNet
from vidar.utils.config import read_config

### Create network

cfg = read_config('demos/run_network/config.yaml')
net = MonoDepthResNet(cfg)

### Create dummy input and run network

rgb = torch.randn((2, 3, 128, 128))
depth = net(rgb=rgb)['depths']
