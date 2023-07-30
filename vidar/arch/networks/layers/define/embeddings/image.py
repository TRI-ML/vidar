# Copyright 2023 Toyota Research Institute.  All rights reserved.

from functools import partial

import torch
import torch.nn as nn

from vidar.arch.networks.encoders.ResNetEncoder import ResNetEncoder
from vidar.utils.config import Config
from vidar.utils.networks import freeze_layers_and_norms
from vidar.utils.tensor import interpolate
from vidar.utils.setup import load_checkpoint


class DownSampleRGB(nn.Module):
    """
    Learned image downsampling.

    Parameters
    ----------
    out_dim : int
        Number of output channels.
    """
    def __init__(self, out_dim):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, out_dim, kernel_size=7, stride=2, padding=3)
        self.norm = torch.nn.BatchNorm2d(out_dim)
        self.actv = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.actv(x)
        x = self.pool(x)
        return x


class ImageEmbeddings(nn.Module):
    """
    Image Embeddings class.

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()
        self.type = cfg.type
        self.use_geometric_embeddings = cfg.has('use_geometric_embeddings', False)
        self.downsample = None
        if self.type == 'convnet':
            self.dim = cfg.dim
            self.encoder = DownSampleRGB(out_dim=self.dim)
        elif self.type.startswith('resnet18'):
            self.dim = {
                'resnet18-2': 64,
                'resnet18_all-2': 1024,
                'resnet18-4': 64,
                'resnet18_all-4': 960,
            }[self.type]
            self.encoder = ResNetEncoder(Config(version=18, pretrained=True, num_rgb_in=1))
        elif self.type.startswith('resnet50'):
            self.dim = {
                'resnet50-2': 64,
                'resnet50_all-2': 1856,
                'resnet50-4': 256,
                'resnet50_all-4': 1792,
            }[self.type]
            self.encoder = ResNetEncoder(Config(version=50, pretrained=True, num_rgb_in=1))
        elif self.type == 'rgb':
            self.dim = 3
            self.encoder = None
        elif self.type == 'pixel_nerf':
            self.dim = 512
            from vidar.arch.networks.layers.define.embeddings.utils.spatial_encoder import SpatialEncoder
            self.encoder = SpatialEncoder(backbone='resnet34', pretrained=True, num_layers=4)
        elif self.type.startswith('dpt'):
            from vidar.arch.networks.depth.OmniDataNet import OmniDataNet
            backbone = self.type.split('-')[-1]
            self.downsample = int(self.type.split('-')[1])
            self.dim = {
                'vitb_rn50_384': {4: 768, 16: 768},
                'vitl16_384': {4: 768, 16: 1024},
            }[backbone][self.downsample]
            cfg2 = Config(
                task='depth',
                backbone=backbone,
                transform=False,
                normalize=True,
                scale_factor=100,
            )
            if self.use_geometric_embeddings:
                cfg2.embeddings = Config(
                    camera=Config(
                        to_world=False,
                        rays=Config(
                            num_bands=16,
                            max_resolution=64,
                        )
                    )
                )
            self.encoder = OmniDataNet(cfg2)
            if cfg.has('checkpoint'):
                load_checkpoint(self.encoder, cfg.checkpoint, strict=False, verbose=True,
                                remove_prefixes=(), replaces=[['networks.depth.', '']])
        elif self.type.startswith('vit'):
            from vidar.arch.networks.layers.vit.models_mae import mae_vit_base_patch16
            backbone = self.type.split('-')[-1]
            self.dim = {
                'base_patch16': 768,
            }[backbone]
            self.encoder = mae_vit_base_patch16(checkpoint=cfg.has('checkpoint', None))
        elif self.type.startswith('mim'):
            from vidar.arch.networks.depth.MIMDepthNet import MIMDepthNet
            backbone = self.type.split('-')[-1]
            self.dim = 1024
            self.encoder = MIMDepthNet(Config())
        elif self.type.startswith('vpd'):
            from vidar.arch.networks.depth.VPDNet import VPDNet
            self.dim = 2240 if self.type.endswith('multi') else 1536
            self.encoder = VPDNet(cfg.model)
        else:
            raise ValueError('Invalid image embeddings type')
        self.interpolate = partial(interpolate, scale_factor=None, mode='bilinear')
        self.freeze = cfg.has('freeze', False)
        self.return_depth = cfg.has('return_depth', False)

    @property
    def channels(self):
        """ Returns the number of channels."""
        return self.dim

    def extract(self, rgb, intrinsics=None):
        """Extract image features given the encodey type specified."""
        if self.training:
            freeze_layers_and_norms(self.encoder, flag_freeze=self.freeze)
        if self.type == 'convnet':
            return {
                'feats': self.encoder(rgb)
            }
        elif self.type in ['resnet18-2', 'resnet18-4', 'resnet50-2', 'resnet50-4']:
            feats_list = self.encoder(rgb)
            if self.type.endswith('2'):
                feats = feats_list[0]
            elif self.type.endswith('4'):
                feats = feats_list[1]
            else:
                raise ValueError('Invalid image encoder type')
            return {
                'feats_list': feats_list,
                'feats': feats,
            }
        elif self.type in ['resnet18_all-2', 'resnet18_all-4', 'resnet50_all-2', 'resnet50_all-4']:
            feats_list = self.encoder(rgb)
            if self.type == 'resnet18_all-2':
                feats = feats_list
            elif self.type == 'resnet18_all-4':
                feats = feats_list[1:]
            elif self.type == 'resnet50_all-2':
                feats = feats_list[:-1]
            elif self.type == 'resnet50_all-4':
                feats = feats_list[1:-1]
            else:
                raise ValueError('Invalid image encoder type')
            for i in range(1, len(feats)):
                feats[i] = self.interpolate(feats[i], size=feats[0])
            feats = torch.cat(feats, 1)
            return {
                'feats_list': feats_list,
                'feats': feats,
            }
        elif self.type == 'rgb':
            return {
                'feats': rgb
            }
        elif self.type == 'pixel_nerf':
            return {
                'feats': self.encoder(rgb)
            }
        elif self.type.startswith('dpt'):
            if not self.return_depth:
                feats_list = self.encoder(rgb, intrinsics, only_features=True)['depths'][0]
                if self.downsample == 4:
                    feats = feats_list[:2]
                    for i in range(1, len(feats)):
                        feats[i] = self.interpolate(feats[i], size=feats[0])
                    feats = torch.cat(feats, 1)
                elif self.downsample == 16:
                    feats = feats_list[2]
                else:
                    raise ValueError('Invalid downsample for DPT')
                return {
                    'feats': feats,
                }
            else:
                output = self.encoder(rgb, intrinsics, return_features=True)['depths'][0]
                depth, feats_list = output[0], output[1][0]
                if self.downsample == 4:
                    feats = feats_list[:2]
                    for i in range(1, len(feats)):
                        feats[i] = self.interpolate(feats[i], size=feats[0])
                    feats = torch.cat(feats, 1)
                elif self.downsample == 16:
                    feats = feats_list[2]
                else:
                    raise ValueError('Invalid downsample for DPT')
                return {
                    'feats': feats,
                    'depth_image': depth,
                }
        elif self.type.startswith('vit'):
            feats = self.encoder.features(rgb)
            return {
                'feats': feats,
            }
        elif self.type.startswith('mim'):
            if not self.return_depth:
                output = self.encoder(rgb, features_only=True)
                return {
                    'feats': output['features'][0],
                }
            else:
                output = self.encoder(rgb)
                return {
                    'feats': output['features'][0],
                    'depth_image': output['depths'][0],
                }
        elif self.type.startswith('vpd'):
            if not self.return_depth:
                output = self.encoder(rgb)
                key = 'unet_features' if self.type.endswith('multi') else 'features'
                return {
                    'feats': output[key],
                }
            else:
                output = self.encoder(rgb)
                key = 'unet_features' if self.type.endswith('multi') else 'features'
                return {
                    'feats': output[key],
                    'depth_image': output['depths'][0],
                }            
        else:
            raise ValueError('Invalid image embedding')

    def forward(self, rgb_dict, cam_dict=None):
        batch_list = [rgb.shape[0] for rgb in rgb_dict.values()]
        rgb_list = torch.cat([val for val in rgb_dict.values()], 0)
        cam_list = torch.cat([val.K[:, :3, :3] for val in cam_dict.values()], 0) if cam_dict is not None else None
        output_feats = self.extract(rgb_list, cam_list)

        output_feat = {key: val for key, val in
                       zip(rgb_dict.keys(), torch.split(output_feats['feats'], batch_list))}
        output = {key: {} for key in output_feat.keys()}


        if 'feats_list' in output_feats:
            n = len(output_feats['feats_list'])
            addeds = [{key: val for key, val in
                zip(rgb_dict.keys(), torch.split(output_feats['feats_list'][i], batch_list))}  for i in range(n)]
            for key in output_feat.keys():
                output[key]['feats_list'] = [added[key] for added in addeds]


        for key in output_feat.keys():
            output[key]['feats'] = output_feat[key]
        if self.return_depth:
            output_depth = {key: val for key, val in
                            zip(rgb_dict.keys(), torch.split(output_feats['depth_image'], batch_list))}
            for key in output_feat.keys():
                output[key]['depth_image'] = [output_depth[key]]
        return output
