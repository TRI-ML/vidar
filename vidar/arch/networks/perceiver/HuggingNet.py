from collections import OrderedDict

import torch
import torch.nn as nn
import abc
from transformers import PerceiverModel, PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import build_position_encoding

from vidar.arch.networks.perceiver.externals.modeling_perceiver import PerceiverDepthDecoder, PerceiverRGBDecoder, build_position_encoding
from vidar.arch.blocks.depth.SigmoidToInvDepth import SigmoidToInvDepth
from vidar.arch.networks.decoders.DepthDecoder import DepthDecoder
from vidar.arch.networks.encoders.ResNetEncoder import ResNetEncoder
from vidar.utils.config import Config
from vidar.utils.networks import freeze_layers_and_norms
from vidar.utils.tensor import interpolate
from vidar.utils.types import is_int


class DownSampleRGB(nn.Module):
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


class HuggingNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tasks = cfg.tasks
        self.to_world = cfg.to_world
        self.depth_range = cfg.depth_range
        self.rgb_feat_dim = cfg.rgb_feat_dim
        self.rgb_feat_type = cfg.rgb_feat_type
        self.encoder_with_rgb = cfg.encoder_with_rgb
        self.decoder_with_rgb = cfg.decoder_with_rgb
        self.output_mode = cfg.output_mode
        self.sample_encoding_rays = cfg.sample_encoding_rays
        self.with_monodepth = cfg.with_monodepth

        self.upsample_convex = cfg.upsample_convex
        self.downsample_encoder = cfg.downsample_encoder
        self.downsample_decoder = cfg.downsample_decoder

        self.image_shape = [s // self.downsample_encoder for s in cfg.image_shape]

        self.fourier_encoding_orig, _ = build_position_encoding(
            position_encoding_type='fourier',
            fourier_position_encoding_kwargs={
                'num_bands': cfg.num_bands_orig,
                'max_resolution': [cfg.max_resolution_orig] * 3,
                'concat_pos': True,
                'sine_only': False,
            }
        )

        self.fourier_encoding_dirs, _ = build_position_encoding(
            position_encoding_type='fourier',
            fourier_position_encoding_kwargs={
                'num_bands': cfg.num_bands_dirs,
                'max_resolution': [cfg.num_bands_dirs] * 3,
                'concat_pos': True,
                'sine_only': False,
            }
        )

        tot_encoder = self.fourier_encoding_orig.output_size() + \
                      self.fourier_encoding_dirs.output_size()
        if self.encoder_with_rgb:
            tot_encoder += self.rgb_feat_dim

        tot_decoder = self.fourier_encoding_orig.output_size() + \
                      self.fourier_encoding_dirs.output_size()
        if self.decoder_with_rgb:
            tot_decoder += self.rgb_feat_dim

        tot_decoder_depth = tot_decoder
        tot_decoder_rgb = tot_decoder

        self.config = PerceiverConfig(
            train_size=self.image_shape,
            d_latents=cfg.d_latents,
            d_model=tot_encoder,
            num_latents=cfg.num_latents,
            hidden_act='gelu',
            hidden_dropout_prob=cfg.hidden_dropout_prob,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            num_blocks=1,
            num_cross_attention_heads=cfg.num_cross_attention_heads,
            num_self_attends_per_block=cfg.num_self_attends_per_block,
            num_self_attention_heads=cfg.num_self_attention_heads,
            qk_channels=None,
            v_channels=None,
        )

        if 'depth' in self.tasks:
            self.decoder = PerceiverDepthDecoder(
                self.config,
                num_channels=tot_decoder_depth,
                use_query_residual=False,
                output_num_channels=1,
                position_encoding_type="none",
                min_depth=self.depth_range[0],
                max_depth=self.depth_range[1],
                num_heads=cfg.decoder_num_heads,
                upsample_mode=cfg.upsample_convex,
                upsample_value=cfg.downsample_decoder,
                output_mode=cfg.output_mode
            )
        if 'rgb' in self.tasks:
            self.decoder_rgb = PerceiverRGBDecoder(
                self.config,
                num_channels=tot_decoder_rgb,
                use_query_residual=False,
                output_num_channels=3,
                position_encoding_type="none",
                num_heads=cfg.decoder_num_heads,
                upsample_mode=cfg.upsample_convex,
                upsample_value=cfg.downsample_decoder,
            )

        self.model = PerceiverModel(
            self.config,
        )

        if self.rgb_feat_type == 'convnet':
            self.feature = DownSampleRGB(out_dim=self.rgb_feat_dim)
        elif self.rgb_feat_type in ['resnet', 'resnet_all', 'resnet_all_rgb']:
            self.feature = ResNetEncoder(Config(version=18, pretrained=True, num_rgb_in=1))

        if self.with_monodepth:
            self.mono_encoder = ResNetEncoder(Config(version=18, pretrained=True, num_rgb_in=1))
            self.mono_decoder = DepthDecoder(Config(
                num_scales=4, use_skips=True, num_ch_enc=self.feature.num_ch_enc,
                num_ch_out=1, activation='sigmoid',
            ))
            self.sigmoid_to_depth = SigmoidToInvDepth(
                min_depth=self.depth_range[0], max_depth=self.depth_range[1], return_depth=True)

    def get_rgb_feat(self, rgb):
        if self.rgb_feat_type == 'convnet':
            return {
                'feat': self.feature(rgb)
            }
        elif self.rgb_feat_type == 'resnet':
            return {
                'feat': self.feature(rgb)[1]
            }
        elif self.rgb_feat_type.startswith('resnet_all'):
            all_feats = self.feature(rgb)
            feats = all_feats[1:]
            for i in range(1, len(feats)):
                feats[i] = interpolate(
                    feats[i], size=feats[0], scale_factor=None, mode='bilinear', align_corners=True)
            if self.rgb_feat_type.endswith('rgb'):
                feats = feats + [interpolate(
                    rgb, size=feats[0], scale_factor=None, mode='bilinear', align_corners=True)]
            feat = torch.cat(feats, 1)
            return {
                'all_feats': all_feats,
                'feat': feat
            }

    def run_monodepth(self, rgb, freeze):
        freeze_layers_and_norms(self.mono_encoder, flag_freeze=freeze)
        freeze_layers_and_norms(self.mono_decoder, flag_freeze=freeze)
        mono_features = self.mono_encoder(rgb)
        mono_output = self.mono_decoder(mono_features)
        sigmoids = [mono_output[('output', i)] for i in range(1)]
        return self.sigmoid_to_depth(sigmoids)[0]

    def embeddings(self, data, sources, downsample):

        if 'rgb' in sources:
            assert 'rgb' in data[0].keys()
            b = [datum['rgb'].shape[0] for datum in data]
            rgb = torch.cat([datum['rgb'] for datum in data], 0)
            output_feats = self.get_rgb_feat(rgb)
            feats = torch.split(output_feats['feat'], b)
            for i in range(len(data)):
                data[i]['feat'] = feats[i]

            if self.with_monodepth:
                depth = self.run_monodepth(rgb, freeze=False)
                depth = torch.split(depth, b)
                for i in range(len(data)):
                    data[i]['depth_mono'] = depth[i]

        encodings = []
        for datum in data:

            encoding = OrderedDict()

            if 'cam' in sources:
                assert 'cam' in data[0].keys()

                cam = datum['cam'].scaled(1. / downsample)
                orig = cam.get_origin(flatten=True)

                if self.to_world:
                    dirs = cam.get_viewdirs(normalize=True, flatten=True, to_world=True)
                else:
                    dirs = cam.no_translation().get_viewdirs(normalize=True, flatten=True, to_world=True)

                orig_encodings = self.fourier_encoding_orig(
                    index_dims=None, pos=orig, batch_size=orig.shape[0], device=orig.device)
                dirs_encodings = self.fourier_encoding_dirs(
                    index_dims=None, pos=dirs, batch_size=dirs.shape[0], device=dirs.device)

                encoding['cam'] = torch.cat([orig_encodings, dirs_encodings], -1)

            if 'rgb' in sources:
                rgb = datum['feat']
                rgb_flat = rgb.view(*rgb.shape[:-2], -1).permute(0, 2, 1)
                encoding['rgb'] = rgb_flat

            encoding['all'] = torch.cat([val for val in encoding.values()], -1)
            encodings.append(encoding)

        return encodings

    @staticmethod
    def sample_decoder(data, embeddings, field, sample_queries, filter_invalid):

        query_idx = []

        if filter_invalid:
            tot_min = []

            for i in range(len(embeddings)):
                for b in range(data[i]['rgb'].shape[0]):
                    tot_min.append((data[i]['rgb'][b].mean(0) >= 0).sum())
            tot_min = min(tot_min)

            tot = embeddings[0][field][0].shape[0]
            tot = int(sample_queries * tot)
            tot = min([tot, tot_min])

        for i in range(len(embeddings)):
            idx = []

            for b in range(data[i]['rgb'].shape[0]):
                if filter_invalid:

                    valid = data[i]['rgb'][b].mean(0, keepdim=True) >= 0
                    valid = valid.view(1, -1).permute(1, 0)

                    num = embeddings[i][field][0].shape[0]
                    all_idx = torch.arange(num, device=valid.device).unsqueeze(1)
                    valid_idx = all_idx[valid]

                    num = valid_idx.shape[0]
                    idx_i = torch.randperm(num)[tot:]
                    valid[valid_idx[idx_i]] = 0
                    idx_i = all_idx[valid]

                else:

                    num = embeddings[i][field][0].shape[0]
                    tot = int(sample_queries * num)
                    idx_i = torch.randperm(num)[:tot]

                idx.append(idx_i)

            idx = torch.stack(idx, 0)
            embeddings[i][field] = torch.stack(
                [embeddings[i][field][b][idx[b]] for b in range(idx.shape[0])], 0)

            query_idx.append(idx)

        return query_idx, embeddings

    def forward(self, encode_data, decode_data=None,
                sample_queries=0, filter_invalid=False):

        encode_field = 'all' if self.encoder_with_rgb else 'cam'
        decode_field = 'all' if self.decoder_with_rgb else 'cam'

        encode_sources = ['rgb', 'cam']
        decode_sources = ['cam']

        shape = encode_data[0]['cam'].hw

        output = {}

        encode_dict = self.encode(
            data=encode_data, field=encode_field, sources=encode_sources
        )

        if 'depth_mono' in encode_data[0].keys():
            output['depth_mono'] = [datum['depth_mono'] for datum in encode_data]

        decode_embeddings = encode_dict['embeddings'] if decode_data is None else None

        decode_dict = self.decode(
            latent=encode_dict['latent'], shape=shape,
            data=decode_data, embeddings=decode_embeddings,
            field=decode_field, sources=decode_sources,
            sample_queries=sample_queries, filter_invalid=filter_invalid
        )

        output.update(decode_dict['output'])

        return {
            'output': output,
            'encode_embeddings': encode_dict['embeddings'],
            'decode_embeddings': decode_dict['embeddings'],
            'latent': encode_dict['latent'],
        }

    def encode(self, field, sources, data=None, embeddings=None):
        assert data is not None or embeddings is not None
        assert data is None or embeddings is None

        if embeddings is None:
            embeddings = self.embeddings(data, sources=sources, downsample=self.downsample_encoder)

        all_embeddings = torch.cat([emb[field] for emb in embeddings], 1)

        if self.training and self.sample_encoding_rays > 0:
            tot = self.sample_encoding_rays if is_int(self.sample_encoding_rays) \
                else int(self.sample_encoding_rays * all_embeddings.shape[1])
            all_embeddings = torch.stack([all_embeddings[i, torch.randperm(all_embeddings.shape[1])[:tot], :]
                                          for i in range(all_embeddings.shape[0])], 0)

        return {
            'embeddings': embeddings,
            'latent': self.model(inputs=all_embeddings).last_hidden_state,
        }

    def decode(self, latent, field, sources=None, data=None, embeddings=None, shape=None,
               sample_queries=0, filter_invalid=False):
        assert data is not None or embeddings is not None
        assert data is None or embeddings is None

        if embeddings is None:
            shape = data[0]['cam'].hw
            shape = [s // self.downsample_decoder for s in shape]
            embeddings = self.embeddings(data, sources=sources, downsample=self.downsample_decoder)

        output = {}

        if self.training and (sample_queries > 0):  #  or filter_invalid):
            output['query_idx'], embeddings = self.sample_decoder(
                data, embeddings, field, sample_queries, filter_invalid)
            shape = None

        if 'rgb' in self.tasks:
            output['rgb'] = [
                self.decoder_rgb(query=emb[field], z=latent, shape=shape).logits
                for emb in embeddings]

        if 'depth' in self.tasks:
            output['depth'] = [
                self.decoder(query=emb[field], z=latent, shape=shape).logits
                for emb in embeddings]

        return {
            'embeddings': embeddings,
            'output': output,
        }
