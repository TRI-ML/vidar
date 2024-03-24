import torch
import torch.nn as nn

from knk_vision.vidar.vidar.arch.networks.layers.define.decoders.utils.conv_decoder import ConvDecoder
from knk_vision.vidar.vidar.arch.networks.layers.define.decoders.utils.upsample_tensor import upsample_tensor
from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.decoder import PerceiverBasicDecoder
from knk_vision.vidar.vidar.utils.config import Config
from knk_vision.vidar.vidar.utils.types import is_list, is_dict
from knk_vision.vidar.vidar.utils.networks import freeze_layers_and_norms
from knk_vision.vidar.vidar.utils.data import get_from_dict


def cat_dict_list(data):
    return {key: torch.cat([val[key] for val in data], 1) for key in data[0].keys()}


class BaseDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        if cfg.upsample_value != 1:
            self.upsample_mode = cfg.upsample_mode
            self.upsample_value = cfg.upsample_value
            if self.upsample_mode == 'convex':
                output_num_channels_mask = 9 * cfg.upsample_value ** 2
                self.decoder_mask = PerceiverBasicDecoder(
                    cfg, output_num_channels=output_num_channels_mask)
            elif self.upsample_mode == 'decode':
                self.decoder_output = ConvDecoder(Config(
                    num_ch_enc=128, num_ch_dec=[128, 64, 32], num_ch_out=cfg.output_num_channels))
                cfg.output_num_channels = self.decoder_output.num_ch_enc
        else:
            self.upsample_mode = None

        self.output_num_channels = cfg.output_num_channels
        self.max_queries = cfg.has('max_queries', 100000)

        self.freeze = cfg.has('freeze', False)
        self.detach_latent = cfg.has('detach_latent', False)
        self.use_prev_pred = cfg.has('use_prev_pred', False)
        self.use_previous = cfg.has('use_previous', None)

        self.multi_decoder = cfg.has('multi_decoder', False)
        if not self.multi_decoder:
            self.decoder = PerceiverBasicDecoder(cfg)
        else:
            self.decoder = nn.ModuleList()
            for _ in range(self.multi_decoder):
                self.decoder.append(PerceiverBasicDecoder(cfg))
            self.decoder_map = {}

    def pre_process(self, pred, info, previous):
        return pred

    def process(self, pred, info, previous):
        return pred

    def post_process(self, pred, info, previous):
        return pred

    def upsample_before(self, pred, query, z, shape):
        if self.upsample_mode == 'decode':
            pred = self.decoder_output(pred)
        return pred

    def upsample_after(self, pred, query, z, shape):
        if self.upsample_mode == 'convex':
            mask = self.decoder_mask(query, z)['predictions']
            if shape is not None:
                mask = mask.reshape([mask.shape[0]] + list(shape) + [mask.shape[-1]]).permute(0, 3, 1, 2)
            pred = upsample_tensor(pred, mask, up=self.upsample_value)
        return pred

    @staticmethod
    def reshape(pred, info):
        cam = get_from_dict(info, 'cam_scaled')
        shape = cam.hw if cam is not None else None
        if cam is not None:
            if not is_dict(pred):
                pred = pred.reshape([pred.shape[0]] + list(shape) + [pred.shape[-1]]).permute(0, 3, 1, 2)
            else:
                for key, val in pred.items():
                    if is_dict(val):
                        for key2, val2 in val.items():
                            if val2.dim() == 3:
                                pred[key][key2] = val2.reshape(
                                    [val2.shape[0]] + list(shape) + list(val2.shape[-1:])).permute(0, 3, 1, 2)
                            elif val2.dim() == 4:
                                pred[key][key2] = val2.reshape(
                                    [val2.shape[0]] + list(shape) + list(val2.shape[-2:])).permute(0, 3, 4, 1, 2)
                    else:
                        if val.dim() == 3:
                            pred[key] = val.reshape(
                                [val.shape[0]] + list(shape) + list(val.shape[-1:])).permute(0, 3, 1, 2)
                        elif val.dim() == 4:
                            pred[key] = val.reshape(
                                [val.shape[0]] + list(shape) + list(val.shape[-2:])).permute(0, 3, 4, 1, 2)
        return pred, shape

    def forward(self, query, z, key, encode_data=None, decode_data=None,
                info=None, previous=None, extra=None, scene=None):

        # If latent is a dict, take the respective timestep
        z = z[key[0]] if is_dict(z) else z

        if self.detach_latent:
            z = z.detach()
        
        # Reshape if input data is 3D
        b = m = n = k = d = None

        is_3D = query.dim() == 4
        if is_3D:
            b, n, k, d = query.shape
            query = query.view(b, n * k, d)
            if extra is not None:
                extra = extra.view(b, n * k, -1)

        is_4D = query.dim() == 5
        if is_4D:
            b, m, n, k, d = query.shape
            query = query.view(b, n * m * k, d)
            if extra is not None:
                extra = extra.view(b, n * m * k, -1)

        # Freeze decoder if requested
        if self.training:
            freeze_layers_and_norms(self.decoder, flag_freeze=self.freeze)

        if not self.multi_decoder:
            decoder = self.decoder
        else:
            scene = scene[0]
            if scene not in self.decoder_map:
                self.decoder_map[scene] = len(self.decoder_map)
            decoder = self.decoder[self.decoder_map[scene]]

        is_grid = z.dim() == 5
        if is_grid:
            s1, s2 = z.shape[1:3]
            b, q, _ = query.shape
            z = z.permute(0, 3, 4, 1, 2).reshape(-1, s1, s2)
            query = query.reshape(b * q, 1, -1)

        shape = None if 'cam_scaled' not in info[key] else info[key]['cam_scaled'].hw

        # Decode queries
        if not is_grid:
            s, t = self.max_queries, query.shape[1]
            steps = t // s + 1
            cross_outputs = []
            for i in range(0, steps):
                st, fn = s * i, min(t, s * (i + 1))
                cross_outputs.append(decoder(
                    query[:, st:fn], z, shape,
                    extra=extra if extra is None else extra[:, st:fn])
                )
        else:
            s, t = self.max_queries, query.shape[0]
            steps = t // s + 1
            cross_outputs = []
            for i in range(0, steps):
                st, fn = s * i, min(t, s * (i + 1))
                cross_outputs.append(decoder(
                    query[st:fn], z[st:fn], shape,
                    extra=extra if extra is None else extra[:, st:fn])
                )
            for i in range(len(cross_outputs)):
                cross_outputs[i]['predictions'] = cross_outputs[i]['predictions'].permute(1, 0, 2)

        cross_output = {'predictions': torch.cat([val['predictions'] for val in cross_outputs], 1)}
        pred = cross_output['predictions']

        # Return to 3D if needed
        pred = pred.view(b, n, k, -1) if is_3D else pred
        pred = pred.view(b, m, n, k, -1) if is_4D else pred

        # prev_key = 'volumetric_1'
        # if self.use_prev_pred and previous is not None and prev_key in previous.keys():
        #
        #     b, n, d, _ = pred.shape
        #     prev_pred = previous[prev_key][key][0]['raw_pred']
        #     all_pred = torch.cat([pred, prev_pred], -2)
        #
        #     zvals = info[key]['zvals']
        #     prev_zvals = previous[prev_key][key][0]['zvals']
        #     prev_zvals = prev_zvals.view(b, -1, n).permute(0, 2, 1)
        #     all_zvals = torch.cat([zvals, prev_zvals], -1)
        #
        #     idx = torch.argsort(all_zvals, -1).unsqueeze(-1)
        #     sorted_pred = torch.gather(all_pred, 2, idx.repeat(1, 1, 1, 4))
        #     sorted_zvals = torch.gather(all_zvals, 2, idx.squeeze(-1))
        #
        #     pred = sorted_pred
        #     info[key]['zvals'] = sorted_zvals

        if previous is not None:
            previous = previous['info'][key]

        # Get embeddings info
        info = info[key]
        info['encode_data'] = encode_data
        info['decode_data'] = decode_data
        info['key'] = key

        pred = self.pre_process(pred, info, previous)
        pred, shape = self.reshape(pred, info)
        pred = self.upsample_before(pred, query, z, shape)
        pred = [self.process(p, info, previous) for p in pred] \
            if is_list(pred) else self.process(pred, info, previous)
        pred = self.upsample_after(pred, query, z, shape)
        pred = self.post_process(pred, info, previous)

        for key in ['raw', 'weights']:
            if key in pred:
                info[key] = pred[key]

        # Return predictions

        return {
            'predictions': pred,
            'cross_output': cross_output,
        }




