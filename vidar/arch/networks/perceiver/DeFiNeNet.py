from collections import OrderedDict

import torch
import torch.nn as nn

from vidar.arch.networks.layers.define.decoders.camera import CameraDecoder
from vidar.arch.networks.layers.define.decoders.depth import DepthDecoder
from vidar.arch.networks.layers.define.decoders.multiview import MultiviewDecoder
from vidar.arch.networks.layers.define.decoders.normals import NormalsDecoder
from vidar.arch.networks.layers.define.decoders.rgb import RGBDecoder
from vidar.arch.networks.layers.define.decoders.semantic import SemanticDecoder
from vidar.arch.networks.layers.define.decoders.volumetric import VolumetricDecoder
from vidar.arch.networks.layers.define.embeddings.camera import CameraEmbeddings
from vidar.arch.networks.layers.define.embeddings.image import ImageEmbeddings
from vidar.arch.networks.layers.define.embeddings.multiview import MultiviewEmbeddings
from vidar.arch.networks.layers.define.embeddings.projection import ProjectionEmbeddings
from vidar.arch.networks.layers.define.embeddings.volumetric import VolumetricEmbeddings
from vidar.arch.networks.layers.define.perceiver.model import PerceiverModel
from vidar.utils.data import make_list, str_not_in
from vidar.utils.networks import freeze_layers_and_norms
from vidar.utils.types import is_int, is_dict, is_tensor, is_list


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


class DeFiNeNet(nn.Module):

    DECODER_CLASSES = {
        'rgb': RGBDecoder,
        'depth': DepthDecoder,
        'normals': NormalsDecoder,
        'semantic': SemanticDecoder,
        'camera': CameraDecoder,
        'volumetric': VolumetricDecoder,
        'multiview': MultiviewDecoder,
    }

    EMBEDDING_CLASSES = {
        'image': ImageEmbeddings,
        'camera': CameraEmbeddings,
        'volumetric': VolumetricEmbeddings,
        'multiview': MultiviewEmbeddings,
        'projection': ProjectionEmbeddings,
    }

    def __init__(self, cfg):
        super().__init__()

        # VARIATIONAL

        self.is_variational = cfg.has('variational')
        self.variational_params = None if not self.is_variational else {
            'kld_weight': cfg.variational.kld_weight,
            'encode_vae': cfg.variational.encode_vae,
            'soft_mask_weight': cfg.variational.has('soft_mask_weight', 1.0),
        }

        # Parameters

        self.encode_sources = cfg.encode_sources
        self.decode_sources = cfg.decode_sources
        self.extra_sources = cfg.has('extra_sources', [])
        self.encode_cameras = cfg.has('encode_cameras', 'mod')
        self.latent_grid = cfg.latent.has('mode') and cfg.latent.mode == 'grid'
        self.latent_grid_dim = cfg.latent.has('grid_dim', 0)

        self.use_image_embeddings = cfg.encoder.has('use_image_embeddings', True)

        if len(self.encode_sources) == 0 or not is_list(self.encode_sources[0]):
            self.encode_sources = [self.encode_sources]

        # Downsample

        self.downsample_encoder = cfg.downsample_encoder
        self.downsample_decoder = cfg.downsample_decoder

        # Create embeddings

        self.embeddings = nn.ModuleDict()
        for key in cfg.embeddings.keys():
            mode = key if '_' not in key else key.split('_')[0]
            self.embeddings[key] = self.EMBEDDING_CLASSES[mode](cfg.embeddings.dict[key])

        # Parse shared decoder parameters

        if cfg.decoders.has('shared'):
            for task in cfg.decoders.keys():
                if task is not 'shared':
                    for key, val in cfg.decoders.shared.items():
                        if key not in cfg.decoders.dict[task].keys():
                            cfg.decoders.dict[task].dict[key] = val

        # Embeddings dimension

        tot_encoders = [self.total_dimension(sources) for sources in self.encode_sources]

        self.models = nn.ModuleList()
        for i in range(len(tot_encoders)):
            cfg.encoder.d_model = tot_encoders[i]
            is_variational = self.is_variational and self.variational_params['encode_vae'][i]
            self.models.append(PerceiverModel(cfg, is_variational=is_variational))

        # Decoders [###### MOVED TO AFTER PERCEIVER MODEL ######]

        self.decoders = nn.ModuleDict()
        for i in range(len(self.decode_sources)):
            is_variational = self.is_variational and \
                             self.variational_params['encode_vae'][self.decode_sources[i][1]]
            self.decode_sources[i] += [is_variational]
        for name, _, _, embeddings, decoders, _ in self.decode_sources:
            tot_decoder = self.total_dimension(embeddings)
            for dec in decoders:
                if dec not in self.decoders.keys():
                    cfg.decoders.dict[dec].name = name
                    cfg.decoders.dict[dec].num_channels = tot_decoder
                    cfg.decoders.dict[dec].d_latents = cfg.latent.dim
                    self.decoders[dec] = self.DECODER_CLASSES[dec.split('_')[0]](cfg.decoders.dict[dec])

        self.bypass_self_attention = cfg.has('bypass_self_attention', True)
        self.freeze_encoder = cfg.encoder.has('freeze', False)
        self.really_encode = cfg.encoder.has('really_encode', True)

    @property
    def tasks(self):
        return self.decoders.keys()

    def total_dimension(self, sources):
        if not self.use_image_embeddings and 'image' in sources:
            sources = list(sources)
            sources.remove('image')
        return sum([self.embeddings[source].channels for source in sources if source not in self.extra_sources]) + \
               self.latent_grid_dim

    def get_embeddings(self, data, camera_mode, sources, downsample, encoded=None, sample=None,
                       previous=None, monodepth=None):

        cam_mode = f'cam_{camera_mode}'

        if 'image' in sources:
            rgb_dict = {key: val['rgb'] for key, val in data.items()}
            feats_dict = self.embeddings['image'](rgb_dict)
            for key in data.keys():
                data[key]['feats'] = feats_dict[key]

        if monodepth is not None:
            monodepth['depth_idx'] = {}

        embeddings = OrderedDict()
        for key, val in data.items():

            embedding = OrderedDict()

            cam_scaled = val[cam_mode].scaled(1. / downsample)
            cam_scaled2 = None

            embedding['info'] = {}

            if self.training and sample is not None and sample > 0.0:
                if previous is None:
                    idx, start = self.sample_decoder_idx(sample, cam_scaled.batch_size, cam_scaled.hw)
                else:
                    idx, start = previous['info'][key]['idx'], previous['info'][key]['start']
                if start is not None:
                    cam_scaled2 = cam_scaled.offset_start(start).scaled(1. / sample)
                    val['cam_scaled'] = embedding['info']['cam_scaled'] = cam_scaled2
            else:
                idx = start = None
                val['cam_scaled'] = embedding['info']['cam_scaled'] = cam_scaled

            if monodepth is not None:
                b, _, h, w = monodepth['depth'][key][0].shape
                depth = monodepth['depth'][key][0].view(b, 1, -1).permute(0, 2, 1)
                if idx is not None:
                    depth = torch.stack([depth[j][idx[j]] for j in range(len(idx))], 0)
                    # from vidar.utils.viz import viz_depth
                    # from vidar.utils.write import write_image
                    # write_image('depth.png', viz_depth(depth1.view(b, 1, 192, 640)[0]))
                    # write_image('depth_sub.png', viz_depth(depth2.view(b, 1, 48, 160)[0]))
                    # import sys
                    # sys.exit()
                monodepth['depth_idx'][key] = depth

            embedding['info']['idx'] = idx
            embedding['info']['start'] = start

            for source in sources:
                if source.startswith('camera'):
                    val['camera'], embedding[source] = self.embeddings[source](
                        cam_scaled, key, idx,
                        meta=val['meta'] if 'meta' in val else None)
                    if start is not None:
                        h, w = cam_scaled2.hw
                        b, n, c = val['camera'].shape
                        val['camera'] = val['camera'].permute(0, 2, 1).reshape(b, c, h, w)

            if 'image' in sources:
                rgb = val['feats']
                embedding['image'] = rgb.view(*rgb.shape[:-2], -1).permute(0, 2, 1)
                if idx is not None:
                    embedding['image'] = torch.stack([
                        embedding['image'][i][idx[i]] for i in range(embedding['image'].shape[0])], 0)

            for source in sources:
                if source.startswith('volumetric'):
                    embedding['info']['z_samples'], embedding[source] = self.embeddings[source](
                        cam_scaled, key, data, previous, idx)

            for source in sources:
                if source.startswith('multiview'):
                    if encoded is None:
                        encoded = {'data': data}
                    multiview_previous = previous if previous is not None else monodepth
                    embedding['info']['z_samples'], embedding['info']['xyz'], embedding[source] = \
                        self.embeddings[source](cam_scaled, key, data, encoded, multiview_previous, idx)
                    embedding.pop('image', None)

            for source in sources:
                if source.startswith('projection'):
                    embedding[source] = self.embeddings[source](
                        cam_scaled2 if cam_scaled2 is not None else cam_scaled,
                        key, encoded, embedding['info'], idx)

            if idx is not None and previous is None:
                data_key = data[key]
                for key_gt in data_key.keys():
                    if data_key[key_gt] is not None and not key_gt.startswith('cam'):
                        if is_tensor(data_key[key_gt]) and data_key[key_gt].dim() == 4:
                            data_key[key_gt] = data_key[key_gt].permute(0, 2, 3, 1).reshape(
                                data_key[key_gt].shape[0], -1, data_key[key_gt].shape[1])
                            data_key[key_gt] = torch.stack([
                                data_key[key_gt][i, idx[i]] for i in range(data_key[key_gt].shape[0])], 0)
                            if start is not None:
                                h, w = cam_scaled2.hw
                                b, n, c = data_key[key_gt].shape
                                data_key[key_gt] = data_key[key_gt].permute(0, 2, 1).reshape(b, c, h, w)

            embeddings[key] = embedding

        if monodepth is not None:
            monodepth.pop('depth_idx')

        return embeddings

    @staticmethod
    def sample_decoder_idx(sample_decoder, b, hw):
        n = hw[0] * hw[1]
        if is_int(sample_decoder):
            idx, start = [], []
            for _ in range(b):
                start_i = torch.randint(0, sample_decoder, (2,))
                idx_i = torch.arange(0, n).reshape(hw)
                idx_i = idx_i[start_i[0]::sample_decoder, start_i[1]::sample_decoder].reshape(-1)
                idx.append(idx_i)
                start.append(start_i)
            start = torch.stack(start, 0)
        else:
            tot = int(sample_decoder * n)
            idx = [torch.randperm(n)[:tot] for _ in range(b)]
            start = None
        idx = torch.stack(idx, 0)
        return idx, start

    def bypass_encode(self, data, scene, idx):
        return {
            'embeddings': None,
            'latent': self.models[idx].embeddings(batch_size=1, scene=scene) if self.bypass_self_attention else
                      self.models[idx](data=None)['last_hidden_state'],
            'data': data,
        }

    def encode(self, sources=None, data=None, embeddings=None, sample_encoder=0, scene=None):
        assert data is not None or embeddings is not None
        assert data is None or embeddings is None

        sources = sources if sources is not None else self.encode_sources

        return [self.single_encode(sources, data, embeddings, sample_encoder, scene, idx=i)
                for i in range(len(sources))]

    def single_encode(self, sources=None, data=None, embeddings=None, sample_encoder=0, scene=None, idx=0):
        assert data is not None or embeddings is not None
        assert data is None or embeddings is None

        # Freeze encoder if requested
        for model in self.models:
            freeze_layers_and_norms(model, flag_freeze=self.freeze_encoder)

        # Get default sources if they are not provided
        sources = sources if sources is not None else self.encode_sources
        sources = sources[idx]

        camera_mode = self.encode_cameras[idx] if is_list(self.encode_cameras) else self.encode_cameras

        # Don't encode if there is no data or sources to use
        if len(data) == 0 or len(sources) == 0:
            return self.bypass_encode(data, scene, idx=idx)

        # Create embeddings if they are not provided
        if embeddings is None:
            embeddings = self.get_embeddings(
                data=data, sources=sources, camera_mode=camera_mode, downsample=self.downsample_encoder)
        embeddings = {key: torch.cat([val[source] for source in sources if source in val], -1) for key, val in embeddings.items()}

        # Sample embeddings if requested
        if self.training and sample_encoder > 0:
            for key in embeddings.keys():
                tot = sample_encoder if is_int(sample_encoder) \
                    else int(sample_encoder * embeddings[key].shape[1])
                embeddings[key] = torch.stack([
                    embeddings[key][i, torch.randperm(embeddings[key].shape[1])[:tot], :]
                    for i in range(embeddings[key].shape[0])], 0)

        # Don't encode if not requested
        if not self.really_encode:
            return self.bypass_encode(data, scene, idx=idx)

        # Encode embeddings
        encode_output = self.models[idx](data=embeddings, scene=scene)

        # Return embeddings, latent space, and updated data
        return {
            'embeddings': embeddings,
            'latent': encode_output['last_hidden_state'],
            'data': data,
        }

    def decode(self, encoded, sources=None, data=None, embeddings=None, sample_decoder=0, scene=None, monodepth=None):
        assert data is not None or embeddings is not None
        assert data is None or embeddings is None

        # Get default sources if they are not provided
        sources = sources if sources is not None else self.decode_sources

        # Initialize structures
        outputs, previous = [], None
        merged_output = {'losses': {}, 'embeddings': {}, 'output': {}}

        # Decode output for each source
        for source in sources:
            output = self.single_decode(encoded, source[1], source[2], source[3], source[4],
                                        data, embeddings, previous, sample_decoder,
                                        is_variational=source[5], scene=scene, monodepth=monodepth)
            previous = output['output']
            outputs.append(output)

        # Combine all outputs
        for output, source in zip(outputs, sources):
            name = '' if len(source[0]) == 0 else '_' + source[0]
            merged_output['losses'].update({'%s%s' % (key, name): val
                                            for key, val in output['losses'].items()})
            merged_output['embeddings'].update({'%s%s' % (key, name): val
                                                for key, val in output['embeddings'].items()})
            merged_output['output'].update({'%s%s' % (key, name): val
                                            for key, val in output['output'].items()})

        # Return merged output
        return merged_output

    def single_decode(self, encoded, idx, camera_mode, sources, tasks, data=None,
                      embeddings=None, previous=None, sample_decoder=0,
                      is_variational=False, scene=None, monodepth=None):

        # Create embeddings if they are not provided
        if embeddings is None:
            embeddings = self.get_embeddings(
                data=data, sources=sources, camera_mode=camera_mode, downsample=self.downsample_decoder,
                encoded=encoded[idx], sample=sample_decoder, previous=previous, monodepth=monodepth,
            )

####

        latent = encoded[idx]['latent']

        if self.latent_grid:
            latent1, latent2 = latent
            for key in embeddings.keys():
                xyz = embeddings[key]['info']['xyz']
                embeddings[key]['grid'] = latent1.sample(xyz).squeeze(1).permute(0, 2, 3, 1)
            sources = [s for s in sources] + ['grid']
            latent = latent2

####

        # Initialize output and losses
        output, losses = {}, {}

        # Get additional information and stack embeddings according to source
        info = {key: val['info'] for key, val in embeddings.items()}
        source_embeddings = {key: torch.cat([
            val[source] for source in sources if source not in self.extra_sources], -1)
            for key, val in embeddings.items()}
        extra_embeddings = {key: torch.cat([
            val[source] for source in self.extra_sources], -1) if len(self.extra_sources) > 0 else None
            for key, val in embeddings.items()}

        # Expand latent space if batch dimensions do not agree
        batch_size = source_embeddings[list(source_embeddings.keys())[0]].shape[0]
        if not is_dict(latent):
            if latent.shape[0] == 1 and latent.shape[0] != batch_size:
                latent = latent.repeat(batch_size, 1, 1)
        else:
            for key in latent.keys():
                if latent[key].shape[0] == 1 and latent[key].shape[0] != batch_size:
                    latent[key] = latent[key].repeat(batch_size, 1, 1)

        # Sample from latent space if it's variational
        if is_variational:
            output_variational = self.sample_from_latent(latent)
            losses.update(**{key: val for key, val in output_variational.items() if 'loss' in key})
            latent = output_variational['sampled_latent']

        # Decode all tasks from each embedding
        for task in tasks:
            output[task] = {key: make_list(self.decoders[task](
                query=val, z=latent[key] if is_dict(latent) else latent,
                key=key, info=info, previous=previous,
                extra=extra_embeddings[key], scene=scene)['predictions'])
                            for key, val in source_embeddings.items()}
            # Break volumetric into rgb and depth predictions
            if task.startswith('volumetric'):
                for task_key in ['rgb', 'depth']:
                    output[task.replace('volumetric', task_key)] = {
                        key: [v[task_key] for v in val] for key, val in output[task].items()}
            if task.startswith('multiview'):
                for task_key in ['rgb', 'depth']:
                    output[task.replace('multiview', task_key)] = {
                        key: [v[task_key] for v in val] for key, val in output[task].items()}

        output['info'] = info

        # Return losses, embeddings, and output
        return {
            'losses': losses,
            'embeddings': embeddings,
            'output': output,
        }

    def multi_decode(self, encoded, sources=None, data=None,
                     embeddings=None, sample_decoder=0, num_evaluations=None, scene=None, monodepth=None):
        output = {}

        for i in range(num_evaluations):
            output_i = self.decode(
                encoded, sources, data, embeddings, sample_decoder, scene=scene, monodepth=monodepth)

            if i == 0:
                output = {key: val for key, val in output_i['output'].items()}
                # embeddings, data = output_i['embeddings'], None
            else:
                for task in output_i['output'].keys():
                    if str_not_in(task, ['info', 'volumetric']):
                        for key in output_i['output'][task].keys():
                            output[task][key].extend(output_i['output'][task][key])


        if not self.training:
            for task in list(output.keys()):
                if str_not_in(task, ['info', 'volumetric']):
                    output[f'{task}_mean'] = {}
                    output[f'stddev_{task}'] = {}
                    for key in output[task].keys():
                        val = torch.stack(output[task][key], 0)
                        output[f'{task}_mean'][key] = [val.mean(0)]
                        output[f'stddev_{task}'][key] = [val.std(0).sum(1, keepdim=True)]

        return {
            'losses': None,
            'embeddings': embeddings,
            'output': output,
        }

    def sample_from_latent(self, latent):

        if is_dict(latent):
            latents = {key: self.sample_from_latent(val) for key, val in latent.items()}
            output = {
                'sampled_latent': {key: val['sampled_latent'] for key, val in latents.items()}
            }
            if self.training:
                output['kld_loss'] = sum([val['kld_loss'] for val in latents.values()]) / len(latents)
            return output

        n = latent.shape[-1] // 2
        mu, logvar = latent[:, :, :n], latent[:, :, n:]
        logvar = logvar.clamp(max=10)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sampled_latent = eps * std + mu

        output = {
            'sampled_latent': sampled_latent
        }

        if self.training:
            output['kld_loss'] = self.variational_params['kld_weight'] * torch.mean(
                - 0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp(), dim=[1, 2]), dim=0)

        return output

