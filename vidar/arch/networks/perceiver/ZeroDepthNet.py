# Copyright 2023 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict

import torch
import torch.nn as nn

from einops import rearrange

from knk_vision.vidar.vidar.arch.networks.layers.define.decoders.depth import DepthDecoder
from knk_vision.vidar.vidar.arch.networks.layers.define.decoders.rgb import RGBDecoder
from knk_vision.vidar.vidar.arch.networks.layers.define.decoders.volumetric import VolumetricDecoder
from knk_vision.vidar.vidar.arch.networks.layers.define.embeddings.camera import CameraEmbeddings
from knk_vision.vidar.vidar.arch.networks.layers.define.embeddings.image import ImageEmbeddings
from knk_vision.vidar.vidar.arch.networks.layers.define.embeddings.volumetric import VolumetricEmbeddings
from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.model import PerceiverModel
from knk_vision.vidar.vidar.utils.augmentations import augment_define
from knk_vision.vidar.vidar.utils.data import make_list, str_not_in, sum_list, get_from_dict, remove_nones_dict, update_dict_nested, get_random
from knk_vision.vidar.vidar.utils.nerf import apply_idx
from knk_vision.vidar.vidar.utils.networks import freeze_layers_and_norms
from knk_vision.vidar.vidar.utils.tensor import grid_sample, norm_pixel_grid, same_shape
from knk_vision.vidar.vidar.utils.types import is_int, is_dict, is_list
from knk_vision.vidar.vidar.geometry.camera_motion import slerp
from knk_vision.vidar.vidar.geometry.camera import Camera


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def perturb_camera(cam, center, cam_noise, center_noise, weight):
    """ Perturb camera by changing its location and orientation from the origin"""
    if cam_noise is None:
        return cam
    if center_noise is not None:
        center_noise = weight * center_noise * get_random(center)
        center = center + center_noise
    cam.look_at(center)
    cam.Twc.translateUp(weight * cam_noise[0] * get_random())
    cam.Twc.translateLeft(weight * cam_noise[1] * get_random())
    cam.Twc.translateForward(weight * cam_noise[2] * get_random())
    cam.look_at(center)
    return cam


def project_virtual_cameras(cams, pcl_proj, clr_proj):
    """Project pointcloud onto virtual cameras to produce virtual supervision"""
    if is_dict(cams):
        virtual_data = {}
        for key, val in cams.items():
            rgb_proj, depth_proj = val.project_pointcloud(pcl_proj, clr_proj)
            virtual_data[key] = {
                'cam': val,
                'gt': {
                    'rgb': rgb_proj.contiguous(),
                    'depth': depth_proj.contiguous(),
                }
            }
    return virtual_data


def create_virtual_decode(encode_data, cams=None, params=None):
    """Create virtual supervision for training and validation"""

    gt_rgb = {key: val['gt']['rgb'] for key, val in encode_data.items()}
    gt_depth = {key: val['gt']['depth'] for key, val in encode_data.items()}
    gt_cams = {key: val['cam'] for key, val in encode_data.items()}

    pcls_proj = {key: gt_cams[key].reconstruct_depth_map(gt_depth[key], to_world=True) for key in gt_cams.keys()}
    pcls_proj = {key: val.reshape(*val.shape[:2], -1) for key, val in pcls_proj.items()}

    pcl_proj = torch.cat([pcl for pcl in pcls_proj.values()], -1)
    clr_proj = torch.cat([rgb.reshape(*rgb.shape[:2], -1) for rgb in gt_rgb.values()], -1)

    # If cameras are provided, use them directly
    if cams is not None:
        return project_virtual_cameras(cams, pcl_proj, clr_proj)

    mode, n_samples = params[:2]
    if mode == 'slerp':

        cams = {}
        keys = list(gt_cams.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                virtual_cams = slerp(
                    [gt_cams[keys[i]], gt_cams[keys[j]]], n=n_samples, keep_edges=False, perturb=True)
                for k, cam in enumerate(virtual_cams):
                    cams[(keys[i], k)] = cam
        return project_virtual_cameras(cams, pcl_proj, clr_proj)

    elif mode == 'jitter':

        tries, decay, thr = 10, 0.9, 100
        cam_noise, center_noise = params[2], params[3]
        gt_pcl_centers = {key: val.mean(-1) for key, val in pcls_proj.items()}

        virtual_data = {}
        for key in encode_data.keys():
            for i in range(n_samples):

                cam = gt_cams[key].clone()
                pcl_center = gt_pcl_centers[key].clone()

                weight = 1.0
                rgb_proj = depth_proj = None
                for _ in range(tries):
                    cam = perturb_camera(cam, pcl_center, cam_noise, center_noise, weight)
                    rgb_proj_try, depth_proj_try = cam.project_pointcloud(pcl_proj, clr_proj)
                    if (depth_proj_try > 0).sum() > thr:
                        rgb_proj, depth_proj = rgb_proj_try, depth_proj_try
                        break
                    else:
                        weight = weight * decay
                if rgb_proj is not None and depth_proj is not None:
                    virtual_data[(key, i)] = {
                        'cam': cam,
                        'gt': {
                            'rgb': rgb_proj.contiguous(),
                            'depth': depth_proj.contiguous(),
                        }
                    }

        return virtual_data


class ZeroDepthNet(nn.Module):

    DECODER_CLASSES = {
        'rgb': RGBDecoder,
        'depth': DepthDecoder,
        'volumetric': VolumetricDecoder,
    }

    EMBEDDING_CLASSES = {
        'image': ImageEmbeddings,
        'camera': CameraEmbeddings,
        'volumetric': VolumetricEmbeddings,
    }

    TASK_KEYS = [
        'rgb', 'depth', 'scnflow', 'logvar', 'stddev',
    ]

    def __init__(self, cfg):
        """
        ZeroDepth network (https://arxiv.org/pdf/2306.17253.pdf) for zero-shot scale-aware depth estimation
        
        Parameters
        ----------
        cfg : Config
            Configuration with parameters
        """            
        super().__init__()

        # VARIATIONAL

        self.is_variational = cfg.has('variational')
        self.variational_params = None if not self.is_variational else {
            'kld_weight': cfg.variational.kld_weight,
            'encode_vae': cfg.variational.encode_vae,
            'n_validation_samples': cfg.variational.n_validation_samples,
        }

        # Parameters

        self.encode_sources = cfg.encode_sources
        self.decode_sources = cfg.decode_sources
        self.merge_encode_sources = cfg.has('merge_encode_sources', False)

        self.decode_sources_train = cfg.has('decode_sources_train', None)
        self.decode_sources_val = cfg.has('decode_sources_val', None)

        self.extra_sources = cfg.has('extra_sources', [])
        self.encode_cameras = cfg.has('encode_cameras', 'mod')
        self.latent_grid = cfg.latent.has('mode') and cfg.latent.mode == 'grid'
        self.latent_grid_dim = cfg.latent.has('grid_dim', 0)

        self.bypass_self_attention = cfg.has('bypass_self_attention', True)
        self.freeze_encoder = cfg.encoder.has('freeze', False)
        self.really_encode = cfg.encoder.has('really_encode', True)
        self.encoder_embeddings = cfg.encoder.has('encoder_embeddings', False)
        self.use_image_embeddings = cfg.encoder.has('use_image_embeddings', True)

        if len(self.encode_sources) == 0 or not is_list(self.encode_sources[0]):
            self.encode_sources = [self.encode_sources]

        self.decode_virtual_cameras = cfg.has('decode_virtual_cameras', None)

        # Downsample

        self.downsample_encoder = make_list(cfg.downsample_encoder)
        self.downsample_decoder = cfg.downsample_decoder
        self.sample_encoder = cfg.has('sample_encoder', 0)
        self.sample_decoder = cfg.sample_decoder
        self.shake_encoder = cfg.has('shake_encoder', False)
        self.shake_decoder = cfg.has('shake_decoder', False)

        # Create embeddings

        self.embeddings = nn.ModuleDict()
        for key in cfg.embeddings.keys():
            mode = key if '_' not in key else key.split('_')[0]
            self.embeddings[key] = self.EMBEDDING_CLASSES[mode](cfg.embeddings.dict[key])

        # Parse shared decoder parameters

        if cfg.decoders.has('shared'):
            for task in cfg.decoders.keys():
                if task != 'shared':
                    for key, val in cfg.decoders.shared.items():
                        if key not in cfg.decoders.dict[task].keys():
                            cfg.decoders.dict[task].dict[key] = val

        # Embeddings dimension

        tot_encoders = [self.total_dimension(sources) for sources in self.encode_sources]

        self.models = nn.ModuleList()
        if self.merge_encode_sources is False:
            for i in range(len(tot_encoders)):
                cfg.encoder.d_model = tot_encoders[i]
                cfg.encoder.use_flash_attention = cfg.has('use_flash_attention', False)
                is_variational = self.is_variational and self.variational_params['encode_vae'][i]
                self.models.append(PerceiverModel(cfg, is_variational=is_variational))
        else:
            cfg.encoder.d_model = self.merge_encode_sources
            cfg.encoder.use_flash_attention = cfg.has('use_flash_attention', False)
            is_variational = self.is_variational and self.variational_params['encode_vae'][0]
            self.models.append(PerceiverModel(cfg, is_variational=is_variational))
            self.merge_networks = nn.ModuleList()
            for i in range(len(tot_encoders)):
                self.merge_networks.append(nn.Linear(tot_encoders[i], self.merge_encode_sources))

        # Decoders

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
                    cfg.decoders.dict[dec].use_flash_attention = cfg.has('use_flash_attention', False)
                    if cfg.decoders.dict[dec].has('ensemble'):
                        self.decoders[dec] = torch.nn.ModuleList(
                            [self.DECODER_CLASSES[dec.split('_')[0]](cfg.decoders.dict[dec])
                             for _ in range(cfg.decoders.dict[dec].ensemble)])
                    else:
                        self.decoders[dec] = self.DECODER_CLASSES[dec.split('_')[0]](cfg.decoders.dict[dec])

        self.encode_augmentation = cfg.has('encode_augmentation', None)
        self.decode_augmentation = cfg.has('decode_augmentation', None)
        self.validation_augmentation = cfg.has('validation_augmentation', None)

        self.compile = cfg.has('compile', False)
        if self.compile:
            for i in range(len(self.models)):
                self.models[i] = torch.compile(self.models[i])
        self.mixed_precision = cfg.has('mixed_precision', False)

    @property
    def tasks(self):
        """Return the tasks of the network"""
        return self.decoders.keys()

    def total_dimension(self, sources):
        """Return the total dimension of the embeddings"""
        if not self.use_image_embeddings and 'image' in sources:
            sources = list(sources)
            sources.remove('image')
        return sum([self.embeddings[source].channels
                    for source in sources if source not in self.extra_sources]) + \
               self.latent_grid_dim

    @staticmethod
    def merge_embeddings(embeddings, sources):
        """Merge embeddings from different sources"""
        needs_repeat, value_repeat = False, None
        embeddings_merge = {}
        for key, val in embeddings.items():
            emb_list = []
            for source in sources:
                if source in val:
                    emb_list.append(val[source])
                    if val[source].dim() == 4:
                        needs_repeat, value_repeat = True, val[source].shape[2]
            embeddings_merge[key] = emb_list
        if needs_repeat:
            for key, val in embeddings_merge.items():
                for i in range(len(val)):
                    if val[i].dim() == 3:
                        val[i] = val[i].unsqueeze(2).repeat(1, 1, value_repeat, 1)
        for key, val in embeddings_merge.items():
            cat = [v for v in val if v.shape[1] > 1]
            stack = [v for v in val if v.shape[1] == 1]
            cat = torch.cat(cat, -1)
            if len(stack) > 0:
                cat = torch.cat([cat] + stack, 1)
            embeddings_merge[key] = cat
        return embeddings_merge

    def get_embeddings(self, data, sources, camera_mode, encoded=None, shake=False,
                       downsample=None, sample=None, previous=None, change_gt=True):
        """Get embeddings from different sources"""
        for source in sources:
            if source.startswith('image'):
                rgb_dict = {key: val['gt']['rgb'] for key, val in data.items()}
                cam_dict = {key: val['cam'] for key, val in data.items()}
                feats_dict = self.embeddings[source](rgb_dict, cam_dict)
                for key in data.keys():
                    for key2 in feats_dict[key].keys():
                        data[key][f'{source}_{key2}'] = feats_dict[key][key2]

        embeddings = OrderedDict()
        for key, val in data.items():
            embedding = OrderedDict()

            cam_scaled, cam_sampled = get_from_dict(val, 'cam'), None
            if cam_scaled is not None and downsample > 1:
                cam_scaled = cam_scaled.scaled(1. / downsample)

            embedding['info'] = {}

            # Reuse the same coordinates if available, and sample them if requested
            if self.training and sample is not None and sample > 0.0:
                if 'coords' in data[key] and 'offset' in data[key]:
                    coords, offset = data[key]['coords'], data[key]['offset']
                elif previous is None:
                    coords, offset = self.sample_coords_offset(sample, cam_scaled.batch_size, cam_scaled.hw)
                else:
                    coords, offset = previous['info'][key]['coords'], previous['info'][key]['offset']
                if coords is not None and offset is not None:
                    cam_sampled = cam_scaled.offset_start(offset).scaled(1. / sample)
                    val['cam_scaled'] = embedding['info']['cam_scaled'] = cam_sampled
            else:
                val['cam_scaled'] = embedding['info']['cam_scaled'] = cam_scaled
                coords = offset = None

            # Save for later
            embedding['info']['coords'] = coords
            embedding['info']['offset'] = offset

            grid = cam_scaled.pixel_grid(shake=True) if shake and self.training else None

            for source in sources:
                # Camera embeddings
                if source.startswith('camera'):
                    depth = None # data[key]['gt']['depth']
                    val['camera'], embedding[source] = self.embeddings[source](
                        cam_scaled, key, coords, depth=depth, grid=grid, meta=get_from_dict(val, 'meta')
                    )
                    if cam_sampled is not None:
                        h, w = cam_sampled.hw
                        b, n, c = val['camera'].shape
                        val['camera'] = val['camera'].permute(0, 2, 1).reshape(b, c, h, w)

            for source in sources:
                # Image embeddings
                if source.startswith('image'):
                    rgb = val[f'{source}_feats']
                    embedding[source] = rgb.reshape(*rgb.shape[:-2], -1).permute(0, 2, 1)
                    if 'depth_image' in val.keys():
                        embedding[f'depth_{source}'] = val['depth_image']
                    if grid is not None:
                        feats = embedding[source]
                        b, _, c = feats.shape
                        feats = feats.reshape(b, *cam_scaled.hw, -1).permute(0, 3, 1, 2)
                        grid = norm_pixel_grid(grid[:, :2].view(b, 2, *cam_scaled.hw))
                        embedding[source] = grid_sample(
                            feats, grid.permute(0, 2, 3, 1), padding_mode='border', mode='bilinear'
                        ).view(b, c, -1).permute(0, 2, 1)
                    if cam_sampled is not None:
                        embedding[source] = apply_idx(embedding[source], coords)
                        if f'depth_{source}' in embedding.keys():
                            embedding[f'depth_{source}'] = apply_idx(embedding[f'depth_{source}'], coords)

            # Clip embeddings
            if 'clip' in sources:
                embedding[source] = self.embeddings[source](val['gt']['rgb'])

            # Volumetric embeddings
            for source in sources:
                if source.startswith('volumetric'):
                    val['camera'], embedding[source], info = self.embeddings[source](
                        cam_scaled, key, data, coords, previous,
                    )
                    if cam_sampled is not None:
                        h, w = cam_sampled.hw
                        b, n, c = val['camera'].shape
                        val['camera'] = val['camera'].permute(0, 2, 1).reshape(b, c, h, w)
                    embedding['info'].update(**info)

            # Modify GT information to conform to the prediction sampling 
            if change_gt and 'gt' in data[key].keys() and previous is None and cam_sampled is not None:
                data_gt = data[key]['gt']
                for key_gt in data_gt.keys():
                    if data_gt[key_gt] is not None:
                        if data_gt[key_gt].dim() == 4 and \
                                not same_shape(cam_sampled.hw, data_gt[key_gt].shape[-2:]):
                            data_gt[key_gt] = rearrange(data_gt[key_gt], 'b c h w -> b (h w) c')
                            data_gt[key_gt] = apply_idx(data_gt[key_gt], coords)
                            if offset is not None:
                                data_gt[key_gt] = rearrange(
                                    data_gt[key_gt], 'b (h w) c -> b c h w',
                                    h=cam_sampled.hw[0], w=cam_sampled.hw[1])

            embeddings[key] = embedding
            remove_nones_dict(val)

        return embeddings

    @staticmethod
    def sample_coords_offset(sample_decoder, b, hw):
        """Sample coordinates and offset for the decoder"""
        n = hw[0] * hw[1]
        if is_int(sample_decoder):
            if sample_decoder < 64:
                coords, offset = [], []
                for _ in range(b):
                    offset_i = torch.randint(0, sample_decoder, (2,))
                    coords_i = torch.arange(0, n).reshape(hw)
                    coords_i = coords_i[offset_i[0]::sample_decoder, offset_i[1]::sample_decoder].reshape(-1)
                    coords.append(coords_i)
                    offset.append(offset_i)
                offset = torch.stack(offset, 0)
            else:
                tot = sample_decoder
                coords = [torch.randperm(n)[:tot] for _ in range(b)]
                offset = None
        else:
            tot = int(sample_decoder * n)
            coords = [torch.randperm(n)[:tot] for _ in range(b)]
            offset = None
        coords = torch.stack(coords, 0)
        return coords, offset

    def bypass_encode(self, data, scene, idx):
        """Bypass the encoder and return the embeddings"""
        return {
            'embeddings': None,
            'latent': self.models[idx].embeddings(batch_size=1, scene=scene) if self.bypass_self_attention else
                      self.models[idx](data=None)['last_hidden_state'],
            'data': data,
        }

    def encode(self, data=None, embeddings=None, scene=None):
        """Encode the data and return the embeddings"""
        # Augment data to encode
        if self.training:
            data = augment_define(data, self.encode_augmentation)
        # Freeze encoder if requested
        if self.training:
            for model in self.models:
                freeze_layers_and_norms(model, flag_freeze=self.freeze_encoder)
        # Create encode embeddings
        embeddings = [self.encode_embeddings(data, embeddings, scene, idx=i)
            for i in range(len(self.encode_sources))]
        # Encode embeddings
        if not self.merge_encode_sources:
            return [self.encode_fn(data, embeddings=embeddings[i], scene=scene, idx=i)
                    for i in range(len(self.encode_sources))]
        else:
            single_embeddings = {}
            for key in embeddings[0].keys():
                single_embeddings[key] = [
                    self.merge_networks[i](embeddings[i][key]) for i in range(len(embeddings))
                ]
                single_embeddings[key] = torch.cat(single_embeddings[key], 1)
            return [self.encode_fn(data, embeddings=single_embeddings, scene=scene, idx=0)]

    def encode_embeddings(self, data, embeddings, scene, idx):
        """Encode the embeddings onto the latent space"""
        # Get information from requested index
        sources = self.encode_sources[idx]
        camera_mode = self.encode_cameras[idx] if is_list(self.encode_cameras) else self.encode_cameras

        # Don't encode if there is no data or sources to use
        if len(data) == 0 or len(sources) == 0:
            return None # self.bypass_encode(data, scene, idx=idx)

        # Create embeddings if they are not provided
        if embeddings is None:
            embeddings = self.get_embeddings(
                data, sources, camera_mode,
                downsample=self.downsample_encoder[idx],
                shake=self.shake_encoder,
                change_gt=False,
            )
            embeddings = self.merge_embeddings(embeddings, sources)
        return embeddings

    def encode_fn(self, data=None, embeddings=None, scene=None, idx=0):
        """Encode function, taking embeddings as input"""
        # Sample embeddings if requested
        if self.training and self.sample_encoder != 0:
            for key in embeddings.keys():
                if is_int(self.sample_encoder):
                    tot = self.sample_encoder
                elif is_list(self.sample_encoder):
                    if len(self.sample_encoder) == 3:
                        if torch.rand(1) > self.sample_encoder[2]:
                            continue
                    tot = self.sample_encoder[0] + (self.sample_encoder[1] - self.sample_encoder[0]) * torch.rand(1)
                    tot = int(tot * embeddings[key].shape[1])
                else:
                    tot = int(self.sample_encoder * embeddings[key].shape[1])
                embeddings[key] = torch.stack([
                    embeddings[key][i, torch.randperm(embeddings[key].shape[1])[:tot], :]
                    for i in range(embeddings[key].shape[0])], 0)

        # Don't encode if not requested
        if not self.really_encode:
            return self.bypass_encode(data, scene, idx=idx)

        # Encode and produce output
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            encode_output = self.models[idx](data=data, embeddings=embeddings, scene=scene)

        # Return embeddings, latent space, and updated data
        return {
            'embeddings': embeddings,
            'latent': encode_output['last_hidden_state'],
            'data': data,
        }

    def decode(self, encoded, encode_data=None, decode_data=None, embeddings=None, scene=None):
        """Decode the data from the latent space"""
        decode_fn = self.multi_decode if self.is_variational else self.single_decode
        return decode_fn(encoded, encode_data, decode_data, embeddings, scene)

    def single_decode(self, encoded=None, encode_data=None, decode_data=None,
                      embeddings=None, scene=None, first=True):
        """Single decode function, taking encoded data and the latent space, and decoding embeddings"""

        # Initialize structures
        outputs, previous = {}, None
        merged_outputs = {'losses': {}, 'embeddings': {}, 'decoded': {}}

        valid_sources = []
        valid_decode_sources = self.decode_sources_train if self.training else self.decode_sources_val

        # Decode output for each source
        for i, source in enumerate(self.decode_sources):
            if valid_decode_sources is not None and source[0] not in valid_decode_sources:
                continue
            valid_sources.append(source)
            output = self.decode_fn(
                encoded, source[1], source[2], source[3], source[4],
                encode_data, decode_data, embeddings, previous,
                is_variational=source[5], scene=scene, first=(first and i==0),
            )
            outputs[source[0] if source[0] != '' else i] = output
            previous = output['decoded']

        # Combine all outputs
        for output, source in zip(outputs.values(), valid_sources):
            for mode in ['losses', 'decoded']:
                merged_outputs[mode].update({
                    key if source[0] == '' else '%s-%s' % (key, source[0]):
                        val for key, val in output[mode].items()
                })
            for mode in ['embeddings']:
                key0 = list(output[mode].keys())[0]
                for key in output[mode][key0].keys():
                    if key != 'info':
                        update_dict_nested(
                            merged_outputs, mode,
                            'embeddings' if source[0] == '' else '%s-%s' % ('embeddings', source[0]),
                            key, {k: output[mode][k][key] for k in output[mode].keys()}
                        )

        # Return merged output
        return merged_outputs

    def decode_fn(self, encoded, idx, camera_mode, sources, tasks,
                  encode_data=None, decode_data=None, embeddings=None, previous=None,
                  is_variational=False, scene=None, first=True):
        """Decode function, taking encoded data and the latent space, and decoding embeddings"""
        # Augment data to decode
        if self.training and first:
            decode_data = augment_define(decode_data, self.decode_augmentation)
        if not self.training and first:
            decode_data = augment_define(decode_data, self.validation_augmentation)

        if self.training and self.decode_virtual_cameras is not None:
            decode_data.update(create_virtual_decode(encode_data, params=self.decode_virtual_cameras))

        # Initialize output and losses
        decoded, losses = {}, {}

        # Create embeddings if they are not provided
        if embeddings is None:
            embeddings_data = encode_data if self.encoder_embeddings else decode_data
            embeddings = self.get_embeddings(
                data=embeddings_data, sources=sources, camera_mode=camera_mode,
                downsample=self.downsample_decoder, sample=self.sample_decoder,
                encoded=encoded[idx], previous=previous, shake=self.shake_decoder,
                change_gt=True,
            )

        # Parse source and extra embeddings
        source_embeddings = self.merge_embeddings(embeddings, sources)
        extra_embeddings = {key: torch.cat([
            val[source] for source in self.extra_sources], -1) if len(self.extra_sources) > 0 else None
            for key, val in embeddings.items()}

        # Get index latent
        latent = encoded[idx]['latent']

        # Get additional information and stack embeddings according to source
        decoded['info'] = {key: val['info'] for key, val in embeddings.items()}

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

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                if is_list(self.decoders[task]):
                    output_task = [{key: self.decoders[task][i](
                        query=val, z=latent,
                        encode_data=encode_data, decode_data=decode_data,
                        key=key, info=decoded['info'], previous=previous,
                        extra=extra_embeddings[key], scene=scene)['predictions']
                                   for key, val in source_embeddings.items()} for i in range(len(self.decoders[task]))]
                    output_task = {key1: {key2: [output_task[i][key1][key2]
                                                 for i in range(len(self.decoders[task]))]
                                          for key2 in output_task[0][key1].keys()}
                                   for key1 in output_task[0].keys()}
                    if not self.training:
                        for key1 in list(output_task.keys()):
                            for key2 in list(output_task[key1].keys()):
                                output_task[key1][key2] = torch.stack(output_task[key1][key2], 0)
                                output_task[key1]['stddev_' + key2] = [output_task[key1][key2].std(0).sum(1, keepdim=True)]
                                output_task[key1][key2] = [output_task[key1][key2].mean(0)]
                else:
                    output_task = {key: self.decoders[task](
                        query=val, z=latent,
                        encode_data=encode_data, decode_data=decode_data,
                        key=key, info=decoded['info'], previous=previous,
                        extra=extra_embeddings[key], scene=scene)['predictions']
                                   for key, val in source_embeddings.items()}

            key0 = list(output_task.keys())[0]
            for sub_task in output_task[key0].keys():
                for task_key in self.TASK_KEYS:
                    if sub_task.startswith(task_key):
                        decoded[sub_task] = {
                            key: make_list(output_task[key][sub_task]) for key in output_task.keys()
                        }

        # Return losses, embeddings, and output
        return {
            'losses': losses,
            'embeddings': embeddings,
            'decoded': decoded,
        }

    def multi_decode(self, encoded, encode_data=None, decode_data=None,
                     embeddings=None,  scene=None):
        """Decode the data from the latent space multiple times, for statistical analysis"""
        num_evaluations = 1 if self.training else \
            self.variational_params['n_validation_samples']

        losses, decoded = [], {}

        for i in range(num_evaluations):
            output_i = self.single_decode(
                encoded, encode_data, decode_data, embeddings, scene=scene, first=i==0)

            if i == 0:
                decoded = {key: val for key, val in output_i['decoded'].items()}
            else:
                for task in output_i['decoded'].keys():
                    if str_not_in(task, ['info', 'volumetric']):
                        for key in output_i['decoded'][task].keys():
                            decoded[task][key].extend(output_i['decoded'][task][key])

            losses.append(output_i['losses'])

        if not self.training:
            for task in list(decoded.keys()):
                if str_not_in(task, ['info', 'volumetric']):
                    mean, stddev = {}, {}
                    for key in decoded[task].keys():
                        val = torch.stack(decoded[task][key], 0)
                        mean[key] = [val.mean(0)]
                        stddev[key] = [val.std(0).sum(1, keepdim=True)]
                    decoded[f'{task}'] = mean
                    decoded[f'stddev_{task}'] = stddev

        return {
            'losses': sum_list(losses),
            'embeddings': output_i['embeddings'],
            'decoded': decoded,
        }

    def sample_from_latent(self, latent):
        """Sample from the latent space, to produce variational inference"""
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
        # eps = 0.001 * torch.ones_like(std)
        sampled_latent = eps * std + mu

        output = {
            'sampled_latent': sampled_latent
        }

        if self.training:
            output['kld_loss'] = self.variational_params['kld_weight'] * torch.mean(
                - 0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp(), dim=[1, 2]), dim=0)

        return output

    def forward(self, rgb, intrinsics):
        """Forward pass of the network, including encoding and decoding stages for depth estimation"""
        
        cam = Camera(K=intrinsics, hw=rgb)

        encode_data = {(0,0): {
            'cam': cam,
            'gt': {'rgb': rgb},
        }}

        decode_data = {(0,0): {
            'cam': cam,
        }}

        encoded_data = self.encode(
            data=encode_data,
        )

        decode_output = self.decode(
            encoded=encoded_data, encode_data=encode_data, decode_data=decode_data,
        )

        depth = decode_output['decoded']['depth'][(0,0)][0]
        return depth