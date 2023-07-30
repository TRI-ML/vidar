# Copyright 2023 Toyota Research Institute.  All rights reserved.

from vidar.arch.models.generic.GenericModel_utils import get_if_not_none, make_pairs, update_predictions
from vidar.utils.data import get_from_dict, remove_nones_dict, update_dict, update_dict_nested, sum_list, tensor_like
import torch


class GenericModelPredictions:
    """Generic model for multi-model inputs and multi-task output. Subclass focused on predictions functionality.

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """
    def __init__(self):
        pass

    def prepare_inputs(self, task_key, tgts, rgb, depth, intrinsics, pose, cams, filename):
        """Prepare inputs for forward pass."""
        input_labels = self.task_cfg[task_key].inputs
        inputs = {}
        for tgt in tgts:
            inputs[tgt] = {}
            if 'rgb' in input_labels:
                inputs[tgt]['rgb'] = get_if_not_none(rgb, tgt)
            if 'depth' in input_labels:
                inputs[tgt]['depth'] = get_if_not_none(depth, tgt)
            if 'intrinsics' in input_labels:
                inputs[tgt]['intrinsics'] = get_if_not_none(intrinsics, tgt,
                                            get_if_not_none(intrinsics, (0, tgt[1])))
            if 'pose' in input_labels:
                inputs[tgt]['pose'] = get_if_not_none(pose, tgt)
            if 'cams' in input_labels:
                inputs[tgt]['cams'] = get_if_not_none(cams, tgt)
            if 'filename' in input_labels:
                inputs[tgt]['filename'] = get_if_not_none(filename, tgt)
        return inputs

################################################

    def forward_generic(self, task_key, batch, predictions, extra):
        """Forward pass for generic model"""
        output = self.networks[task_key](batch, predictions, extra)
        return {
            'predictions': output,
            'losses': {},
            'metrics': {},
            'extra': {},
        }

################################################

    def forward_rgb(self, task_key, rgb, depth, intrinsics, pose, cams, filename):
        """Forward pass for rgb model"""
        tgts = self.get_targets(self.task_cfg[task_key], self.base_cam, rgb.keys())
        inputs = self.prepare_inputs(task_key, tgts, rgb, depth, intrinsics, pose, cams, filename)

        rgb_output = {
            tgt: self.networks[task_key](**inputs[tgt]) for tgt in tgts
        }
        rgb_pred = {
            key: val['rgb'] for key, val in rgb_output.items()
        }

        return {
            'predictions': {
                task_key: rgb_pred,
            },
            'losses': {},
            'metrics': {},
            'extra': {},
        }

################################################

    def forward_depth(self, task_key, rgb, depth, intrinsics, pose, cams, filename):
        """Forward pass for depth model"""
        tgts = self.get_targets(self.task_cfg[task_key], self.base_cam, rgb.keys())
        inputs = self.prepare_inputs(task_key, tgts, rgb, depth, intrinsics, pose, cams, filename)

        depth_output = {
            tgt: self.networks[task_key](**inputs[tgt]) for tgt in tgts
        }
        depth_pred = {
            key: val['depths'] for key, val in depth_output.items()
        }

        key0, losses = list(depth_output.keys())[0], {}
        if 'losses' in depth_output[key0].keys():
            for key in depth_output[key0]['losses'].keys():
                losses[key] = {tgt: depth_output[tgt]['losses'][key] for tgt in depth_output.keys()}
                losses[key] = sum(losses[key].values())

        return {
            'predictions': {
                task_key: depth_pred,
            },
            'losses': losses,
            'metrics': {},
            'extra': {},
        }

    def forward_multi_depth(self, task_key, rgb, cams):
        """Forward pass for multi-depth model"""
        tgts = self.get_targets(self.task_cfg[task_key], self.base_cam, rgb.keys())
        ctxs = self.get_contexts(self.task_cfg[task_key], tgts, rgb.keys())
        keys, pairs = make_pairs(tgts, ctxs, only_both_ways=False)

        predictions = {}
        for tgt, ctx in pairs:
            output = self.networks[task_key](rgb[tgt], rgb[ctx], cams[tgt], cams[ctx])
            update_dict_nested(predictions, task_key.replace('multi_depth', 'depth'), tgt, ctx, output['depth'])

        return {
            'predictions': predictions,
            'losses': {},
            'metrics': {},
            'extra': {},
        }

################################################

    def forward_normals(self, task_key, rgb, intrinsics):
        """Forward pass for normals model"""
        tgts = self.get_targets(self.task_cfg[task_key], self.base_cam, rgb.keys())

        normals_output = {
            tgt: self.networks[task_key](
                rgb=rgb[tgt],
                intrinsics=get_if_not_none(intrinsics, (0, tgt[1])),
            ) for tgt in tgts
        }
        normals_pred = {
            key: val['depths'] for key, val in normals_output.items()
        }

        return {
            'predictions': {
                task_key: normals_pred,
            },
            'losses': {},
            'metrics': {},
            'extra': {},
        }

################################################

    def forward_optical_flow(self, task_key, rgb):
        """Forward pass for optical flow model"""
        tgts = self.get_targets(self.task_cfg[task_key], self.base_cam, rgb.keys())
        ctxs = self.get_contexts(self.task_cfg[task_key], tgts, rgb.keys())
        keys, pairs = make_pairs(tgts, ctxs, only_both_ways=False)

        predictions = {}
        for tgt, ctx in pairs:
            output = self.networks[task_key](rgb[tgt], rgb[ctx])
            if self.task_cfg[task_key].calc_bidir:
                output['bwd_optical_flow'] = self.networks[task_key](rgb[ctx], rgb[tgt])['fwd_optical_flow']
            if 'fwd_optical_flow' in output:
                update_dict_nested(predictions, task_key, tgt, ctx, output['fwd_optical_flow'])
            if 'bwd_optical_flow' in output:
                update_dict_nested(predictions, task_key, ctx, tgt, output['bwd_optical_flow'])

        return {
            'predictions': predictions,
            'losses': {},
            'metrics': {},
            'extra': {},
        }

################################################

    def forward_scene_flow(self, task_key, rgb, intrinsics):
        """Forward pass for scene flow model"""
        tgts = self.get_targets(self.task_cfg[task_key], self.base_cam, rgb.keys())

        depth_output = {
            tgt: self.networks[task_key](
                rgb=rgb[tgt],
                intrinsics=get_if_not_none(intrinsics, (0, tgt[1])),
            ) for tgt in tgts
        }
        depth_pred = {
            key: val['depths'] for key, val in depth_output.items()
        }

        return {
            'predictions': {
                task_key: depth_pred,
            },
            'losses': {},
            'metrics': {},
            'extra': {},
        }

################################################

    def forward_pose(self, task_key, rgb, tgt, ctxs, invert=True):
        """Forward pass for pose model"""
        return {ctx: self.networks[task_key](
            [rgb[tgt[0], ctx[1]], rgb[ctx]], invert=(tgt[0] < ctx[0]) and invert)['transformation']
                for ctx in ctxs}

################################################

    def forward_intrinsics(self, task_key, rgb):
        """Forward pass for intrinsics model"""
        tgts = self.get_targets(self.task_cfg[task_key], self.base_cam, rgb.keys())
        return {tgt: self.networks[task_key](rgb[tgt]) for tgt in tgts}

################################################

    def forward_perceiver(self, task_key, rgb, depth, scene, cams, tag):
        """Forward pass for perceiver model"""
        tgts = self.get_targets(self.task_cfg[task_key], self.base_cam, cams.keys())
        encs = self.get_encodes(self.task_cfg[task_key], self.base_cam, cams.keys())

        predictions, losses = {}, []

        encode_data = {key: {
            'cam': cams[key],
            'gt': {
                'rgb': get_if_not_none(rgb, key),
                'depth': get_if_not_none(depth, key),
            },
            'meta': {
                'timestep': tensor_like([key[0]], rgb[key]),
            }
        } for key in encs}

        decode_data = {key: {
            'cam': cams[key],
            'gt': {
                'rgb': get_if_not_none(rgb, key),
                'depth': get_if_not_none(depth, key),
            },
            'meta': {
                'timestep': tensor_like([key[0]], rgb[key]),
                'tag': tag,
            }
        } for key in tgts}

        encoded_data = self.networks[task_key].encode(
            data=encode_data, scene=scene,
        )

        decode_output = self.networks[task_key].decode(
            encoded=encoded_data, encode_data=encode_data, decode_data=decode_data,
        )

        decoded_keep = [
            'rgb', 'depth', 'scnflow', 'logvar', 'stddev', 'info',
        ]

        info_keep = [
            ['cam_scaled', 'cams'],
            ['coords', 'coords'],
        ]

        for i in range(len(encoded_data)):
            key0 = list(encoded_data[i]['data'].keys())[0]
            for key, val in encoded_data[i]['data'][key0].items():
                for task in decoded_keep:
                    if key.startswith(task):
                        predictions[key.replace(task, f'{task}_{task_key}')] = {
                            k: v[key] for k, v in encoded_data[i]['data'].items()}

        decoded = decode_output['decoded']
        for key, val in decoded.items():
            for task in decoded_keep:
                if key.startswith(task) and '_' not in key:
                    update_dict(predictions, f'{task}_{task_key}{key.replace(task, "")}', val)
                elif key.startswith(f'stddev_{task}'):
                    update_dict(predictions, f'stddev_{task}_{task_key}{key.replace(f"stddev_{task}", "")}', val)
            if key.startswith('info'):
                for key_info, val_info in decoded[key].items():
                    suffix = f'{task_key}{key.replace("info", "")}'
                    for keep in info_keep:
                        if get_from_dict(val_info, keep[0]) is not None:
                            if f'{keep[1]}_{suffix}' not in predictions:
                                predictions[f'{keep[1]}_{suffix}'] = {}
                            predictions[f'{keep[1]}_{suffix}'][key_info] = val_info[keep[0]]

        embeddings = decode_output['embeddings']
        for key, val in embeddings.items():
            update_dict(predictions, f'embeddings_{key.replace("embeddings", task_key)}', val)

        losses.append({f'{task_key}_{key}': val for key, val in decode_output['losses'].items()})

        return {
            'predictions': predictions,
            'losses': sum_list(losses),
            'metrics': {},
            'extra': {
                'encode_data': encode_data,
                'encoded_data': encoded_data,
                'decode_data': decode_data,
                'latents': [e['latent'] for e in encoded_data],
            },
            'gt': remove_nones_dict({
              'rgb': remove_nones_dict({key: val['gt']['rgb'] for key, val in decode_data.items()}),
              'depth': remove_nones_dict({key: val['gt']['depth'] for key, val in decode_data.items()}),
            })
        }

################################################

    def forward_nerf(self, task_key, rgb, depth, optflow, scnflow, valid_motion, valid_optflow,
                     motion_mask, timestep, predictions):
        """Forward pass for nerf model"""
        return self.networks[task_key](
            rgb, depth, optflow, scnflow, valid_motion, valid_optflow,
            motion_mask, timestep, predictions, define=get_from_dict(self.networks, 'perceiver'),
        )

################################################

    def forward_predictions(self, batch, epoch):
        """Forward pass for predictions (calls specific forward passes as needed)"""

        predictions, losses, extra = {}, {}, {}

        # Parse batch information

        rgb = get_from_dict(batch, 'rgb')
        depth = get_from_dict(batch, 'depth')
        optflow = get_from_dict(batch, 'optical_flow')
        valid_motion = get_from_dict(batch, 'mask_motion_gt')
        valid_optflow = get_from_dict(batch, 'mask_optical_flow_gt') or \
                        get_from_dict(batch, 'valid_optical_flow')
        scnflow = get_from_dict(batch, 'scene_flow')
        mask_motion = get_from_dict(batch, 'mask_motion')
        scene = get_from_dict(batch, 'scene')
        timestep = get_from_dict(batch, 'timestep')
        tag = get_from_dict(batch, 'tag')
        filename = get_from_dict(batch, 'filename')

        # Calculate camera predictions

        pose = predictions['pose'] = self.calc_pose('pose', batch)
        intrinsics = predictions['intrinsics'] = self.calc_intrinsics('intrinsics', batch)

        if 'cams' in batch:
            cams = get_from_dict(batch, 'cams')
        else:
            cams = predictions['cams'] = self.calc_cams(rgb, predictions, 'pose', 'intrinsics')

        # Run all networks

        for key in self.networks.keys():
            if key.startswith('generic'):
                output = self.forward_generic(
                    key, batch, predictions, extra)
                update_predictions(batch, predictions, losses, extra, output, key)
            elif key.startswith('rgb'):
                output = self.forward_rgb(
                    key, rgb, depth, intrinsics, pose, cams, filename)
                update_predictions(batch, predictions, losses, extra, output, key)
            elif key.startswith('depth'):
                output = self.forward_depth(
                    key, rgb, depth, intrinsics, pose, cams, filename)
                update_predictions(batch, predictions, losses, extra, output, key)
            if key.startswith('multi_depth'):
                output = self.forward_multi_depth(
                    key, rgb, cams)
                update_predictions(batch, predictions, losses, extra, output, key)
            elif key.startswith('normals'):
                output = self.forward_normals(
                    key, rgb, intrinsics)
                update_predictions(batch, predictions, losses, extra, output, key)
            elif key.startswith('scene_flow'):
                output = self.forward_scene_flow(
                    key, rgb, intrinsics)
                update_predictions(batch, predictions, losses, extra, output, key)
            elif key.startswith('optical_flow'):
                output = self.forward_optical_flow(
                    key, rgb)
                update_predictions(batch, predictions, losses, extra, output, key)
            elif key.startswith('perceiver'):
                output = self.forward_perceiver(
                    key, rgb, depth, scene, cams, tag)
                update_predictions(batch, predictions, losses, extra, output, key)
            elif key.startswith('nsff'):
                output = self.forward_nerf(
                    key, rgb, depth, optflow, scnflow, valid_motion, valid_optflow,
                    mask_motion, timestep, cams)
                update_predictions(batch, predictions, losses, extra, output, key)

        # Return predictions and losses
        return remove_nones_dict(predictions), losses, extra

################################################

