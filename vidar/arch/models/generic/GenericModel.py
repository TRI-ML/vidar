# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC
from functools import partial

import torch

from knk_vision.vidar.vidar.arch.models.BaseModel import BaseModel
from knk_vision.vidar.vidar.arch.models.generic.GenericModel_losses import GenericModelLosses
from knk_vision.vidar.vidar.arch.models.generic.GenericModel_predictions import GenericModelPredictions
from knk_vision.vidar.vidar.arch.models.generic.GenericModel_utils import \
    filter_targets, filter_contexts, filter_encodes, filter_losses, get_if_not_none, \
    filter_params, sum_valid
from knk_vision.vidar.vidar.arch.models.generic.GenericModel_warps import GenericModelWarps
from knk_vision.vidar.vidar.geometry.camera import Camera
from knk_vision.vidar.vidar.geometry.pose import Pose
from knk_vision.vidar.vidar.utils.augmentations import augment_batch1, augment_batch2
from knk_vision.vidar.vidar.utils.config import Config
from knk_vision.vidar.vidar.utils.data import get_from_dict, update_dict_nested, all_in
from knk_vision.vidar.vidar.utils.flow import residual_scene_flow, residual_scene_flow_from_depth_optflow, to_world_scene_flow
from knk_vision.vidar.vidar.utils.types import is_list


class GenericModel(BaseModel, ABC, GenericModelLosses, GenericModelWarps, GenericModelPredictions):
    """Generic model for multi-model inputs and multi-task output

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        GenericModelLosses.__init__(self, cfg)
        GenericModelWarps.__init__(self, cfg)

        self.base_cam = (0, 0)

        # Prepare display window (requires Camviz)
        if cfg.model.has('display') and cfg.model.display.enabled:
            from knk_vision.vidar.vidar.arch.models.generic.GenericModel_display import Display
            self.display = Display(cfg.model.display)
            self.display_mode = cfg.model.display.mode
        else:
            self.display = self.display_mode = None

        # Set batch prepare parameters
        self.prepare_cfg = Config(
            scene_flow=cfg.model.prepare.has('scene_flow', None) if cfg.model.has('prepare') else None,
            depth_to_z=cfg.model.prepare.has('depth_to_z', False) if cfg.model.has('prepare') else False,
            depth_to_e=cfg.model.prepare.has('depth_to_e', False) if cfg.model.has('prepare') else False,
        )
        self.augmentation_cfg = cfg.model.has('augmentation', None)

        # Parse configuration file for processing 
        self.task_cfg = {}
        if cfg.model.has('params'):
            for key in cfg.model.params.dict.keys():
                cfg_key = cfg.model.params.dict[key]
                for _ in cfg.model.params.dict.keys():
                    self.task_cfg[key] = Config(
                        # Key filtering
                        targets_train=
                        partial(filter_targets, params=cfg_key.targets_train) if cfg_key.has('targets_train') else
                        partial(filter_targets, params=cfg_key.targets) if cfg_key.has('targets') else None,
                        targets_val=
                        partial(filter_targets, params=cfg_key.targets_val) if cfg_key.has('targets_val') else
                        partial(filter_targets, params=cfg_key.targets) if cfg_key.has('targets') else None,
                        contexts_train=
                        partial(filter_contexts, params=cfg_key.contexts_train) if cfg_key.has('contexts_train') else
                        partial(filter_contexts, params=cfg_key.contexts) if cfg_key.has('contexts') else None,
                        contexts_val=
                        partial(filter_contexts, params=cfg_key.contexts_val) if cfg_key.has('contexts_val') else
                        partial(filter_contexts, params=cfg_key.contexts) if cfg_key.has('contexts') else None,
                        encodes_train=
                        partial(filter_encodes, params=cfg_key.encodes_train) if cfg_key.has('encodes_train') else
                        partial(filter_encodes, params=cfg_key.encodes) if cfg_key.has('encodes') else None,
                        encodes_val=
                        partial(filter_encodes, params=cfg_key.encodes_val) if cfg_key.has('encodes_val') else
                        partial(filter_encodes, params=cfg_key.encodes) if cfg_key.has('encodes') else None,
                        losses=
                        partial(filter_losses, params=cfg_key.losses) if cfg_key.has('losses') else None,
                        # Miscellaneous keys
                        use_gt=cfg_key.has('use_gt', True),
                        use_temporal=cfg_key.has('use_temporal', False),
                        geometry=cfg_key.has('geometry', 'pinhole'),
                        make_global=cfg_key.has('make_global', True),
                        invert_order=cfg_key.has('invert_order', True),
                        zero_origin=cfg_key.has('zero_origin', True),
                        calc_bidir=cfg_key.has('calc_bidir', False),
                        detach_depth=cfg_key.has('detach_depth', False),
                        use_pose=cfg_key.has('use_pose', True),
                        inputs=cfg_key.has('inputs', ['rgb']),
                        # Warping information
                        warps=Config(
                            source_depth=cfg_key.warps.has('source_depth', None),
                            source_optical_flow=cfg_key.warps.has('source_optical_flow', None),
                            source_scene_flow=cfg_key.warps.has('source_scene_flow', None),
                            mask=cfg_key.warps.has('mask', False),
                            use_mask=cfg_key.warps.has('use_mask', 1e10),
                            depth=cfg_key.warps.has('depth', False),
                            reverse=cfg_key.warps.has('reverse', False),
                            triangulation=cfg_key.warps.has('triangulation', False),
                            scene_flow=cfg_key.warps.has('scene_flow', False),
                        ) if cfg_key.has('warps') else None,
                    )

################################################

    def get_losses(self, cfg, tgts, keys):
        """Get loss parameters from configuration"""
        keys = [key for key in keys]
        return cfg.losses(tgts, keys) if cfg.losses is not None else keys

    def get_targets(self, cfg, tgts, keys):
        """Get target parameters from configuration"""
        keys = [key for key in keys]
        return cfg.targets_train(tgts, keys) if self.training else cfg.targets_val(tgts, keys)

    def get_contexts(self, cfg, tgts, keys):
        """Get context parameters from configuration"""
        keys = [key for key in keys]
        return cfg.contexts_train(tgts, keys) if self.training else cfg.contexts_val(tgts, keys)

    def get_encodes(self, cfg, tgts, keys):
        """Get encode parameters from configuration"""
        keys = [key for key in keys]
        return cfg.encodes_train(tgts, keys) if self.training else cfg.encodes_val(tgts, keys)

    def get_task_key(self, pred_key, task):
        """Get task key parameters from configuration"""
        suffix = pred_key.replace(f'{task}_', '')
        return pred_key if pred_key in self.task_cfg else suffix if suffix in self.task_cfg else task

    def get_task_cfg(self, pred_key, task):
        """Get task configuration parameters from configuration"""
        return get_if_not_none(self.task_cfg, self.get_task_key(pred_key, task))

################################################

    def calc_pose(self, task_key, batch, tgt=(0, 0)):
        """Calculate camera motion from batch information given a target key"""
        task_cfg = get_from_dict(self.task_cfg, task_key)
        if task_cfg is None:
            return None

        if task_cfg.use_gt is True:
            # Get GT pose if available
            pose = get_from_dict(batch, 'pose')
        else:
            # Use GT or predicted pose
            if task_cfg.use_gt is False:
                pose = {}
            else:
                gt_keys = filter_params(tgt, batch['pose'].keys(), task_cfg.use_gt)
                pose = {key: val for key, val in batch['pose'].items() if key in gt_keys}
            rgb = get_from_dict(batch, 'rgb')
            # Calculate pose from RGB
            if task_key in self.networks:
                ctxs = [key for key in rgb.keys() if key not in pose and key != tgt]
                pose.update(self.forward_pose(task_key, rgb, tgt=tgt, ctxs=ctxs, invert=task_cfg.invert_order))
            # Target pose is identity if not provided
            if tgt not in pose.keys():
                b, dtype, device = rgb[tgt].shape[0], rgb[tgt].dtype, rgb[tgt].device
                pose[tgt] = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(b, 1, 1)
        return pose

    def calc_intrinsics(self, task_key, batch, tgts=((0, 0),)):
        """Calculate camera intrinsics from batch information given a target key"""
        task_cfg = get_from_dict(self.task_cfg, task_key)
        if task_cfg is None:
            return None

        if task_cfg.use_gt:
            # Use GT intrinsics if available
            intrinsics = get_from_dict(batch, 'intrinsics')
        elif 'intrinsics' in self.networks:
            # Calculate intrinsics from RGB
            rgb = get_from_dict(batch, 'rgb')
            intrinsics = self.forward_intrinsics(task_key, rgb)
        else:
            intrinsics = None
        return intrinsics

    def calc_cams(self, rgb, data, pose_key, intrinsics_key, from_batch=False):
        """Produces camera from batch information given pose and intrinsic keys"""
        pose_cfg = get_from_dict(self.task_cfg, pose_key)
        intrinsics_cfg = get_from_dict(self.task_cfg, intrinsics_key)
        if pose_cfg is None or intrinsics_cfg is None:
            return None
        if from_batch and (not pose_cfg.use_gt or not intrinsics_cfg.use_gt):
            return None

        # Return None if intrinsics are not available 
        intrinsics = get_from_dict(data, intrinsics_key)
        if intrinsics is None:
            return None

        # Get pose from data dictionary
        pose = get_from_dict(data, pose_key)

        # Prepare pose from data dictionary
        if pose is not None:
            pose = Pose.from_dict(pose,
                zero_origin=pose_cfg.zero_origin,
                to_global=pose_cfg.make_global,
                to_matrix=True, broken=True)
        elif pose_cfg.zero_origin:
            pose = {key: None for key in rgb.keys()}
        else:
            return None

        # Parse pose and intrinsics to return camera
        return {
            key: Camera(
                K=intrinsics[key if key in intrinsics else (0, key[1])], 
                Twc=pose[key],
                hw=rgb[key],
                geometry=intrinsics_cfg.geometry,
            ) for key in pose.keys()
        }

    def prepare_batch(self, batch, pose_key, intrinsics_key):
        """Prepare batch before processing (augmentations, transformations, etc)"""
        if 'prepared' in batch and batch['prepared'] is True:
            return batch

        # Augment batch if requested
        batch = augment_batch1(batch, self.augmentation_cfg, self.training)

        # Prepare cameras if not there
        if 'cams' not in batch:
            cams = self.calc_cams(batch['rgb'], batch, pose_key, intrinsics_key, from_batch=True)
            if cams is not None:
                batch['cams'] = cams

        # Convert depth from euclidean to zbuffer and vice-versa
        if self.prepare_cfg.depth_to_z:
            batch['depth'] = {key: batch['cams'][key].e2z(batch['depth'][key]) for key in batch['cams'].keys()}
        if self.prepare_cfg.depth_to_e:
            batch['depth'] = {key: batch['cams'][key].z2e(batch['depth'][key]) for key in batch['cams'].keys()}

        # Augment batch if requested
        batch = augment_batch2(batch, self.augmentation_cfg, self.training)

        # Prepare different types of scene flow and optical flow
        if self.prepare_cfg.has('scene_flow') and self.prepare_cfg.scene_flow is not None:
            if 'residual' in self.prepare_cfg.scene_flow:
                if all_in(batch, 'scene_flow', 'depth'):
                    for tgt in batch['scene_flow'].keys():
                        for ctx in batch['scene_flow'][tgt].keys():
                            batch['scene_flow'][tgt][ctx] = residual_scene_flow(
                                batch['depth'][tgt], batch['scene_flow'][tgt][ctx],
                                batch['cams'][tgt].relative_to(batch['cams'][ctx]))
            elif 'optflow' in self.prepare_cfg.scene_flow:
                if all_in(batch, 'optical_flow', 'depth'):
                    batch['scene_flow'] = {}
                    for tgt in batch['optical_flow'].keys():
                        for ctx in batch['optical_flow'][tgt].keys():
                            if (tgt not in batch['scene_flow'] or ctx not in batch['scene_flow'][tgt]) and \
                                    (ctx not in batch['scene_flow'] or tgt not in batch['scene_flow'][ctx]):
                                scnflow21, scnflow12 = residual_scene_flow_from_depth_optflow(
                                    batch['depth'][tgt], batch['depth'][ctx], batch['cams'][tgt], batch['cams'][ctx],
                                    batch['optical_flow'][ctx][tgt], batch['optical_flow'][tgt][ctx],
                                )
                                update_dict_nested(batch, 'scene_flow', tgt, ctx, scnflow12)
                                update_dict_nested(batch, 'scene_flow', ctx, tgt, scnflow21)
            if 'world' in self.prepare_cfg.scene_flow:
                if all_in(batch, 'scene_flow', 'depth'):
                    for tgt in batch['scene_flow'].keys():
                        for ctx in batch['scene_flow'][tgt].keys():
                            batch['scene_flow'][tgt][ctx] = to_world_scene_flow(
                                batch['cams'][ctx], batch['depth'][ctx], batch['scene_flow'][tgt][ctx])

        # Set flag so that batch is not prepared again
        batch['prepared'] = True

        return batch

################################################

    def forward(self, batch, epoch=0):
        """Model forward pass"""

        if is_list(batch) and self.training:
            outputs = [self.forward(b, epoch=epoch) for b in batch]
            return {
                'loss': sum([out['loss'] for out in outputs]),
                'metrics': outputs[0]['metrics'],
                'predictions': outputs[0]['predictions'],
            }

        # Prepare and warp batch
        batch = self.prepare_batch(batch, 'pose', 'intrinsics')
        self.forward_warps(batch)

        # Predict and warp batch
        predictions, losses_predictions, extra_predictions = self.forward_predictions(batch, epoch=epoch)
        self.forward_warps(batch, predictions)

        # Display options
        if self.display_mode == 'gt':
            self.display.loop_gt(batch)
        if self.display_mode == 'perceiver':
            self.display.loop_perceiver(batch, predictions, extra_predictions)
        if self.display_mode == 'pred':
            self.display.loop_pred(batch, predictions, extra_predictions)
        if self.display_mode == 'sandbox':
            self.display.loop_sandbox(batch, predictions, extra_predictions)
        if self.display_mode == 'train':
            self.display.loop_train(batch, predictions, extra_predictions)

        # Return only predictions if not training
        if not self.training:
            return {
                'predictions': predictions,
                'batch': batch,
            }

        # Calculate and prepare losses
        losses, metrics, extra_losses = self.forward_losses(
            batch, predictions, extra=extra_predictions, epoch=epoch)
        loss = sum_valid({**losses_predictions, **losses})
        metrics = {**metrics, **losses_predictions}

        # Return predictions and losses
        return {
            'loss': loss,
            'metrics': metrics,
            'predictions': predictions,
            'batch': batch,
        }

################################################
