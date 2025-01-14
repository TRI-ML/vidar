# Copyright 2024 Toyota Motor Corporation.  All rights reserved. 
# The implementation is derived from TRI-VIDAR -- 2023 Toyota Research Institute.

from abc import ABC
from typing import Any, Dict, List, Tuple
from functools import partial
import torch

from camviz import Draw
from camviz.objects.camera import Camera as CVCam

from vidar.arch.blocks.image.ViewSynthesis import ViewSynthesis
from vidar.arch.models.BaseModel import BaseModel
from vidar.arch.networks.layers.selffsm.metrics import cam_norm
from vidar.geometry.camera import Camera
from vidar.geometry.pose import Pose
from vidar.geometry.pose_utils import get_scaled_translation
from vidar.utils.config import Config
from vidar.utils.data import get_from_dict
from vidar.utils.distributed import print0
from vidar.utils.logging import pcolor
from vidar.utils.viz import viz_depth
from vidar.utils.types import is_list


def _print_method(text: str, font=None, prefix: str = ''):
    """ Customized print method to (1) specify the GPU:0 to run and (2) colorization."""
    print_as = font
    if font is not None:
        return print0(pcolor(prefix + text, **print_as))
    else:
        return print0(prefix + text)


class SelfSupervisedVOModel(BaseModel, ABC):
    """ Self-supervised depth and ego-motion model for Visual Odometry.
        Please refer to the https://arxiv.org/abs/2112.03325 for the intrinsic learning integration.

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """

    printW = partial(_print_method, font={'color': 'yellow'}, prefix='%%% [WARN] ')
    printI = partial(_print_method, font=None, prefix='%%% [INFO] ')
    printR = partial(_print_method, font={'color': 'blue'}, prefix='% [RESULT] ')

    IMPLEMENTED_LOSSES = ('reprojection', 'smoothness', 'supervision')
    IMPLEMENTED_NETWORKS = ('depth', 'pose', 'intrinsics')

    def __init__(self, cfg):
        """
        Configure SelfSupervisedVOModel for;
        - Ego-motion learning
        - Intrinsic self-calibration learning
        - Pose scaling ---We STRONGLY recommend raising this flag for fine-tuning by feeding `checkpoint: ....ckpt`
            (Please refer to https://arxiv.org/pdf/2308.02153 for its technical discussion)
        - Visualization based by CamViz, such as
            ```
            display:
                keys_for_visualization: [[0,0]] # key for image
                viz_wh: [640, 384] # image size to view
            ```
        """
        super().__init__(cfg)
        self.view_synthesis = ViewSynthesis()
        self.tgt_key = 0 if cfg.model.has('broken_key') and not cfg.model.broken_key else (0, 0)
        self.broken = cfg.model.has("broken_key", True)

        # Define ego-motion learning
        self.set_attr(cfg.model, 'use_gt_pose', False)
        self.printW("Network `pose` will be ignored because of `use_gt_pose") if self.use_gt_pose and cfg.networks.has(
            'pose', False) else True

        # Define intrinsic self-calibration
        self.set_attr(cfg.model, 'use_gt_intrinsics', True)
        self.printW("Network `intrinsics` will be ignored because of `use_gt_intrinsics`") if (
                self.use_gt_intrinsics and cfg.networks.has('intrinsics', False)) else True
        if cfg.networks.has('intrinsics', False):
            self.camera_geometry = cfg.networks.intrinsics.has('camera_model', 'ucm')
            self.printI('camera_model: {}'.format(self.camera_geometry))
        else:
            self.camera_geometry = 'pinhole'

        # Pose scaling
        self.set_attr(cfg.model, 'pose_scaling', False)
        self.printI("pose_scaling is True") if self.pose_scaling else True
        if self.pose_scaling:
            assert self.broken, "Set `broken: True` for pose_scaling mode"

        # For visualization
        self.display = cfg.model.has('display', False)
        self.viz_wh = None
        self.window_scale = None
        self.draw = None if not self.display else \
            self._display_start(cfg)
        self.set_attr(cfg.model, 'print_intrinsics', False)
        self.is_val_first_loop = None

    def _display_start(self, cfg: Config):
        """  Setup visualization by CamViz """
        self.printW("Predicted intrinsics is NOT SUPPORTED for Viz")
        self.viz_wh = cfg.model.display.has("viz_wh", [320, 192])
        self.window_scale = cfg.model.display.has("window_scale", 0.5)
        self.keys_for_visualization = self._list2tup_elem(cfg.model.display.has("keys_for_visualization", [[0, 0]]))
        self.printI("self.keys_for_visualization: {}".format(self.keys_for_visualization))

        draw = Draw(wh=(1200, 800), title='CamViz Pointcloud Demo')
        draw.setSize((int(draw.wh[0] * self.window_scale), int(draw.wh[1] * self.window_scale)))
        # Create image screen to show the RGB image
        draw.add2Dimage('rgb', luwh=(0.00, 0.00, 0.50, 0.50), res=self.viz_wh)
        # Create image screen to show the depth visualization
        draw.add2Dimage('dep', luwh=(0.00, 0.50, 0.50, 1.00), res=self.viz_wh)
        # Create world screen at specific position inside the window (% left/up/right/down)
        draw.add3Dworld('wld', luwh=(0.50, 0.00, 1.00, 1.00),
                        pose=(-1.92221, -18.34873, -29.71936, 0.98062, 0.19322, -0.03208, -0.00390)
                        )
        return draw

    def _display_loop(self, rgb, cams, depth: Dict):
        """Update the visualized depth to monitor per forward loop"""
        cvcams = {key: CVCam.from_vidar(val, b=0) for key, val in cams.items()}
        points = {}
        get_depth = partial(self._get_max_resolution_depth, arg_depth=depth)
        for key in self.keys_for_visualization:
            points[key] = cams[key].reconstruct_depth_map(
                get_depth(arg_key=key), to_world=False)
            viz_batch_id = 0
            for i, key in enumerate(self.keys_for_visualization):
                dep_out = get_depth(arg_key=key)
                self.draw.addTexture('rgb', rgb[key][viz_batch_id])
                self.draw.addTexture('dep', viz_depth(dep_out[viz_batch_id]))
                self.draw.addBufferf('pts', points[key][viz_batch_id])
                self.draw.addBufferf('clr', rgb[key][viz_batch_id])
        self.draw.clear()

        for i, key in enumerate(self.keys_for_visualization):
            self.draw['rgb'].image('rgb')
            self.draw['dep'].image('dep')
            self.draw['wld'].size(2).points('pts', 'clr')
            self.draw['wld'].object(cvcams[key], tex='dep')
        self.draw.update(30)

    def _get_max_resolution_depth(self, arg_depth: Any, arg_key: Any) -> torch.Tensor:
        """Return predicted depth with max resolution, [b,1,h,w]"""
        return arg_depth[arg_key][0] if is_list(arg_depth[arg_key]) else arg_depth[arg_key]

    def forward(self, batch, epoch=0) -> Dict[str, Any]:
        """ Forward wrapped-predictions (with loss and metric calculation if self.training is True)  """

        predictions = {}

        rgb = batch['rgb']
        if self.use_gt_intrinsics:
            intrinsics = get_from_dict(batch, 'intrinsics')
        else:
            intrinsics = {ctx: self.networks['intrinsics'](rgb[ctx]) for ctx in rgb.keys()}
            predictions.update({'intrinsics': {self.tgt_key: intrinsics[self.tgt_key]}})

        valid_mask = get_from_dict(batch, 'mask')
        gt_pose = get_from_dict(batch, 'pose')

        depth_output = self.networks['depth'](rgb=rgb[self.tgt_key],
                                              intrinsics=intrinsics[self.tgt_key] if intrinsics is not None else None)
        pred_depth = depth_output['depths']
        predictions.update({'depth': {self.tgt_key: pred_depth}})

        pred_logvar = get_from_dict(depth_output, 'logvar')
        if pred_logvar is not None:
            predictions['logvar'] = {self.tgt_key: pred_logvar}

        pose_pred = None
        if 'pose' in self.networks:
            pose_pred = self.forward_pose(rgb=rgb, tgt=self.tgt_key, broken=self.broken,
                                          gt_pose=gt_pose)
            predictions.update({"pose": pose_pred})
        if self.use_gt_pose:
            pose_pred = gt_pose

        if not self.training:
            if not self.use_gt_intrinsics and self.is_val_first_loop and self.print_intrinsics:
                print0()
                self.printR('(fx,fy,cx,cy,alpha(optional))={}'.format(intrinsics[self.tgt_key][0].cpu().numpy()))
                print0()
                self.is_val_first_loop = False
            return {
                'predictions': predictions,
            }

        pose = Pose.from_dict(pose_pred, zero_origin=True, to_global=True, to_matrix=True, broken=self.broken)
        cams = {key: Camera(
            K=self._intrinsics2camera(intrinsics[key]),
            Twc=pose[key],
            hw=rgb[key],
            geometry=self.camera_geometry) for key in pose.keys()}

        gt_depth = None if 'depth' not in batch else batch['depth'][self.tgt_key]
        loss, metrics = self.compute_loss_and_metrics(
            rgb, predictions['depth'], cams, gt_depth=gt_depth,
            logvar=pred_logvar, valid_mask=valid_mask
        )

        if self.display:
            gt_intrinsics_cams = {
                key: Camera(K=batch['intrinsics'][self.tgt_key], Twc=pose[key], hw=rgb[key], geometry='pinhole') for key
                in pose.keys()}
            self._display_loop(rgb=rgb, cams=gt_intrinsics_cams, depth=predictions['depth'])
            pass

        self.is_val_first_loop = True

        return {
            'loss': loss,
            'metrics': metrics,
            'predictions': predictions,
        }

    def forward_pose(self, rgb: Dict[Any, torch.Tensor], tgt: Any, invert=True, broken=True,
                     gt_pose: Dict[Any, torch.Tensor] = None) -> Dict[Any, torch.Tensor]:
        """
        Relative pose estimation

        Parameters
        ----------
        rgb : Dict
            Dictionary with input images [B,3,H,W]
        tgt : Tuple[int, int] or int
            (Temporal, Camera ID) or just index for Temporal
        invert : bool
            RGB inputs are sorted temporal order if True and combines with tgt[0] < ctx[0]
        broken : bool
            Flag for ``tgt'' type (tuple or int)
        gt_pose : Dict[Any, torch.Tensor]
            Pose reference for scaling

        Returns
        -------
        Dict[Any, torch.Tensor]
            Predicted ego-motion (Note that pose of the target frame is forcibly Identity Matrix)
        """
        pose_network = self.networks['pose']
        output_key2pose = {}
        if broken:
            ctxs: List[tuple, torch.Tensor] = [adj_ctx for adj_ctx in rgb.keys() if adj_ctx != tgt]
            pred_poses = {ctx: pose_network(
                [rgb[tgt[0], ctx[1]], rgb[ctx]], invert=(tgt[0] < ctx[0]) and invert)['transformation']
                          for ctx in ctxs}
        else:
            pred_poses = self.compute_pose(rgb, pose_network, tgt=tgt, invert=True)
        if self.pose_scaling:
            for ctx, tensor in pred_poses.items():  # Tuple. torch.Tensor
                posenet_norm = cam_norm(tensor)
                scale_from_gt = cam_norm(gt_pose[ctx])  # (B, )
                output_key2pose[ctx] = get_scaled_translation(transformation=tensor,
                                                              multiply=scale_from_gt / posenet_norm)
                pass
            pass
        else:
            output_key2pose = pred_poses
        ref_tensor = rgb[tgt]
        output_key2pose[tgt] = torch.eye(4, requires_grad=False,  # Set NOT trainable canonical pose
                                         device=ref_tensor.device).repeat([ref_tensor.shape[0], 1, 1])
        return output_key2pose

    def compute_loss_and_metrics(self, rgb: Dict[Any, torch.Tensor], depths: Dict[Any, List[torch.Tensor]],
                                 cams: Dict[Any, Camera], gt_depth: torch.Tensor = None,
                                 logvar=None, valid_mask=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss and metrics for training (please refer to the vidar/arch/models/depth/SelfSupervisedModel.py)

        Parameters that differs from SelfSupervisedModel.py:
        ----------
        depths : Dict[Any, List[torch.Tensor]]

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            - loss is for backpropagation
            - metric is for monitoring the training
        """
        ctx = [key for key in rgb.keys() if key != self.tgt_key]

        loss, metrics = [], {}
        depths: List[torch.Tensor] = depths[self.tgt_key]

        if 'reprojection' in self.losses:
            synth = self.view_synthesis(
                rgb=rgb, ctxs=ctx, depths=depths, cams=cams, return_masks=True, tgt=self.tgt_key)
            reprojection_output = self.losses['reprojection'](
                rgb[self.tgt_key], rgb, synth['warps'], logvar=logvar,
                valid_mask=valid_mask, overlap_mask=synth['masks'])
            loss.append(reprojection_output['loss'])
            metrics.update(**reprojection_output['metrics'])
        if 'smoothness' in self.losses:
            smoothness_output = self.losses['smoothness'](rgb[self.tgt_key], depths)
            loss.append(smoothness_output['loss'])
            metrics.update(**smoothness_output['metrics'])
        if 'supervision' in self.losses and gt_depth is not None:
            supervision_output = self.losses['supervision'](depths, gt_depth)
            loss.append(supervision_output['loss'])
            metrics.update(**supervision_output['metrics'])
        loss = sum(loss)

        return loss, metrics

    def _intrinsics2camera(self, param: torch.Tensor) -> torch.Tensor:
        """
        Reshape IntrinsicNet output to be compatible with the Camera instance following the geometry.

        Parameters
        ----------
        param : torch.Tensor
            [b,dim] of intrinsic parameter or [b,3,3] when it is from GT intrinsics

        Returns
        -------
        torch.Tensor
        - K matrix for the pinhole model
        - flatten parameters for ucm model
        """
        if self.camera_geometry == 'pinhole':
            if self.use_gt_intrinsics:
                return param
            else:
                b = param.shape[0]
                fx, fy, cx, cy = param[:, 0], param[:, 1], param[:, 2], param[:, 3]
                ret = torch.eye(3, device=param.device, dtype=param.dtype, requires_grad=False).repeat(b, 1, 1)
                ret[:, 0, 0] = fx
                ret[:, 0, 2] = cx
                ret[:, 1, 1] = fy
                ret[:, 1, 2] = cy
                return ret
        elif self.camera_geometry == 'ucm':
            return param
        else:
            raise NotImplementedError()

    @staticmethod
    def _list2tup_elem(lst: List[list]) -> List[Any]:
        """double-list to List of tuples to make it as key"""
        return [tuple(val) if len(val) == 2 else val[0] for val in lst]
