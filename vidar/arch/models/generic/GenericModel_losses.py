# Copyright 2023 Toyota Research Institute.  All rights reserved.

from vidar.arch.blocks.image.ViewSynthesis import ViewSynthesis
from vidar.arch.models.generic.GenericModel_utils import get_if_not_none, apply_masks, dense_batches, \
    sum_valid, get_gt, get_mask_valid, apply_loss, update_losses, sample_supervision, valid_batches
from vidar.utils.config import Config
from vidar.utils.data import get_from_dict, strs_not_in_key, sum_list, make_list, detach_dict
from vidar.utils.depth import calculate_normals
from vidar.utils.types import is_dict, is_list


def apply_valid(data, value):
    return None if data is None else [d[value] for d in data] if is_list(data) else data[value]


class GenericModelLosses:
    """Generic model for multi-model inputs and multi-task output. Subclass focused on loss functionality.

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """    
    def __init__(self, cfg):

        self.view_synthesis = ViewSynthesis()

        self.losses_cfg = {}
        if cfg.has('losses'):
            for key1 in cfg.losses.dict.keys():
                self.losses_cfg[key1] = {}
                for key2 in cfg.losses.dict[key1].dict.keys():
                    loss_cfg = cfg.losses.dict[key1].dict[key2]
                    self.losses_cfg[key1][key2] = Config(
                        gt=loss_cfg.has('gt', 'gt'),
                        masks=make_list(loss_cfg.has('masks', None)),
                        masks_epoch=make_list(loss_cfg.has('masks_epoch', None)),
                        apply_to=make_list(loss_cfg.has('apply_to', [])),
                    )

################################################

    def loss_depth_self_supervision(self, pred_key, loss_fn, rgb, depth, cams, predictions,
                                    mask=None, task='depth', epoch=None):
        """Calculate self-supervised depth loss."""
        task_cfg = self.get_task_cfg(pred_key, task)
        if task_cfg is None: return
        loss, metrics = [], []
        warps, masks, photo = {}, {}, {}

        tgts = self.get_losses(task_cfg, self.base_cam, depth)
        cams_tgt = get_if_not_none(predictions, pred_key.replace(task, 'cams'), cams)
        cams = [cams_tgt, cams]

        for tgt in tgts:

            depth_tgt = get_from_dict(depth, tgt)
            rgb_tgt = sample_supervision(pred_key, task, predictions, rgb, tgt)
            mask_tgt = sample_supervision(pred_key, task, predictions, mask, tgt)
            ctxs = self.get_contexts(task_cfg, tgt, rgb)
            rgb_ctx = {ctx: rgb[ctx] for ctx in ctxs}

            if is_dict(depth_tgt):
                warps[tgt], masks[tgt], photo[tgt] = {}, {}, {}
                for key in depth_tgt.keys():
                    depth_tgt_key = get_from_dict(depth_tgt, key)
                    mask_tgt_key = get_from_dict(mask_tgt, key)

                    synth = self.view_synthesis(
                        tgt, ctxs, rgb=rgb_ctx, depths=depth_tgt_key, cams=cams, return_masks=True)
                    synth['masks'] = apply_masks(synth['masks'], mask_tgt_key)
                    reprojection_output = loss_fn(rgb_tgt, rgb_ctx, synth['warps'], overlap_mask=synth['masks'])

                    loss.append(reprojection_output['loss'])
                    metrics.append(reprojection_output['metrics'])

                    warps[tgt][key] = synth['warps']
                    masks[tgt][key] = synth['masks']
                    photo[tgt][key] = reprojection_output['photo']

            else:

                synth = self.view_synthesis(
                    tgt, ctxs, rgb=rgb_ctx, depths=depth_tgt, cams=cams, return_masks=True)
                synth['masks'] = apply_masks(synth['masks'], mask_tgt)
                reprojection_output = loss_fn(rgb_tgt, rgb_ctx, synth['warps'], overlap_mask=synth['masks'])

                loss.append(reprojection_output['loss'])
                metrics.append(reprojection_output['metrics'])

                warps[tgt] = synth['warps']
                masks[tgt] = synth['masks']
                photo[tgt] = reprojection_output['photo']

        return {
            'loss': sum_list(loss),
            'metrics': sum_list(metrics),
            'extras': {
                'warps_depth': warps,
                'masks_depth': masks,
                'photo_depth': photo,
            }
        }

################################################

    def loss_depth_smoothness(self, pred_key, loss_fn, rgb, depth,
                              predictions, mask=None, task='depth', epoch=None):
        """Calculate smoothness depth loss."""
        task_cfg = self.get_task_cfg(pred_key, task)
        if task_cfg is None: return
        loss, metrics = [], []

        tgts = self.get_losses(task_cfg, self.base_cam, depth)

        for tgt in tgts:

            depth_tgt = get_from_dict(depth, tgt)
            rgb_tgt = sample_supervision(pred_key, task, predictions, rgb, tgt)
            mask_tgt = sample_supervision(pred_key, task, predictions, mask, tgt)

            if is_dict(depth_tgt):
                for key in depth_tgt.keys():

                    depth_tgt_key = get_from_dict(depth_tgt, key)
                    rgb_tgt_key = get_from_dict(rgb_tgt, key)
                    mask_tgt_key = get_from_dict(mask_tgt, key)

                    smoothness_output = loss_fn(rgb_tgt_key, depth_tgt_key, mask_tgt_key)
                    loss.append(smoothness_output['loss'])
                    metrics.append(smoothness_output['metrics'])

            else:

                smoothness_output = loss_fn(rgb_tgt, depth_tgt, mask=mask_tgt)
                loss.append(smoothness_output['loss'])
                metrics.append(smoothness_output['metrics'])

        return {
            'loss': sum_list(loss),
            'metrics': sum_list(metrics),
            'extras': {}
        }

################################################

    def loss_depth_supervision(self, pred_key, loss_fn, depth, gt_depth,
                               predictions, mask=None, logvar=None, task='depth', epoch=None):
        """Calculate supervised depth loss."""
        task_cfg = self.get_task_cfg(pred_key, task)
        if task_cfg is None: return
        loss, metrics = [], []

        tgts = self.get_losses(task_cfg, self.base_cam, depth)

        for tgt in tgts:

            depth_tgt = get_from_dict(depth, tgt)
            logvar_tgt = get_from_dict(logvar, tgt)
            gt_depth_tgt = sample_supervision(pred_key, task, predictions, gt_depth, tgt)
            mask_tgt = sample_supervision(pred_key, task, predictions, mask, tgt)

            if is_dict(depth_tgt):
                for key in depth_tgt.keys():

                    depth_tgt_key = get_from_dict(depth_tgt, key)
                    logvar_tgt_key = get_from_dict(logvar_tgt, key)
                    gt_depth_tgt_key = get_from_dict(gt_depth_tgt, key)
                    mask_tgt_key = get_from_dict(mask_tgt, key)

                    valid = valid_batches(gt_depth_tgt_key)
                    if len(valid) == 0:
                        continue
                    if len(valid) < gt_depth_tgt_key.shape[0]:
                        depth_tgt_key = apply_valid(depth_tgt_key, valid)
                        logvar_tgt_key = apply_valid(logvar_tgt_key, valid)
                        gt_depth_tgt_key = apply_valid(gt_depth_tgt_key, valid)
                        mask_tgt_key = apply_valid(mask_tgt_key, valid)

                    depth_supervision_output = loss_fn(
                        pred=depth_tgt_key, gt=gt_depth_tgt_key,
                        mask=mask_tgt_key, logvar=logvar_tgt_key,
                        epoch=epoch,
                    )

                    loss.append(depth_supervision_output['loss'])
                    metrics.append(depth_supervision_output['metrics'])

            else:

                valid = valid_batches(gt_depth_tgt)
                if len(valid) == 0:
                    continue
                if len(valid) < gt_depth_tgt.shape[0]:
                    depth_tgt = apply_valid(depth_tgt, valid)
                    logvar_tgt = apply_valid(logvar_tgt, valid)
                    gt_depth_tgt = apply_valid(gt_depth_tgt, valid)
                    mask_tgt = apply_valid(mask_tgt, valid)

                depth_supervision_output = loss_fn(
                    pred=depth_tgt, gt=gt_depth_tgt,
                    mask=mask_tgt, logvar=logvar_tgt,
                    epoch=epoch,
                )

                loss.append(depth_supervision_output['loss'])
                metrics.append(depth_supervision_output['metrics'])

        return {
            'loss': sum_list(loss),
            'metrics': sum_list(metrics),
            'extras': {},
        }

    def loss_depth_bins_supervision(self, pred_key, loss_fn, bins, z_vals, gt_depth,
                                    predictions, mask=None, logvar=None, task='bins', epoch=None):
        """Calculate bin-supervised depth loss."""
        task_cfg = self.get_task_cfg(pred_key, task)
        if task_cfg is None: return
        loss, metrics = [], []

        tgts = self.get_losses(task_cfg, self.base_cam, bins)

        for tgt in tgts:

            bins_tgt = get_from_dict(bins, tgt)
            z_vals_tgt = get_from_dict(z_vals, tgt)
            gt_depth_tgt = sample_supervision(pred_key, task, predictions, gt_depth, tgt)
            mask_tgt = sample_supervision(pred_key, task, predictions, mask, tgt)

            if is_dict(bins_tgt):
                pass

            else:

                from vidar.utils.depth import depth2index
                gt_bins_tgt = [depth2index(gt_depth_tgt, z_vals_tgt[i]) for i in range(len(z_vals_tgt))]

                depth_bins_supervision_output = loss_fn(
                    pred=bins_tgt, gt=gt_bins_tgt,
                    mask=mask_tgt, logvar=None,
                )

                loss.append(depth_bins_supervision_output['loss'])
                metrics.append(depth_bins_supervision_output['metrics'])

        return {
            'loss': sum_list(loss),
            'metrics': sum_list(metrics),
            'extras': {},
        }

################################################

    def loss_depth_normals_supervision(self, pred_key, loss_fn, depth, gt_depth, cams,
                                       predictions, mask=None, task='depth', epoch=None):
        """Calculate normal-supervised depth loss."""
        task_cfg = self.get_task_cfg(pred_key, task)
        if task_cfg is None: return
        loss, metrics = [], []

        cams = get_if_not_none(predictions, pred_key.replace(task, 'cams'), cams)
        tgts = self.get_losses(task_cfg, self.base_cam, depth)

        for tgt in tgts:

            depth_tgt = get_from_dict(depth, tgt)
            cams_tgt = get_from_dict(cams, tgt)
            gt_depth_tgt = sample_supervision(pred_key, task, predictions, gt_depth, tgt)
            mask_tgt = sample_supervision(pred_key, task, predictions, mask, tgt)

            if is_dict(depth_tgt):
                for key in depth_tgt.keys():

                    depth_tgt_key = get_from_dict(depth_tgt, key)
                    cams_tgt_key = get_from_dict(cams_tgt, key)
                    gt_depth_tgt_key = get_from_dict(gt_depth_tgt, key)
                    mask_tgt_key = get_from_dict(mask_tgt, key)

                    normals_tgt_key = [calculate_normals(d, camera=cams_tgt_key) for d in depth_tgt_key]
                    gt_normals_tgt = calculate_normals(gt_depth_tgt_key, camera=cams_tgt_key)

                    normals_output = loss_fn(normals_tgt_key, gt_normals_tgt, gt_depth_tgt_key, mask=mask_tgt_key)
                    loss.append(normals_output['loss'])
                    metrics.append(normals_output['metrics'])

            else:

                normals_tgt = [calculate_normals(d, camera=cams_tgt) for d in depth_tgt]
                gt_normals_tgt = calculate_normals(gt_depth_tgt, camera=cams_tgt)

                normals_output = loss_fn(normals_tgt, gt_normals_tgt, gt_depth_tgt, mask=mask_tgt)
                loss.append(normals_output['loss'])
                metrics.append(normals_output['metrics'])

        return {
            'loss': sum_list(loss, None),
            'metrics': sum_list(metrics, {}),
            'extras': {},
        }

################################################

    def loss_normals_supervision(self, pred_key, loss_fn, normals, gt_normals, cams,
                                 mask=None, task='normals', epoch=None):
        """Calculate supervised normals loss."""
        task_cfg = self.get_task_cfg(pred_key, task)
        if task_cfg is None: return

        loss, metrics = [], []

        tgts = self.get_losses(task_cfg, self.base_cam, normals)

        for tgt in tgts:

            normals_tgt = get_from_dict(normals, tgt)
            gt_normals_tgt = get_from_dict(gt_normals, tgt)
            cams_tgt = get_from_dict(cams, tgt)

            valid = dense_batches(gt_normals_tgt)
            if len(valid) == 0:
                continue
            if len(valid) < gt_normals_tgt.shape[0]:
                normals_tgt = apply_valid(normals_tgt, valid)
                gt_normals_tgt = apply_valid(gt_normals_tgt, valid)
                cams_tgt = apply_valid(cams_tgt, valid)

            normals_output = loss_fn(normals_tgt, gt_normals_tgt, cams_tgt)
            loss.append(normals_output['loss'])
            metrics.append(normals_output['metrics'])

        return {
            'loss': sum_valid(loss),
            'metrics': sum_valid(metrics),
            'extras': {},
        }

################################################

    def loss_optical_flow_self_supervision(self, pred_key, loss_fn, rgb, optflow, predictions,
                                           mask=None, task='optical_flow', epoch=None):
        """Calculate self-supervised optical flow loss."""
        task_cfg = self.get_task_cfg(pred_key, task)
        if task_cfg is None: return
        loss, metrics = [], []
        warps, masks, photo = {}, {}, {}

        tgts = self.get_losses(task_cfg, self.base_cam, optflow)

        for tgt in tgts:

            optflow_tgt = get_from_dict(optflow, tgt)
            mask_tgt = sample_supervision(pred_key, task, predictions, mask, tgt)
            rgb_tgt = sample_supervision(pred_key, task, predictions, rgb, tgt)
            ctxs = self.get_contexts(task_cfg, tgt, rgb)
            rgb_ctx = {ctx: rgb[ctx] for ctx in ctxs}

            synth = self.view_synthesis(
                tgt, ctxs, rgb=rgb_ctx, optical_flow=optflow_tgt, return_masks=True)
            synth['masks'] = apply_masks(synth['masks'], mask_tgt)
            reprojection_output = loss_fn(rgb_tgt, rgb_ctx, synth['warps'], overlap_mask=synth['masks'])

            loss.append(reprojection_output['loss'])
            metrics.append(reprojection_output['metrics'])

            warps[tgt] = synth['warps']
            masks[tgt] = synth['masks']
            photo[tgt] = reprojection_output['photo']

        return {
            'loss': sum_list(loss),
            'metrics': sum_list(metrics),
            'extras': {
                'warps_optflow': warps,
                'masks_optflow': masks,
                'photo_optflow': photo,
            }
        }

################################################

    def loss_optical_flow_supervision(self, pred_key, loss_fn, optflow, gt_optflow,
                                      predictions, mask=None, task='optical_flow', epoch=None):
        """Calculate supervised optical flow loss."""
        task_cfg = self.get_task_cfg(pred_key, task)
        if task_cfg is None: return
        loss, metrics = [], []

        tgts = self.get_losses(task_cfg, self.base_cam, optflow)

        for tgt in tgts:

            optflow_tgt = get_from_dict(optflow, tgt)
            gt_optflow_tgt = sample_supervision(pred_key, task, predictions, gt_optflow, tgt)
            mask_tgt = sample_supervision(pred_key, task, predictions, mask, tgt)

            for key in optflow_tgt.keys():
                optflow_tgt_key = get_from_dict(optflow_tgt, key)
                gt_optflow_tgt_key = get_from_dict(gt_optflow_tgt, key)
                mask_tgt_key = get_from_dict(mask_tgt, key)

                optflow_supervision_output = loss_fn(optflow_tgt_key, gt_optflow_tgt_key, mask_tgt_key)
                loss.append(optflow_supervision_output['loss'])
                metrics.append(optflow_supervision_output['metrics'])

        return {
            'loss': sum_list(loss),
            'metrics': sum_list(metrics),
            'extras': {},
        }

################################################

    def loss_scene_flow_self_supervision(self, pred_key, loss_fn, rgb, depth, scene_flow, cams, predictions,
                                    mask=None, task='scene_flow', epoch=None):
        """Calculate self-supervised scene flow loss."""
        task_cfg = self.get_task_cfg(pred_key, task)
        if task_cfg is None: return
        loss, metrics = [], []
        warps, masks, photo = {}, {}, {}

        if not task_cfg.use_pose:
            cams = {key: val.no_pose() for key, val in cams.items()}

        tgts = self.get_losses(task_cfg, self.base_cam, scene_flow)
        cams_tgt = get_if_not_none(predictions, pred_key.replace(task, 'cams'), cams)
        cams = [cams_tgt, cams]

        for tgt in tgts:

            depth_tgt = get_from_dict(depth, tgt)
            scene_flow_tgt = get_from_dict(scene_flow, tgt)
            rgb_tgt = sample_supervision(pred_key, task, predictions, rgb, tgt)
            mask_tgt = sample_supervision(pred_key, task, predictions, mask, tgt)
            ctxs = self.get_contexts(task_cfg, tgt, rgb)
            rgb_ctx = {ctx: rgb[ctx] for ctx in ctxs}

            synth = {'warps': {}, 'masks': {}}
            for ctx in ctxs:

                depth_tgt_key = get_from_dict(depth_tgt, ctx)
                scene_flow_tgt_key = get_from_dict(scene_flow_tgt, ctx)
                mask_tgt_key = get_from_dict(mask_tgt, ctx)

                if task_cfg.detach_depth:
                    depth_tgt_key = detach_dict(depth_tgt_key)

                synth_ctx = self.view_synthesis(
                    tgt, [ctx], rgb=rgb_ctx, depths=depth_tgt_key,
                    scene_flows=scene_flow_tgt_key, cams=cams, return_masks=True)
                synth['masks'] = apply_masks(synth['masks'], mask_tgt_key)
                synth['warps'][ctx] = synth_ctx['warps'][ctx]
                synth['masks'][ctx] = synth_ctx['masks'][ctx]

            reprojection_output = loss_fn(rgb_tgt, rgb_ctx, synth['warps'], overlap_mask=synth['masks'])

            loss.append(reprojection_output['loss'])
            metrics.append(reprojection_output['metrics'])

            warps[tgt] = synth['warps']
            masks[tgt] = synth['masks']
            photo[tgt] = reprojection_output['photo']

            from vidar.utils.write import write_image
            for ctx in ctxs:
                write_image(f'scene_flow_{tgt}_{ctx}.png', synth['warps'][ctx][0])
            write_image(f'scene_flow_{tgt}.png', rgb_tgt)

        return {
            'loss': sum_list(loss),
            'metrics': sum_list(metrics),
            'extras': {
                'warps_depth': warps,
                'masks_depth': masks,
                'photo_depth': photo,
            }
        }
        
################################################

    def loss_rgb_supervision(self, pred_key, loss_fn, rgb, gt_rgb,
                               predictions, mask=None, task='rgb', epoch=None):
        """Calculate supervised rgb loss."""
        task_cfg = self.get_task_cfg(pred_key, task)
        if task_cfg is None: return
        loss, metrics = [], []

        tgts = self.get_losses(task_cfg, self.base_cam, rgb)

        for tgt in tgts:

            rgb_tgt = get_from_dict(rgb, tgt)
            gt_rgb_tgt = sample_supervision(pred_key, task, predictions, gt_rgb, tgt)
            mask_tgt = sample_supervision(pred_key, task, predictions, mask, tgt)

            valid = valid_batches(gt_rgb_tgt)
            if len(valid) == 0:
                continue
            if len(valid) < gt_rgb_tgt.shape[0]:
                rgb_tgt = apply_valid(rgb_tgt, valid)
                gt_rgb_tgt = apply_valid(gt_rgb_tgt, valid)
                mask_tgt = apply_valid(mask_tgt, valid)

            rgb_supervision_output = loss_fn(
                pred=rgb_tgt, gt=gt_rgb_tgt, mask=mask_tgt)

            loss.append(rgb_supervision_output['loss'])
            metrics.append(rgb_supervision_output['metrics'])

        return {
            'loss': sum_list(loss),
            'metrics': sum_list(metrics),
            'extras': {},
        }

################################################

    def forward_losses(self, batch, predictions, extra=None, epoch=None):
        """Forward pass for predictions (calls specific forward passes as needed)"""

        losses, metrics, extras = {}, {}, {}

        rgb = get_from_dict(batch, 'rgb')

        if 'cams' in batch:
            cams = get_from_dict(batch, 'cams')
        else:
            cams = get_from_dict(predictions, 'cams')


        for loss_key in self.losses.keys():
            if loss_key.startswith('generic'):
                for sub_loss_key, loss_fn in self.losses[loss_key].items():
                    output = loss_fn(batch, predictions, extra, self.networks)
                    update_losses(losses, metrics, extras, output, loss_key, sub_loss_key)

        for pred_key in predictions.keys():
            for loss_key in self.losses.keys():

                ####### RGB LOSSES
                task = 'rgb'
                if pred_key.startswith(task) and (loss_key == task or loss_key == pred_key):
                    ### GET PREDICTIONS
                    rgb = get_from_dict(predictions, pred_key)
                    ### LOOP OVER SUB LOSSES
                    for sub_loss_key, loss_fn in self.losses[loss_key].items():
                        ### GET GROUND TRUTH AND MASKS
                        cfg = self.losses_cfg[loss_key][sub_loss_key]
                        rgb_gt = get_gt(cfg, batch, predictions, task, pred_key)
                        mask = get_mask_valid(cfg, batch, predictions)
                        if not apply_loss(pred_key, cfg.apply_to): continue
                        ### SUPERVISION
                        if sub_loss_key.startswith('supervision'):
                            output = self.loss_rgb_supervision(
                                pred_key, loss_fn, rgb, rgb_gt, predictions,
                                mask=mask, epoch=epoch)
                            update_losses(losses, metrics, extras, output, pred_key, sub_loss_key)

                ####### DEPTH LOSSES
                task = 'depth'
                if pred_key.startswith(task) and (loss_key == task or loss_key == pred_key) and \
                        strs_not_in_key(pred_key, ['triangulate']):
                    ### GET PREDICTIONS
                    depth = get_from_dict(predictions, pred_key)
                    logvar = get_from_dict(predictions, pred_key.replace(task, 'logvar'))
                    ### LOOP OVER SUB LOSSES
                    for sub_loss_key, loss_fn in self.losses[loss_key].items():
                        ### GET GROUND TRUTH AND MASKS
                        cfg = self.losses_cfg[loss_key][sub_loss_key]
                        if not apply_loss(pred_key, cfg.apply_to): continue
                        depth_gt = get_gt(cfg, batch, predictions, task, pred_key)
                        mask = get_mask_valid(cfg, batch, predictions)
                        ### SELF-SUPERVISION
                        if sub_loss_key.startswith('self_supervision'):
                            output = self.loss_depth_self_supervision(
                                pred_key, loss_fn, rgb, depth, cams, predictions,
                                mask=mask, epoch=epoch)
                            update_losses(losses, metrics, extras, output, pred_key, sub_loss_key)
                        ### SMOOTHNESS
                        if sub_loss_key.startswith('smoothness'):
                            output = self.loss_depth_smoothness(
                                pred_key, loss_fn, rgb, depth, predictions,
                                mask=mask, epoch=epoch)
                            update_losses(losses, metrics, extras, output, pred_key, sub_loss_key)
                        ### SUPERVISION
                        if sub_loss_key.startswith('supervision'):
                            output = self.loss_depth_supervision(
                                pred_key, loss_fn, depth, depth_gt, predictions,
                                mask=mask, logvar=logvar, epoch=epoch)
                            update_losses(losses, metrics, extras, output, pred_key, sub_loss_key)
                        ### NORMALS SUPERVISION
                        if sub_loss_key.startswith('normals_supervision'):
                            output = self.loss_depth_normals_supervision(
                                pred_key, loss_fn, depth, depth_gt, cams, predictions,
                                mask=mask, epoch=epoch)
                            update_losses(losses, metrics, extras, output, pred_key, sub_loss_key)

                ####### DEPTH bins LOSSES
                task = 'bins'
                if pred_key.startswith(task) and (loss_key == task or loss_key == pred_key):
                    ### GET PREDICTIONS
                    bins = get_from_dict(predictions, pred_key)
                    z_vals = get_from_dict(predictions, pred_key.replace('bins', 'zvals'))
                    ### LOOP OVER SUB LOSSES
                    for sub_loss_key, loss_fn in self.losses[loss_key].items():
                        ### GET GROUND TRUTH AND MASKS
                        cfg = self.losses_cfg[loss_key][sub_loss_key]
                        depth_gt = get_gt(cfg, batch, predictions, task.replace('bins', 'depth'), pred_key.replace('bins', 'depth'))
                        mask = get_mask_valid(cfg, batch, predictions)
                        if not apply_loss(pred_key, cfg.apply_to): continue
                        ### SUPERVISION
                        if sub_loss_key.startswith('supervision'):
                            output = self.loss_depth_bins_supervision(
                                pred_key, loss_fn, bins, z_vals, depth_gt, predictions,
                                mask=mask, logvar=None, epoch=epoch)
                            update_losses(losses, metrics, extras, output, pred_key, sub_loss_key)

                ####### NORMALS LOSSES
                task = 'normals'
                if pred_key.startswith(task) and (loss_key == task or loss_key == pred_key):
                    ### GET PREDICTIONS
                    normals = get_from_dict(predictions, pred_key)
                    ### LOOP OVER SUB LOSSES
                    for sub_loss_key, loss_fn in self.losses[loss_key].items():
                        ### GET GROUND TRUTH AND MASKS
                        cfg = self.losses_cfg[loss_key][sub_loss_key]
                        normals_gt = get_gt(cfg, batch, predictions, task, pred_key)
                        mask = get_mask_valid(cfg, batch, predictions)
                        if not apply_loss(pred_key, cfg.apply_to): continue
                        ### SUPERVISION
                        if sub_loss_key.startswith('supervision'):
                            output = self.loss_normals_supervision(
                                pred_key, loss_fn, normals, normals_gt, cams,
                                mask=mask, epoch=epoch)
                            update_losses(losses, metrics, extras, output, pred_key, sub_loss_key)

                ####### OPTICAL FLOW LOSSES
                task = 'optical_flow'
                if pred_key.startswith(task) and (loss_key == task or loss_key == pred_key) and \
                        strs_not_in_key(pred_key, ['reverse']):
                    ### GET PREDICTIONS
                    optical_flow = predictions[pred_key]
                    ### LOOP OVER SUB LOSSES
                    for sub_loss_key, loss_fn in self.losses[loss_key].items():
                        ### GET GROUND TRUTH AND MASKS
                        cfg = self.losses_cfg[loss_key][sub_loss_key]
                        optical_flow_gt = get_gt(cfg, batch, predictions, task, pred_key)
                        mask = get_mask_valid(cfg, batch, predictions)
                        if not apply_loss(pred_key, cfg.apply_to): continue
                        ### SELF-SUPERVISION
                        if sub_loss_key.startswith('self_supervision'):
                            output = self.loss_optical_flow_self_supervision(
                                pred_key, loss_fn, rgb, optical_flow, predictions,
                                mask=mask, epoch=epoch)
                            update_losses(losses, metrics, extras, output, pred_key, sub_loss_key)
                        ### SUPERVISION
                        if sub_loss_key.startswith('supervision'):
                            output = self.loss_optical_flow_supervision(
                                pred_key, loss_fn, optical_flow, optical_flow_gt, predictions,
                                mask=mask, epoch=epoch)
                            update_losses(losses, metrics, extras, output, pred_key, sub_loss_key)

                ####### SCENE FLOW LOSSES
                task = 'scene_flow'
                if pred_key.startswith(task) and (loss_key == task or loss_key == pred_key) and \
                        strs_not_in_key(pred_key, ['triangulate']):
                    ### GET PREDICTIONS
                    depth = get_from_dict(predictions, pred_key.replace(task, 'depth'))
                    scene_flow = get_from_dict(predictions, pred_key)
                    ### LOOP OVER SUB LOSSES
                    for sub_loss_key, loss_fn in self.losses[loss_key].items():
                        ### GET GROUND TRUTH AND MASKS
                        cfg = self.losses_cfg[loss_key][sub_loss_key]
                        depth_gt = get_gt(cfg, batch, predictions, task, pred_key)
                        mask = get_mask_valid(cfg, batch, predictions)
                        if not apply_loss(pred_key, cfg.apply_to): continue
                        ### SELF-SUPERVISION
                        if sub_loss_key.startswith('self_supervision'):
                            output = self.loss_scene_flow_self_supervision(
                                pred_key, loss_fn, rgb, depth, scene_flow, cams, predictions,
                                mask=mask, epoch=epoch, task=task)
                            update_losses(losses, metrics, extras, output, pred_key, sub_loss_key)

        return losses, metrics, extras

################################################
