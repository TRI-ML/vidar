# Copyright 2023 Toyota Research Institute.  All rights reserved.

from functools import partial

from knk_vision.vidar.vidar.arch.models.generic.GenericModel_utils import \
    filter_targets, filter_contexts, get_if_not_none, make_pairs, sample_from_coords, parse_source
from knk_vision.vidar.vidar.utils.config import Config
from knk_vision.vidar.vidar.utils.data import update_dict_nested, get_from_dict, not_none, one_is
from knk_vision.vidar.vidar.utils.flow import \
    fwd_bwd_optflow_consistency_check, optflow_from_motion, \
    warp_depth_from_motion, reverse_optflow, depth_from_optflow, \
    residual_scene_flow_from_depth_optflow, to_world_scene_flow
from knk_vision.vidar.vidar.utils.types import is_list


class GenericModelWarps:
    """Generic model for multi-model inputs and multi-task output. Subclass focused on warping functionality.

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """
    def __init__(self, cfg):

        # Prepare warp configuration
        self.warps_cfg = {}
        if cfg.model.has('warps'):
            for key in cfg.model.warps.dict.keys():
                cfg_key = cfg.model.warps.dict[key]
                sources = cfg_key.sources

                depth = [val for val in sources if 'depth' in val]
                optical_flow = [val for val in sources if 'optical_flow' in val]
                scene_flow = [val for val in sources if 'scene_flow' in val]
                cams = [val for val in sources if 'cams' in val]

                depth = None if len(depth) == 0 else depth[0]
                optical_flow = None if len(optical_flow) == 0 else optical_flow[0]
                scene_flow = None if len(scene_flow) == 0 else scene_flow[0]
                cams = None if len(cams) == 0 else cams[0]

                mode, add_to, name = key.split('_')

                self.warps_cfg[key] = Config(
                    targets=partial(filter_targets, params=cfg_key.targets),
                    contexts=partial(filter_contexts, params=cfg_key.contexts),
                    depth=depth, optical_flow=optical_flow, scene_flow=scene_flow, cams=cams,
                    output=cfg_key.output, mode=mode, add_to=add_to, name=name,
                )

################################################

    def forward_warps(self, batch, predictions=None):
        """ Forward loop for warping"""

        warps = {}

        keys = batch['rgb'].keys()
        for name, cfg in self.warps_cfg.items():

            # Get sources for warping
            depth_src, depth_key, depth = parse_source(cfg.depth, batch, predictions)
            scnflow_src, scnflow_key, scnflow = parse_source(cfg.scene_flow, batch, predictions)
            optflow_src, optflow_key, optflow = parse_source(cfg.optical_flow, batch, predictions)

            has_pred = one_is('pred', depth_src, scnflow_src, optflow_src)
            if (predictions is None and has_pred) or (predictions is not None and not has_pred):
                continue

            # Get warping coordinates
            coords_depth = None if depth is None else \
                get_if_not_none(predictions, depth_key.replace('depth', 'coords'))
            coords_scnflow = None if scnflow is None else \
                get_if_not_none(predictions, scnflow_key.replace('scene_flow', 'coords'))
            coords_optflow = None if optflow is None else \
                get_if_not_none(predictions, optflow_key.replace('optical_flow', 'coords'))

            # Parse predictions with coordinates
            if coords_depth is not None:
                coords, cams = coords_depth, predictions[depth_key.replace('depth', 'cams')]
            elif coords_scnflow is not None:
                coords, cams = coords_scnflow, predictions[scnflow_key.replace('scene_flow', 'cams')]
            elif coords_optflow is not None:
                coords, cams = coords_optflow, predictions[optflow_key.replace('optical_flow', 'cams')]
            else:
                coords, cams = None, batch['cams'] if predictions is None else predictions['cams']

            # Set targets and contexts for warping
            tgts = cfg.targets(self.base_cam, keys)
            ctxs = cfg.contexts(tgts, keys)

            warps[name] = {}
            
            # Warps for motion (ego-motion + depth)
            if cfg.mode == 'motion' and not_none(depth, cams):
                _, pairs = make_pairs(tgts, ctxs, only_both_ways=False)

                for pair in pairs:
                    tgt, ctx = pair

                    depth_tgt = get_from_dict(depth, tgt)
                    depth_ctx = get_from_dict(depth, ctx)
                    scnflow_tgt = get_from_dict(scnflow, tgt, ctx)
                    scnflow_ctx = get_from_dict(scnflow, ctx, tgt)
                    cams_tgt, cams_ctx = cams[tgt], cams[ctx]

                    if coords is not None:
                        scnflow_tgt = sample_from_coords(scnflow_tgt, coords[tgt], cams[tgt])
                        scnflow_ctx = sample_from_coords(scnflow_ctx, coords[ctx], cams[ctx])

                    if is_list(depth_tgt) and not is_list(scnflow_tgt):
                        scnflow_tgt = [scnflow_tgt] * len(depth_tgt)
                    elif not is_list(depth_tgt) and is_list(scnflow_tgt):
                        depth_tgt = [depth_tgt] * len(scnflow_tgt)
                    if is_list(depth_ctx) and not is_list(scnflow_ctx):
                        scnflow_ctx = [scnflow_ctx] * len(depth_ctx)
                    elif not is_list(depth_ctx) and is_list(scnflow_ctx):
                        depth_ctx = [depth_ctx] * len(scnflow_ctx)

                    for output in cfg.output:
                        if output == 'optical_flow_motion':
                            optflow_tgt = optflow_from_motion(
                                cams_ctx, depth_tgt, cams_tgt, tgt_world_scnflow=scnflow_tgt)
                            optflow_ctx = optflow_from_motion(
                                cams_tgt, depth_ctx, cams_ctx, tgt_world_scnflow=scnflow_ctx)
                            update_dict_nested(warps[name], output, tgt, ctx, optflow_tgt)
                            update_dict_nested(warps[name], output, ctx, tgt, optflow_ctx)
                        if output == 'mask_motion':
                            mask_motion_tgt, mask_motion_ctx = fwd_bwd_optflow_consistency_check(
                                optflow_from_motion(cams_ctx, depth_tgt, cams_tgt, tgt_world_scnflow=scnflow_tgt),
                                optflow_from_motion(cams_tgt, depth_ctx, cams_ctx, tgt_world_scnflow=scnflow_ctx))
                            update_dict_nested(warps[name], output, tgt, ctx, mask_motion_tgt)
                            update_dict_nested(warps[name], output, ctx, tgt, mask_motion_ctx)
                        if output == 'depth_warped':
                            depth_warped_tgt = warp_depth_from_motion(
                                depth_ctx, cams_ctx, depth_tgt, cams_tgt,
                                tgt_world_scnflow=scnflow_tgt, ctx_world_scnflow=scnflow_ctx)
                            depth_warped_ctx = warp_depth_from_motion(
                                depth_tgt, cams_tgt, depth_ctx, cams_ctx,
                                tgt_world_scnflow=scnflow_ctx, ctx_world_scnflow=scnflow_tgt)
                            update_dict_nested(warps[name], output, tgt, ctx, depth_warped_tgt)
                            update_dict_nested(warps[name], output, ctx, tgt, depth_warped_ctx)

            # Warps for optical flow 
            elif cfg.mode == 'optflow' and not_none(optflow):
                _, pairs = make_pairs(tgts, ctxs, only_both_ways=True)

                for pair in pairs:
                    tgt, ctx = pair

                    depth_tgt = get_from_dict(depth, tgt)
                    depth_ctx = get_from_dict(depth, ctx)
                    optflow_tgt = get_from_dict(optflow, tgt, ctx)
                    optflow_ctx = get_from_dict(optflow, ctx, tgt)
                    scnflow_tgt = get_from_dict(scnflow, tgt, ctx)
                    scnflow_ctx = get_from_dict(scnflow, ctx, tgt)
                    cams_tgt, cams_ctx = cams[tgt], cams[ctx]

                    if is_list(depth_tgt) and not is_list(optflow_tgt):
                        optflow_tgt = [optflow_tgt] * len(depth_tgt)
                    elif not is_list(depth_tgt) and is_list(optflow_tgt):
                        depth_tgt = [depth_tgt] * len(optflow_tgt)
                    if is_list(depth_ctx) and not is_list(optflow_ctx):
                        optflow_ctx = [optflow_ctx] * len(depth_ctx)
                    elif not is_list(depth_ctx) and is_list(optflow_ctx):
                        depth_ctx = [depth_ctx] * len(optflow_ctx)

                    for output in cfg.output:
                        if output == 'mask_optical_flow':
                            mask_optflow_tgt, mask_optflow_ctx = fwd_bwd_optflow_consistency_check(optflow_tgt, optflow_ctx)
                            update_dict_nested(warps[name], output, tgt, ctx, mask_optflow_tgt)
                            update_dict_nested(warps[name], output, ctx, tgt, mask_optflow_ctx)
                        if output == 'optical_flow_reverse':
                            update_dict_nested(warps[name], output, tgt, ctx, reverse_optflow(optflow_ctx, optflow_tgt))
                            update_dict_nested(warps[name], output, ctx, tgt, reverse_optflow(optflow_tgt, optflow_ctx))
                        if output == 'optical_flow_residual':
                            optflow_tgt_motion = optflow_from_motion(cams_ctx, depth_tgt, cams_tgt, tgt_world_scnflow=scnflow_tgt)
                            optflow_ctx_motion = optflow_from_motion(cams_tgt, depth_ctx, cams_ctx, tgt_world_scnflow=scnflow_ctx)
                            update_dict_nested(warps[name], output, tgt, ctx, optflow_tgt - optflow_tgt_motion)
                            update_dict_nested(warps[name], output, ctx, tgt, optflow_ctx - optflow_ctx_motion)
                        if output == 'depth_triangulate':
                            update_dict_nested(warps[name], output, tgt, ctx, depth_from_optflow(
                                optflow_tgt, cams_ctx.K, cams_ctx.relative_to(cams_tgt).Twc.T, [optflow_tgt]))
                            update_dict_nested(warps[name], output, ctx, tgt, depth_from_optflow(
                                optflow_ctx, cams_tgt.K, cams_tgt.relative_to(cams_ctx).Twc.T, [optflow_ctx]))
                        if output == 'scene_flow_optical_flow':
                            scnflow21, scnflow12 = residual_scene_flow_from_depth_optflow(
                                depth_tgt, depth_ctx, cams_tgt, cams_ctx, optflow_ctx, optflow_tgt)
                            scnflow21 = to_world_scene_flow(cams_ctx, depth_ctx, scnflow21)
                            scnflow12 = to_world_scene_flow(cams_tgt, depth_tgt, scnflow12)
                            update_dict_nested(warps[name], output, tgt, ctx, scnflow12)
                            update_dict_nested(warps[name], output, ctx, tgt, scnflow21)

            # Add warps to batch or predictions
            if cfg.add_to is not None and len(warps[name]) > 0:
                warp = {f'{key}_{cfg.name}': val for key, val in warps[name].items()}
                data = {'batch': batch, 'predictions': predictions}[cfg.add_to]
                data.update(**warp)
                if coords is not None:
                    for task in ['depth', 'scene_flow']:
                        for key, val in warp.items():
                            if key.startswith(task):
                                data[key.replace(task, 'coords')] = coords
                                data[key.replace(task, 'cams')] = cams

################################################
