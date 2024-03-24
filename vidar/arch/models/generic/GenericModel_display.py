# Copyright 2023 Toyota Research Institute.  All rights reserved.
import torch
import numpy as np

from einops import rearrange

from camviz import Draw
from camviz.utils.cmaps import jet

from knk_vision.vidar.vidar.utils.data import get_from_list, get_from_dict, interleave_dict
from knk_vision.vidar.vidar.utils.depth import calculate_normals
from knk_vision.vidar.vidar.utils.flow import warp_motion_dict, warp_optflow_dict
from knk_vision.vidar.vidar.utils.viz import viz_depth, viz_optical_flow, viz_scene_flow, viz_normals
from knk_vision.vidar.vidar.utils.types import is_list, is_tuple
from knk_vision.vidar.vidar.utils.data import modrem

from knk_vision.vidar.vidar.utils.write import write_pickle
from knk_vision.vidar.vidar.utils.read import read_pickle



def get_sub_key(data, sub_key1, sub_key2=None):
    """Get sub key from data dict."""
    data = {key: val.get(sub_key1, None) for key, val in data.items()}
    if sub_key2 is None:
        return data
    return {key: val.get(sub_key2, None) for key, val in data.items()}


def do_draw(self, i, key, val, task_show, prefix, 
            with_images=True, show_rays=False):
    """Draw information for a single camera."""
    r, c = modrem(i, 3)
    task = {0: 'rgb', 1: 'depth', 2: 'normals'}[task_show]
    if with_images and r < 4 and c < 3:
        self.draw[f'img{r}{c}'].image(f'{prefix}_{task}_{key}')
    color = 'whi' if is_tuple(key[0]) else {'encode': 'mag', 'decode': 'cya'}[prefix]
    width = 2 if is_tuple(key[0]) else 4
    self.draw['wld'].object(val, color=color, width=width, tex=f'{prefix}_{task}_{key}')
    color = key[0][0] if is_tuple(key[0]) else key[0]
    self.draw['wld'].size(2).color(color).points(
        f'{prefix}_pts_xyz_{key}', f'{prefix}_pts_rgb_{key}' if self.mode_color == 0 else None)
    if show_rays:
        self.draw['wld'].width(1).color('yel').lines(f'{prefix}_rays_{key}')


def interleave_points(orig_rays):
    """Interleave points for visualization."""
    b, c, h, w = orig_rays.shape
    orig = orig_rays[:, :3].permute(0, 2, 3, 1).view(b, -1, 3)
    rays = orig_rays[:, 3:].permute(0, 2, 3, 1).view(b, -1, 3)
    interleave = torch.zeros((b, 2 * h * w, 3), device=orig.device)
    interleave[:, 0::2] = orig
    interleave[:, 1::2] = orig + rays * 2
    return interleave


def prep_mask(data):
    """Prepare mask for visualization."""
    for key1 in data:
        for key2 in data[key1]:
            data[key1][key2] = get_from_list(data[key1][key2])[0].permute(1, 2, 0).cpu().numpy()
    return data


class Display:
    """Generic model for multi-model inputs and multi-task output. Subclass focused on display functionality.

    Parameters
    ----------
    cfg : Config
        Configuration file for model generation
    """    
    def __init__(self, cfg):

        wh = cfg.resolution[::-1]

        self.draw = Draw((2100, 900))
        self.draw.add3Dworld('wld', (0.5, 0.0, 1.0, 1.0),
            pose=(0.98506, -19.68949, -18.51727, -0.94434, -0.32815, -0.01349, 0.01923),
        )
        self.draw.add2DimageGrid('img', (0.0, 0.0, 0.5, 1.0), n=(4, 3), res=wh)

        self.mode_color = 0
        self.mode_camera = 2 # 1
        self.first = True

    def prep_single(self, pref, data, viz=None):
        """Prepare data from a single camera for visualization"""
        if data is None:
            return None
        for key, val in data.items():
            val = val[0] if val is not None and is_list(val) else val
            viz_val = viz(val[0]) if viz is not None else val[0].permute(1, 2, 0).cpu().numpy()
            self.draw.addTexture(f'{pref}_{key}', viz_val)
            self.draw.addBufferf(f'{pref}_{key}', viz_val)

    def prep_multi(self, pref, data, viz=None, valid=None):
        """Prepare data from multiple cameras for visualization"""
        if data is None:
            return None
        for key1, val1 in data.items():
            for key2, val2 in val1.items():
                viz_val = viz(val2[0]) if viz is not None else val2[0].permute(1, 2, 0).cpu().numpy()
                if valid is not None:
                    viz_val *= valid[key1][key2]
                self.draw.addTexture(f'{pref}_{key1}_{key2}', viz_val)
                self.draw.addBufferf(f'{pref}_{key1}_{key2}', viz_val)
        for key in data.keys():
            for i in range(-2, 3):
                if (i, key[1]) not in data[key].keys():
                    self.draw.addTexture(f'{pref}_{key}_{i}_{key1}', (640, 192))
                    self.draw.addBuffer3f(f'{pref}_{key}_{i}_{key1}', 640 * 192)

    def loop_gt(self, batch):
        """Loop to display GT information"""

        single_tasks = ['rgb', 'depth', 'normals']
        multi_tasks = [
            'depth_tri', 'depth_warped',
            'optflow', 'optflow_reverse', 'optflow_motion', 'optflow_residual',
            'scnflow', 'scnflow_optflow',
        ]
        tasks = single_tasks + multi_tasks * 2

        n_single = len(single_tasks)
        n_multi = len(multi_tasks)

        offset = {
            **{val: -1 for val in list(range(n_single, n_single + n_multi))},
            **{val:  1 for val in list(range(n_single + n_multi, n_single + 2 * n_multi))},
        }
        colors = {-1: 'cya', 0: 'gre', 1: 'yel'}

        task_id = 0

        rgb = batch['rgb']
        cams = batch['cams']

        depth = batch['depth']
        depth_tri = batch['depth_triangulate_gt']
        depth_warped = batch['depth_warped_gt']

        optflow = batch['optical_flow']
        optflow_motion = batch['optical_flow_motion_gt']
        optflow_reverse = batch['optical_flow_reverse_gt']
        optflow_residual = batch['optical_flow_residual_gt']

        scnflow = get_from_dict(batch, 'scene_flow')
        scnflow_optflow = batch['scene_flow_optical_flow_gt']

        valid_motion = prep_mask(batch['mask_motion_gt'])
        valid_optflow = prep_mask(batch['mask_optical_flow_gt'])

        keys = {key: list(val.keys()) for key, val in optflow.items()}

        pts_motion = {key: cams[key].reconstruct_depth_map(
            val, to_world=True) for key, val in depth.items()}

        pts_scnflow = {key: cams[key].reconstruct_depth_map(
            val, world_scene_flow=None if key[0] == 0 else scnflow[key][(0,key[1])],
            to_world=True) for key, val in depth.items()
        }

        normals = calculate_normals(depth, cams, to_world=True)
        pts_normals = {key: cams[key].reconstruct_depth_map(
            depth[key], to_world=True, scene_flow=normals[key] / 5) for key in depth.keys()}
        pts_normals = interleave_dict(pts_motion, pts_normals)

        clr_normals = {key: viz_normals(val[0]).reshape(-1, 3) for key, val in normals.items()}
        for key, val in clr_normals.items():
            tmp = np.zeros((1, val.shape[0] * 2, 3))
            tmp[:, 0::2], tmp[:, 1::2] = val, val
            clr_normals[key] = tmp

        self.prep_single('rgb', rgb)
        self.prep_single('depth', depth, viz_depth)
        self.prep_single('normals', normals, viz_normals)
        self.prep_multi('optflow', optflow, viz_optical_flow)
        self.prep_multi('scnflow', scnflow, viz_scene_flow)

        self.prep_multi('depth_tri', depth_tri, viz_depth, valid=valid_motion)
        self.prep_multi('depth_warped', depth_warped, viz_depth, valid=valid_motion)

        self.prep_multi('optflow_motion', optflow_motion, viz_optical_flow, valid=valid_motion)
        self.prep_multi('optflow_reverse', optflow_reverse, viz_optical_flow, valid=valid_optflow)
        self.prep_multi('optflow_residual', optflow_residual, viz_optical_flow, valid=valid_motion)

        self.prep_multi('scnflow', scnflow, viz_scene_flow, valid=valid_motion)
        self.prep_multi('scnflow_optflow', scnflow_optflow, viz_scene_flow, valid=valid_motion)

        synth_motion = warp_motion_dict(rgb, depth, cams, world_scnflow=scnflow, keys=keys)
        synth_motion_optflow = warp_motion_dict(rgb, depth, cams, world_scnflow=scnflow_optflow, keys=keys)
        synth_motion_warped = warp_motion_dict(rgb, depth_warped, cams, world_scnflow=scnflow, keys=keys)
        synth_motion_tri = warp_motion_dict(rgb, depth_tri, cams, world_scnflow=scnflow, keys=keys)

        synth_optflow = warp_optflow_dict(rgb, optflow, keys=keys)
        synth_optflow_motion = warp_optflow_dict(rgb, optflow_motion, keys=keys)
        synth_optflow_reverse = warp_optflow_dict(rgb, optflow_reverse, keys=keys)

        def save(data, name):
            gt = {}
            for key, val in data.items():
                gt[key] = {}
                for key2, val2 in val.items():
                    gt[key][key2] = val2
            write_pickle(name + '.pkl', gt)

        def check(data, name):
            gt = read_pickle(name + '.pkl')
            for key, val in data.items():
                for key2, val2 in val.items():
                    assert torch.allclose(data[key][key2], gt[key][key2])

        self.prep_single('synth_rgb', rgb)
        self.prep_single('synth_depth', rgb)

        self.prep_multi('synth_depth_tri', synth_motion_tri, valid=valid_motion)
        self.prep_multi('synth_depth_warped', synth_motion_warped, valid=valid_motion)

        self.prep_multi('synth_optflow', synth_optflow, valid=valid_optflow)
        self.prep_multi('synth_optflow_motion', synth_optflow_motion, valid=valid_motion)
        self.prep_multi('synth_optflow_reverse', synth_optflow_reverse, valid=valid_optflow)

        self.prep_multi('synth_scnflow', synth_motion, valid=valid_motion)
        self.prep_multi('synth_scnflow_optflow', synth_motion_optflow, valid=valid_motion)

        cv_cams = {key: self.draw.cvcam(val, scale=0.1) for key, val in cams.items()}

        for key, val in pts_motion.items():
            self.draw.addBufferf(f'pts_xyz_motion_{key}', val[0])
        for key, val in pts_scnflow.items():
            self.draw.addBufferf(f'pts_xyz_scnflow_{key}', val[0])
        for key, val in pts_normals.items():
            self.draw.addBufferf(f'pts_xyz_normals_{key}', val[0])
        for key, val in clr_normals.items():
            self.draw.addBufferf(f'pts_clr_normals_{key}', val[0])

        pref = {0: '', 1: 'synth_'}
        pref_id = 0

        pcl = {0: 'motion', 1: 'scnflow'}
        pcl_id = 0

        show_normals = False

        while self.draw.input():
            if self.draw.RETURN:
                pref_id = (pref_id + 1) % len(pref)
            if self.draw.SPACE:
                pcl_id = (pcl_id + 1) % len(pcl)
            if self.draw.LEFT:
                task_id = (task_id - 1) % len(tasks)
                self.draw.halt(200)
            if self.draw.RIGHT:
                task_id = (task_id + 1) % len(tasks)
                self.draw.halt(200)
            if self.draw.KEY_X:
                show_normals = not show_normals
                self.draw.halt(200)
            self.draw.clear()
            for i in range(-1, 2):
                for j in range(2):
                    if tasks[task_id] in single_tasks:
                        self.draw[f'img{i+1}{j}'].image(
                            f'{pref[pref_id]}{tasks[task_id]}_{i}_{j}')
                    elif tasks[task_id] in multi_tasks:
                        self.draw[f'img{i+1}{j}'].image(
                            f'{pref[pref_id]}{tasks[task_id]}_{i}_{j}_{i + offset[task_id]}_{j}')
            for key, val in cv_cams.items():
                self.draw['wld'].object(val, color='gra', tex=f'rgb_{key}')
            for key in pts_motion.keys():
                if tasks[task_id] in single_tasks:
                    self.draw['wld'].color(colors[key[0]]).size(1).points(
                        f'pts_xyz_{pcl[pcl_id]}_{key}',
                        f'{tasks[task_id]}_{key}')
                elif tasks[task_id] in multi_tasks:
                    self.draw['wld'].size(1).points(
                        f'pts_xyz_{pcl[pcl_id]}_{key}',
                        f'{tasks[task_id]}_{key}_{key[0] + offset[task_id]}_{key[1]}')
            if show_normals:
                for key in pts_normals.keys():
                    self.draw['wld'].color('red').width(2).lines(
                        f'pts_xyz_normals_{key}', f'pts_clr_normals_{key}')
            self.draw.update(30)

    def loop_define(self, batch, predictions, extra):
        """Loop to display DeFiNe information"""

        rgb = batch['rgb']
        depth = batch['depth']
        cams = batch['cams']

        info_key = 'info_define-trainD'
        if info_key in predictions:
            xyz = {key: rearrange(val['xyz'], 'b n d c -> b (n d) c') 
                   for key, val in predictions[info_key].items()}
            for key, val in xyz.items():
                self.draw.addBufferf(f'sample_xyz_{key}', val[0])

        colors = {-1: 'cya', 0: 'gre', 1: 'yel'}

        pts = {key: cams[key].reconstruct_depth_map(
            val, to_world=True) for key, val in depth.items()}

        self.prep_single('rgb', rgb)
        self.prep_single('depth', depth, viz_depth)

        cv_cams = {key: self.draw.cvcam(val, scale=0.1) for key, val in cams.items()}

        for key, val in pts.items():
            self.draw.addBufferf(f'pts_xyz_{key}', val[0])
            self.draw.addBufferf(f'pts_rgb_{key}', rgb[key][0])

        extra_key = 'define_encode_data'
        if extra_key in extra:

            encode_rays = get_sub_key(extra[extra_key], 'camera')
            encode_rgb = get_sub_key(extra[extra_key], 'gt', 'rgb')
            encode_depth = get_sub_key(extra[extra_key], 'gt', 'depth')
            encode_cams = get_sub_key(extra[extra_key], 'cam')
            encode_pts = {key: encode_cams[key].reconstruct_depth_map(
                val, to_world=True) for key, val in encode_depth.items()}
            encode_normals = {key: calculate_normals(encode_depth[key], encode_cams[key], to_world=True) 
                              for key in encode_depth.keys()}
            encode_cv_cams = {key: self.draw.cvcam(val, scale=0.1) for key, val in encode_cams.items()}

            for key, val in encode_pts.items():
                self.draw.addBufferf(f'encode_pts_xyz_{key}', val[0])
            for key, val in encode_rgb.items():
                self.draw.addBufferf(f'encode_pts_rgb_{key}', val[0])
            for key, val in encode_rays.items():
                if val is not None:
                    self.draw.addBufferf(f'encode_rays_{key}', interleave_points(val)[0])
            self.prep_single('encode_rgb', encode_rgb)
            self.prep_single('encode_depth', encode_depth, viz=viz_depth)
            self.prep_single('encode_normals', encode_normals, viz=viz_normals)
        else:
            encode_cv_cams = None

        extra_key = 'define_decode_data'
        if extra_key in extra:

            decode_rays = get_sub_key(extra[extra_key], 'camera')
            decode_rgb = get_sub_key(extra[extra_key], 'gt', 'rgb')
            decode_depth = get_sub_key(extra[extra_key], 'gt', 'depth')
            decode_cams = get_sub_key(extra[extra_key], 'cam_scaled')
            decode_pts = {key: decode_cams[key].reconstruct_depth_map(
                val, to_world=True) for key, val in decode_depth.items()}
            decode_normals = {key: calculate_normals(decode_depth[key], decode_cams[key], to_world=True) 
                              for key in decode_depth.keys()}
            decode_cv_cams = {key: self.draw.cvcam(val, scale=0.1) for key, val in decode_cams.items()}

            for key, val in decode_pts.items():
                self.draw.addBufferf(f'decode_pts_xyz_{key}', val[0])
            for key, val in decode_rgb.items():
                self.draw.addBufferf(f'decode_pts_rgb_{key}', val[0])
            for key, val in decode_rays.items():
                if val is not None:
                    self.draw.addBufferf(f'decode_rays_{key}', interleave_points(val)[0])
            self.prep_single('decode_rgb', decode_rgb)
            self.prep_single('decode_depth', decode_depth, viz=viz_depth)
            self.prep_single('decode_normals', decode_normals, viz=viz_normals)
        else:
            decode_cv_cams = None

        task_show = 0
        show_rays = 0
        show_samples = 0

        while self.draw.input():
            if self.draw.RETURN:
                self.mode_color = (self.mode_color + 1) % 2
                self.draw.halt(100)
            if self.draw.SPACE:
                self.mode_camera = (self.mode_camera + 1) % 4
                self.draw.halt(100)
            if self.draw.LEFT:
                task_show = (task_show - 1) % 3
                self.draw.halt(100)
            if self.draw.RIGHT:
                task_show = (task_show + 1) % 3
                self.draw.halt(100)
            if self.draw.KEY_R:
                show_rays = not show_rays
                self.draw.halt(100)
            if self.draw.KEY_S:
                show_samples = not show_samples
                self.draw.halt(100)
            self.draw.clear()
            if self.mode_camera == 0:
                for i, (key, val) in enumerate(cv_cams.items()):
                    r, c = modrem(i, 3)
                    task = {0: 'rgb', 1: 'depth', 2: 'normals'}[task_show]
                    self.draw[f'img{r}{c}'].image(f'{task}_{key}')
                    self.draw['wld'].object(val, color=colors[key[0]], tex=f'{task}_{key}')
                    self.draw['wld'].size(2).color(colors[key[0]]).points(
                        f'pts_xyz_{key}', f'pts_rgb_{key}' if self.mode_color == 0 else None)
                self.draw['wld'].text('batch', (0, 0))
            elif self.mode_camera == 1:
                for i, (key, val) in enumerate(encode_cv_cams.items()):
                    do_draw(self, i, key, val, task_show, 'encode', show_rays=show_rays)
                self.draw['wld'].text('encode', (0, 0))
            elif self.mode_camera == 2:
                for i, (key, val) in enumerate(decode_cv_cams.items()):
                    do_draw(self, i, key, val, task_show, 'decode', show_rays=show_rays)
                    if show_samples:
                        self.draw['wld'].size(1).color('red').points(f'sample_xyz_{key}')
                self.draw['wld'].text('decode', (0, 0))
            elif self.mode_camera == 3:
                for key, val in encode_cv_cams.items():
                    do_draw(self, i, key, val, task_show, 'encode', 
                            with_images=False, show_rays=show_rays)
                for key, val in decode_cv_cams.items():
                    do_draw(self, i, key, val, task_show, 'decode', 
                            with_images=False, show_rays=show_rays)
                self.draw['wld'].text('encode+decode', (0, 0))
            self.draw['wld'].scr('wld').text(batch['tag'][0], (200, 0))
            self.draw.update(30)

    def loop_pred(self, batch, predictions, extra):
        """Loop to display prediction information"""

        rgb = batch['rgb']
        depth = batch['depth']
        cams = batch['cams']

        normals = {key: calculate_normals(depth[key], cams[key]) for key in depth.keys()}

        pred_depth = get_from_dict(predictions, 'depth_define')
        pred_normals = {key: calculate_normals(pred_depth[key][0], cams[key]) for key in pred_depth.keys()}

        colors = {-1: 'cya', 0: 'gre', 1: 'yel'}

        pts = {key: cams[key].reconstruct_depth_map(
            val, to_world=True) for key, val in depth.items()}

        bins = get_from_dict(predictions, 'bins_define')
        if bins is not None:
            z_vals = get_from_dict(predictions, 'zvals_define')
            ones = {key: torch.ones_like(depth[key]) for key in depth.keys()}

        self.prep_single('rgb', rgb)
        self.prep_single('depth', depth, viz_depth)
        self.prep_single('normals', normals, viz_normals)

        self.prep_single('pred_depth', pred_depth, viz_depth)
        self.prep_single('pred_normals', pred_normals, viz_normals)

        cv_cams = {key: self.draw.cvcam(val, scale=0.1) for key, val in cams.items()}

        for key, val in pts.items():
            self.draw.addBufferf(f'pts_xyz_{key}', val[0])
            self.draw.addBufferf(f'pts_rgb_{key}', rgb[key][0])

        if 'encode_data' in extra:
            encode_rgb = {key: val['rgb'] for key, val in extra['encode_data'].items()}
            encode_depth = {key: val['depth'] for key, val in extra['encode_data'].items()}
            encode_cams = {key: val['cam_mod'] for key, val in extra['encode_data'].items()}
            encode_normals = {key: calculate_normals(encode_depth[key], encode_cams[key]) for key in encode_depth.keys()}
            encode_cv_cams = {key: self.draw.cvcam(val, scale=0.1) for key, val in encode_cams.items()}
            encode_pts = {key: encode_cams[key].reconstruct_depth_map(
                val, to_world=True) for key, val in encode_depth.items()}
            for key, val in encode_pts.items():
                self.draw.addBufferf(f'encode_pts_xyz_{key}', val[0])
            for key, val in encode_rgb.items():
                self.draw.addBufferf(f'encode_pts_rgb_{key}', val[0])
            self.prep_single('encode_rgb', encode_rgb)
            self.prep_single('encode_depth', encode_depth, viz_depth)
            self.prep_single('encode_normals', encode_normals, viz_normals)
        else:
            encode_cv_cams = None

        if 'decode_data' in extra:
            decode_rgb = {key: val['rgb'] for key, val in extra['decode_data'].items()}
            decode_depth = {key: val['depth'] for key, val in extra['decode_data'].items()}
            decode_cams = {key: val['cam_scaled'] for key, val in extra['decode_data'].items()}
            decode_normals = {key: calculate_normals(decode_depth[key], decode_cams[key]) for key in decode_depth.keys()}
            decode_cv_cams = {key: self.draw.cvcam(val, scale=0.1) for key, val in decode_cams.items()}
            decode_pts = {key: decode_cams[key].reconstruct_depth_map(
                val, to_world=True) for key, val in decode_depth.items()}
            for key, val in decode_pts.items():
                self.draw.addBufferf(f'decode_pts_xyz_{key}', val[0])
            for key, val in decode_rgb.items():
                self.draw.addBufferf(f'decode_pts_rgb_{key}', val[0])
            self.prep_single('decode_rgb', decode_rgb)
            self.prep_single('decode_depth', decode_depth, viz_depth)
            self.prep_single('decode_normals', decode_normals, viz_normals)
        else:
            decode_cv_cams = None

        while self.draw.input():
            if self.draw.RETURN:
                self.mode_color = (self.mode_color + 1) % 2
            if self.draw.SPACE:
                self.mode_camera = (self.mode_camera + 1) % 4
            self.draw.clear()

            self.draw['img00'].image('rgb_0_0')
            self.draw['img10'].image('encode_rgb_0_0')
            self.draw['img20'].image('encode_rgb_0_0')

            self.draw['img01'].image('depth_0_0')
            self.draw['img11'].image('depth2_0_0')
            self.draw['img21'].image('decode_depth_0_0')

            self.draw['img02'].image('normals_0_0')
            self.draw['img12'].image('encode_normals_0_0')

            self.draw['img31'].image('pred_depth_0_0')
            self.draw['img32'].image('pred_normals_0_0')

            if self.mode_camera == 0:
                for key, val in cv_cams.items():
                    self.draw['wld'].object(val, color=colors[key[0]], tex=f'rgb_{key}')
                self.draw['wld'].text('batch', (0, 0))
            elif self.mode_camera == 1:
                for key, val in encode_cv_cams.items():
                    self.draw['wld'].object(val, color='mag', tex=f'encode_rgb_{key}')
                    self.draw['wld'].size(2).color(colors[key[0]]).points(
                        f'encode_pts_xyz_{key}', f'encode_pts_rgb_{key}' if self.mode_color == 0 else None)
                self.draw['wld'].text('encode', (0, 0))
            elif self.mode_camera == 2:
                for key, val in decode_cv_cams.items():
                    self.draw['wld'].object(val, color='cya')
                    self.draw['wld'].size(2).color(colors[key[0]]).points(
                        f'decode_pts_xyz_{key}', f'decode_pts_rgb_{key}' if self.mode_color == 0 else None)
                self.draw['wld'].text('decode', (0, 0))
            elif self.mode_camera == 3:
                for key, val in encode_cv_cams.items():
                    self.draw['wld'].object(val, color='mag')
                    self.draw['wld'].size(2).color(colors[key[0]]).points(
                        f'encode_pts_xyz_{key}', f'encode_pts_rgb_{key}' if self.mode_color == 0 else None)
                for key, val in decode_cv_cams.items():
                    self.draw['wld'].object(val, color='cya')
                    self.draw['wld'].size(2).color(colors[key[0]]).points(
                        f'decode_pts_xyz_{key}', f'decode_pts_rgb_{key}' if self.mode_color == 0 else None)
                self.draw['wld'].text('encode+decode', (0, 0))
            self.draw['wld'].scr('wld').text(batch['tag'][0], (200, 0))
            self.draw.update(30)

            break

    def loop_train(self, batch, predictions, extra):
        """Loop to display training information"""

        rgb = batch['rgb']
        depth = batch['depth']
        pred_depth = predictions['depth']

        cams = batch['cams']
        normals = calculate_normals(depth, cams, to_world=True)
        pred_normals = calculate_normals(pred_depth, cams, to_world=True)

        if self.first:
            self.draw.scr('wld').viewer.setPose((5.66071, -6.69242, -9.98902, 0.99028, 0.10289, 0.09348, -0.00536))
            self.first = False

        self.prep_single('rgb', rgb)
        self.prep_single('depth', depth, viz_depth)
        self.prep_single('pred_depth', pred_depth, viz_depth)
        self.prep_single('normals', normals, viz_normals)
        self.prep_single('pred_normals', pred_normals, viz_normals)

        pts = {key: cams[key].reconstruct_depth_map(depth[key], to_world=False) for key in cams.keys()}
        pred_pts = {key: cams[key].reconstruct_depth_map(pred_depth[key][0], to_world=False) for key in cams.keys()}

        tgt = (0,0)
        self.draw.addBufferf('pts', pts[tgt][0])
        self.draw.addBufferf('pred_pts', pred_pts[tgt][0])

        diff = (depth[tgt] - pred_depth[tgt][0]).abs()
        diff = diff[0].view(1, -1).permute(1, 0)
        self.draw.addBufferf('diff', jet(diff, range=(0.0, 10.0)))

        while self.draw.input():
            self.draw.clear()

            self.draw['img00'].image('rgb_0_0')
            self.draw['img01'].image('depth_0_0')
            self.draw['img02'].image('pred_depth_0_0')

            self.draw['img11'].image('normals_0_0')
            self.draw['img12'].image('pred_normals_0_0')

            self.draw['img10'].image('grad1_0_0')
            self.draw['img20'].image('grad2_0_0')

            self.draw['wld'].size(1).color('blu').points('pts')
            self.draw['wld'].size(2).color('red').points('pred_pts', 'diff')

            self.draw.update(30)

            break
