# Copyright 2023 Toyota Research Institute.  All rights reserved.

import os

from vidar.utils.config import cfg_has
from vidar.utils.data import make_list, expand_and_break
from vidar.utils.types import is_dict, is_list
from vidar.utils.viz import viz_depth, viz_optical_flow, viz_normals, viz_stddev
from vidar.utils.write import write_image, write_depth, write_normals, write_optical_flow, write_pickle


def ctx_str(ctx):
    """Parse context to string"""
    replaces = [[' ', ''], ['(', ''], [')', ''], [',', '_']]
    ctx = str(ctx)
    for replace in replaces:
        ctx = ctx.replace(replace[0], replace[1])
    return ctx


def data_to_cpu(data, i):
    """Bring data back to cpu"""
    if is_dict(data[list(data.keys())[0]]):
        return data_to_cpu2(data, i)
    else:
        return data_to_cpu1(data, i)


def data_to_cpu1(data, i):
    """Bring data back to cpu (1 level)"""
    return {k1: val[0][i].cpu() if is_list(val) else val[i].cpu()
            for k1, val in data.items()}


def data_to_cpu2(data, i):
    """Bring data back to cpu (2 levels)"""
    return {k1: {k2: v2[0][i].cpu() if is_list(v2) else v2[i].cpu()
                 for k2, v2 in v1.items()} for k1, v1 in data.items()}


class Saver:
    def __init__(self, cfg, ckpt=None):
        """Saver manager class.

        Parameters
        ----------
        cfg : Config
            Configuration file with saver parameters
        ckpt : str, optional
            Path to checkpoint model, by default None
        """
        self.folder = cfg_has(cfg, 'folder', None)

        self.rgb = make_list(cfg.rgb) if cfg_has(cfg, 'rgb') else []
        self.depth = make_list(cfg.depth) if cfg_has(cfg, 'depth') else []
        self.normals = make_list(cfg.normals) if cfg_has(cfg, 'normals') else []
        self.pose = make_list(cfg.pose) if cfg_has(cfg, 'pose') else []
        self.intrinsics = make_list(cfg.intrinsics) if cfg_has(cfg, 'intrinsics') else []
        self.optical_flow = make_list(cfg.optical_flow) if cfg_has(cfg, 'optical_flow') else []
        self.stddev = make_list(cfg.stddev) if cfg_has(cfg, 'stddev') else []

        self.store_data = cfg_has(cfg, 'store_data', False)
        self.separate = cfg.has('separate', False)
        self.broken = cfg.has('broken', False)

        self.ckpt = None if ckpt is None else \
            os.path.splitext(os.path.basename(ckpt))[0]

        self.naming = cfg_has(cfg, 'naming', 'filename')
        assert self.naming in ['filename', 'splitname'], \
            'Invalid naming for saver: {}'.format(self.naming)

    def get_filename(self, path, batch, idx):
        """Get filename from batch information"""
        if self.naming == 'filename':
            filenames = [os.path.join(path, batch['filename'][(0, 0)][i]) for i in range(len(idx))] #.replace('{}', 'rgb')
            for filename in filenames:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            return filenames
        elif self.naming == 'splitname':
            if self.separate:
                return [os.path.join(path, '%010d' % idx[i], '%010d' % idx[i]) for i in range(len(idx))]
            else:
                return [os.path.join(path, '%010d' % idx[i]) for i in range(len(idx))]
        else:
            raise NotImplementedError('Invalid naming for saver: {}'.format(self.naming))

    def save_data(self, batch, output, prefix):
        """Save data from batch and output"""

        if self.folder is None:
            return

        idx = batch['idx']
        predictions = output['predictions']

        path = os.path.join(self.folder, prefix)
        if self.ckpt is not None:
            path = os.path.join(path, self.ckpt)
        os.makedirs(path, exist_ok=True)

        self.save(batch, predictions, path, idx)

    def save(self, batch, predictions, path, idx):
        """Loops over all filenames and saves data"""

        filenames = self.get_filename(path, batch, idx)
        for i, filename in enumerate(filenames):
            self.save_item(batch, predictions, filename, i)

    def save_item(self, batch, predictions, filename, i):
        """Save individual batch labels and predictions to file"""

        data = {}

        # Save GT data from batch
        for key in batch.keys():

            if key.startswith('rgb'):
                if self.broken:
                    batch[key] = expand_and_break(batch[key], 1, 4)
                data[key + '_gt'] = data_to_cpu(batch[key], i)
                for ctx in batch[key].keys():
                    rgb = batch[key][ctx][i].cpu()
                    if 'gt' in self.rgb:
                        if rgb.dim() == 5:
                            for j in range(rgb.shape[1]):
                                write_image('%s_%s_(%d_%d)_gt.png' % (filename, key, j, ctx),
                                            rgb[:, j])
                        else:
                            write_image('%s_%s_(%s)_gt.png' % (filename, key, ctx_str(ctx)),
                                        rgb)

            if key.startswith('depth'):
                if self.broken:
                    batch[key] = expand_and_break(batch[key], 1, 4)
                data[key + '_gt'] = data_to_cpu(batch[key], i)
                for ctx in batch[key].keys():
                    depth = batch[key][ctx][i].cpu()
                    if 'gt_png' in self.depth:
                        write_depth('%s_%s_(%s)_gt.png' % (filename, key, ctx_str(ctx)),
                                    depth)
                    if 'gt_npz' in self.depth:
                        write_depth('%s_%s_(%s)_gt.npz' % (filename, key, ctx_str(ctx)),
                                    depth)
                    if 'gt_viz' in self.depth:
                        write_image('%s_%s_(%s)_gt_viz.png' % (filename, key, ctx_str(ctx)),
                                    viz_depth(depth, filter_zeros=True))

            if key.startswith('pose'):
                if self.broken:
                    batch[key] = expand_and_break(batch[key], 1, 3)
                pose = data_to_cpu(batch[key], i)
                data[key + '_gt'] = pose
                if 'gt' in self.pose:
                    write_pickle('%s_%s_gt' % (filename, key),
                                 pose)

            if key.startswith('intrinsics'):
                if self.broken:
                    batch[key] = expand_and_break(batch[key], 1, 3)
                intrinsics = data_to_cpu(batch[key], i)
                data[key + '_gt'] = intrinsics
                if 'gt' in self.intrinsics:
                    write_pickle('%s_%s_gt' % (filename, key),
                                 intrinsics)

        # Save predictions
        for key in predictions.keys():

            if key.startswith('rgb'):
                data[key + '_pred'] = data_to_cpu(predictions[key], i)
                for ctx in predictions[key].keys():
                    rgb = predictions[key][ctx][0][i].cpu()
                    if 'pred' in self.rgb:
                        if rgb.dim() == 5:
                            for j in range(rgb.shape[1]):
                                write_image('%s_%s_(%d_%d)_pred.png' % (filename, key, j, ctx),
                                            rgb[:, j])
                        else:
                            write_image('%s_%s_(%s)_pred.png' % (filename, key, ctx_str(ctx)),
                                        rgb)

            if key.startswith('depth'):
                data[key + '_pred'] = data_to_cpu(predictions[key], i)
                for tgt in predictions[key].keys():
                    if is_dict(predictions[key][tgt]):
                        for ctx in predictions[key][tgt].keys():
                            depth = predictions[key][tgt][ctx][0][i].cpu()
                            string = '%s_%s_(%s)_(%s)_pred' % (filename, key, ctx_str(tgt), ctx_str(ctx))
                            if 'png' in self.depth:
                                write_depth('%s.png' % string, depth)
                            if 'npz' in self.depth:
                                write_depth('%s.npz' % string, depth)
                            if 'viz' in self.depth:
                                write_image('%s_viz.png' % string, viz_depth(depth))
                    else:
                        depth = predictions[key][tgt][0][i].cpu()
                        string = '%s_%s_(%s)_pred' % (filename, key, ctx_str(tgt))
                        if 'png' in self.depth:
                            write_depth('%s.png' % string, depth)
                        if 'npz' in self.depth:
                            write_depth('%s.npz' % string, depth)
                        if 'viz' in self.depth:
                            write_image('%s_viz.png' % string, viz_depth(depth))

            if key.startswith('normals'):
                data[key + '_pred'] = data_to_cpu(predictions[key], i)
                for ctx in predictions[key].keys():
                    normals = predictions[key][ctx][0][i].cpu()
                    if 'npz' in self.normals:
                        write_normals('%s_%s_(%s)_pred.npz' % (filename, key, ctx_str(ctx)),
                                      normals)
                    if 'viz' in self.normals:
                        write_image('%s_%s_(%s)_pred_viz.png' % (filename, key, ctx_str(ctx)),
                                    viz_normals(normals))

            if key.startswith('pose'):
                pose = data_to_cpu(predictions[key], i)
                data[key + '_pred'] = pose
                if 'pred' in self.pose:
                    write_pickle('%s_%s_pred' % (filename, key),
                                 pose)

            if key.startswith('intrinsics'):
                intrinsics = data_to_cpu(predictions[key], i)
                data[key + '_pred'] = intrinsics
                if 'pred' in self.intrinsics:
                    write_pickle('%s_%s_pred' % (filename, key),
                                 intrinsics)

            if key.startswith('optical_flow'):
                data[key + '_pred'] = data_to_cpu(predictions[key], i)
                for tgt in predictions[key].keys():
                    for ctx in predictions[key][tgt].keys():
                        optical_flow = predictions[key][tgt][ctx][0][i].cpu()
                        if 'npz' in self.optical_flow:
                            write_optical_flow('%s_%s_(%s)_(%s)_pred.npz' % (filename, key, ctx_str(tgt), ctx_str(ctx)),
                                               optical_flow)
                        if 'viz' in self.optical_flow:
                            write_image('%s_%s_(%s)_(%s)_pred_viz.png' % (filename, key, ctx_str(tgt), ctx_str(ctx)),
                                        viz_optical_flow(optical_flow, clip_value=100.0))

            if key.startswith('mask') or key.startswith('valid'):
                data[key + '_pred'] = data_to_cpu(predictions[key], i)
                if is_dict(predictions[key]):
                    for tgt in predictions[key].keys():
                        mask = predictions[key][tgt][0][i].cpu()
                        write_image('%s_%s_(%s)_pred_viz.png' % (filename, key, ctx_str(tgt)),
                                    mask)

            if key.startswith('stddev'):
                data[key + '_pred'] = data_to_cpu(predictions[key], i)
                if is_dict(predictions[key]):
                    for tgt in predictions[key].keys():
                        stddev = predictions[key][tgt][0][i].cpu()
                        if 'viz' in self.stddev:
                            write_image('%s_%s_(%s)_pred_viz.png' % (filename, key, ctx_str(tgt)),
                                        viz_stddev(stddev))

        # Save data pickle if requested
        if self.store_data:
            write_pickle('%s' % filename, data)

        return data
