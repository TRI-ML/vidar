# Copyright 2023 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch

from vidar.metrics.base import BaseEvaluation
from vidar.metrics.utils import create_crop_mask, scale_output
from vidar.utils.config import cfg_has
from vidar.utils.data import remove_nones_dict
from vidar.utils.depth import post_process_depth, scale_and_shift_pred
from vidar.utils.distributed import on_rank_0
from vidar.utils.logging import pcolor
from vidar.utils.types import is_dict


class DepthEvaluation(BaseEvaluation):
    """Depth evalution class"""
    def __init__(self, cfg):
        super().__init__(cfg,
            name='depth', task='depth',
            metrics=('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'silog', 'a1', 'a2', 'a3'),
        )

        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth
        self.crop = cfg_has(cfg, 'crop', '')
        self.scale_output = cfg_has(cfg, 'scale_output', 'resize')

        self.post_process = cfg_has(cfg, 'post_process', False)
        self.median_scaling = cfg_has(cfg, 'median_scaling', False)
        self.shift_scaling = cfg_has(cfg, 'shift_scaling', False)
        self.valid_threshold = cfg.has('valid_threshold', 0)
        self.sparse_predictions = cfg.has('sparse_predictions', False)

        if self.post_process:
            self.modes += ['pp']
        if self.median_scaling:
            self.modes += ['md']
            if self.post_process:
                self.modes += ['pp_md']
        if self.shift_scaling:
            self.modes += ['ss']
            if self.post_process:
                self.modes += ['pp_ss']

    @staticmethod
    def reduce_fn(metrics, seen, stride=10000):
        """Reduce function for distributed evaluation"""
        split_mean, split_valid = [], []
        for i in range(0, len(seen), stride):
            valid_i = seen[i:i+stride] > 0
            seen_i = seen[i:i+stride][valid_i]
            metrics_i = metrics[i:i+stride][valid_i]
            split_mean.append(metrics_i.sum(0))
            split_valid.append(seen_i.sum().item())
        return torch.stack(split_mean, 0).sum(0) / sum(split_valid)

    def populate_metrics_dict(self, metrics, metrics_dict, prefix):
        """Populate metrics dictionary with metrics from metrics dict"""
        for metric in metrics:
            if metric.startswith(self.task):
                name, suffix = metric.rsplit('|', 1)
                for i, key in enumerate(self.metrics):
                    metrics_dict[f'{prefix}-{name}|{key}_{suffix}'] = \
                        metrics[metric][0][i].item()

    @on_rank_0
    def print(self, reduced_data, prefixes, task):
        """Print evaluation results"""
        print()
        print(self.horz_line)
        print(self.metr_line.format(*((task.upper(),) + self.metrics)))
        for n, metrics in enumerate(reduced_data):
            if sum([self.name in key for key in metrics.keys()]) == 0:
                continue
            valid_keys_and_metrics, seen = self.get_valid_keys_and_metrics(metrics, task)
            if len(valid_keys_and_metrics) == 0:
                continue
            print(self.horz_line)
            print(self.wrap(pcolor('*** {:<111}{:>6}/{:<6}'.format(prefixes[n], seen[0], seen[1]), **self.font1)))
            print(self.horz_line)
            for key, metric in valid_keys_and_metrics.items():
                print(self.wrap(pcolor(self.outp_line.format(
                    *((key.upper(),) + tuple(metric.tolist()))), **self.font2)))
        print(self.horz_line)
        print()

    def compute(self, gt, pred, median_scaling=False, shift_scaling=False, mask=None):
        """Compute depth metrics"""
        assert not median_scaling or not shift_scaling
        # Match predicted depth map to ground-truth resolution
        pred = scale_output(pred, gt, self.scale_output)
        # Create crop mask if requested
        crop_mask = create_crop_mask(self.crop, gt)
        # For each batch sample
        metrics = []
        for i, (pred_i, gt_i) in enumerate(zip(pred, gt)):

            # Squeeze GT and PRED
            gt_i, pred_i = torch.squeeze(gt_i), torch.squeeze(pred_i)
            mask_i = torch.squeeze(mask[i]) if mask is not None else None

            # Keep valid pixels (min/max depth and crop)
            valid = (gt_i > self.min_depth) & (gt_i < self.max_depth)
            # Remove invalid predicted pixels as well
            if self.sparse_predictions:
                valid = valid & (pred_i > 0)
            # Apply crop mask if requested
            valid = valid & crop_mask.bool() if crop_mask is not None else valid
            # Apply provided mask if available
            valid = valid & mask_i.bool() if mask is not None else valid

            # Invalid evaluation
            if valid.sum() <= self.valid_threshold:
                metrics.append([-1.] * 8)
                continue

            # Keep only valid pixels
            gt_i, pred_i = gt_i[valid], pred_i[valid]
            # GT median scaling if needed
            if shift_scaling:
                _, _, pred_i = scale_and_shift_pred(pred_i, gt_i)
            if median_scaling:
                pred_i = pred_i * torch.median(gt_i) / torch.median(pred_i)
            # Clamp PRED depth values to min/max values
            pred_i = pred_i.clamp(self.min_depth, self.max_depth)

            # Calculate depth metrics

            thresh = torch.max((gt_i / pred_i), (pred_i / gt_i))
            a1 = (thresh < 1.25).float().mean()
            a2 = (thresh < 1.25 ** 2).float().mean()
            a3 = (thresh < 1.25 ** 3).float().mean()

            diff_i = gt_i - pred_i
            abs_rel = torch.mean(torch.abs(diff_i) / gt_i)
            sq_rel = torch.mean(diff_i ** 2 / gt_i)
            rmse = torch.sqrt(torch.mean(diff_i ** 2))
            rmse_log = torch.sqrt(torch.mean((torch.log(gt_i) - torch.log(pred_i)) ** 2))

            err = torch.log(pred_i) - torch.log(gt_i)
            silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

            metrics.append([abs_rel, sq_rel, rmse, rmse_log, silog, a1, a2, a3])

        # Return metrics
        return torch.tensor(metrics, dtype=gt.dtype)

    def evaluate(self, batch, output, task, flipped_output=None):
        """Evaluate depth metrics"""
        midfix = (task.split('|')[-1] + '|') if '|' in task else ''
        metrics, predictions = {}, {}
        if self.task not in batch:
            return metrics, predictions
        # For each output item
        for key, val in output.items():
            # If it corresponds to this task
            if key.startswith(self.task) and 'debug' not in key:
                # Loop over every context
                for tgt in val.keys():
                    gt = batch[self.task][tgt]
                    if gt.max() == 0:
                        continue
                    if is_dict(val[tgt]):
                        for ctx in val[tgt].keys():
                            # Loop over every scale
                            for i in range(1 if self.only_first else len(val[tgt][ctx])):

                                pred = val[tgt][ctx][i]

                                if self.post_process:
                                    pred_flipped = flipped_output[key][tgt][ctx][i]
                                    pred_pp = post_process_depth(pred, pred_flipped, method='mean')
                                else:
                                    pred_pp = None

                                if i > 0:
                                    pred = self.interp_nearest(pred, val[tgt][ctx][0])
                                    if self.post_process:
                                        pred_pp = self.interp_nearest(pred_pp, val[tgt][ctx][0])

                                suffix = self.suffix_single2(tgt, ctx, i)
                                for mode in self.modes:
                                    metrics[f'{key}|{midfix}{mode}{suffix}'] = \
                                        self.compute(
                                            gt=gt,
                                            pred=pred_pp if 'pp' in mode else pred,
                                            median_scaling='md' in mode,
                                            shift_scaling='ss' in mode,
                                            mask=None,
                                        )
                    else:
                        # Loop over every scale
                        for i in range(1 if self.only_first else len(val[tgt])):

                            pred = val[tgt][i]

                            if self.post_process:
                                pred_flipped = flipped_output[key][tgt][i]
                                pred_pp = post_process_depth(pred, pred_flipped, method='mean')
                            else:
                                pred_pp = None

                            if i > 0:
                                pred = self.interp_nearest(pred, val[tgt][0])
                                if self.post_process:
                                    pred_pp = self.interp_nearest(pred_pp, val[tgt][0])

                            suffix = self.suffix_single(tgt, i)
                            for mode in self.modes:
                                metrics[f'{key}|{midfix}{mode}{suffix}'] = \
                                    self.compute(
                                        gt=gt,
                                        pred=pred_pp if 'pp' in mode else pred,
                                        median_scaling='md' in mode,
                                        shift_scaling='ss' in mode,
                                        mask=None,
                                    )

        return remove_nones_dict(metrics), predictions

