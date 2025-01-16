# Copyright 2023 Toyota Research Institute.  All rights reserved.

from functools import partial

import lpips
import skimage
import skimage.measure
import skimage.metrics
import torch

from knk_vision.vidar.vidar.metrics.base import BaseEvaluation
from knk_vision.vidar.vidar.utils.data import remove_nones_dict
from knk_vision.vidar.vidar.utils.distributed import on_rank_0
from knk_vision.vidar.vidar.utils.logging import pcolor
from knk_vision.vidar.vidar.utils.tensor import interpolate


class SSIM:
    """Structure Similarity Index Metric"""
    def __init__(self):
        self.criterion = partial(skimage.metrics.structural_similarity, channel_axis=2)

    def __call__(self, pred, gt):
        return self.criterion(
            pred.permute(1, 2, 0).cpu().numpy(),
            gt.permute(1, 2, 0).cpu().numpy(),
        )


class PSNR:
    """Peak Signal to Noise Ratio"""
    def __init__(self):
        self.criterion = partial(skimage.metrics.peak_signal_noise_ratio)

    def __call__(self, pred, gt):
        return self.criterion(
            pred.permute(1, 2, 0).cpu().numpy(),
            gt.permute(1, 2, 0).cpu().numpy(),
        )


class LPIPS:
    """Learned Perceptual Image Patch Similarity"""
    def __init__(self):
        self.criterion = lpips.LPIPS(net='vgg', verbose=False)

    def __call__(self, pred, gt):
        return self.criterion(pred, gt)


class ImageEvaluation(BaseEvaluation):
    """Image evalution class"""
    def __init__(self, cfg):
        super().__init__(cfg,
            name='rgb', task='rgb',
            metrics=('SSIM', 'PSNR'),
        )

        self.ssim = SSIM()
        self.psnr = PSNR()
        self.lpips = LPIPS()

        self.crop_edges = cfg.has('crop_edges', None)

        if cfg.has('resize'):
            self.resize = partial(interpolate, mode='bilinear', size=cfg.resize)
        else:
            self.resize = None

    @staticmethod
    def reduce_fn(metrics, seen, stride=10000):
        """Reduce function for distributed evaluation"""
        split_mean, split_valid = [], []
        for i in range(0, len(seen), stride):
            valid_i = seen[i:i+stride] > 0
            metrics_i = metrics[i:i+stride][valid_i]
            split_mean.append(metrics_i.sum(0))
            split_valid.append(valid_i.sum().item())
        return torch.stack(split_mean, 0).sum(0) / sum(split_valid)

    def populate_metrics_dict(self, metrics, metrics_dict, prefix):
        """Populate metrics dictionary"""
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
            print(self.wrap(pcolor('*** {:<59}'.format(prefixes[n]), **self.font1)))
            print(self.horz_line)
            for key, metric in valid_keys_and_metrics.items():
                print(self.wrap(pcolor(self.outp_line.format(
                    *((key.upper(),) + tuple(metric.tolist()))), **self.font2)))
        print(self.horz_line)
        print()

    def compute(self, gt, pred):
        """Compute image metrics"""
        # For each batch sample
        metrics = []
        for pred_i, gt_i in zip(pred, gt):

            gt_i = gt_i.unsqueeze(0).clone()
            pred_i = pred_i.unsqueeze(0).clone()
            pred_i = self.interp_bilinear(pred_i, gt_i)

            if self.resize is not None:
                gt_i = self.resize(gt_i)
                pred_i = self.resize(pred_i)

            gt_i = gt_i.clamp(min=0.0, max=1.0)
            pred_i = pred_i.clamp(min=0.0, max=1.0)

            if self.crop_edges is not None:
                h, w = gt_i.shape[-2:]
                crop_h = int(self.crop_edges * h)
                crop_w = int(self.crop_edges * w)
                gt_i = gt_i[:, :, crop_h:-crop_h, crop_w:-crop_w]
                pred_i = pred_i[:, :, crop_h:-crop_h, crop_w:-crop_w]

            ssim = self.ssim(pred_i[0], gt_i[0])
            psnr = self.psnr(pred_i[0], gt_i[0])

            metrics.append([ssim, psnr])

        # Return metrics
        return torch.tensor(metrics, dtype=gt.dtype)

    def evaluate(self, batch, output, task, flipped_output=None):
        """Evaluate image metrics"""
        midfix = (task.split('|')[-1] + '|') if '|' in task else ''
        metrics, predictions = {}, {}
        if self.name not in batch:
            return metrics, predictions
        # For each output item
        for key, val in output.items():
            # If it corresponds to this task
            if key.startswith(self.name) and 'debug' not in key:
                # Loop over every context
                for ctx in val.keys():
                    # Loop over every scale
                    for i in range(1 if self.only_first else len(val[ctx])):

                        pred = val[ctx][i]
                        gt = batch[self.name][ctx]

                        if i > 0:
                            pred = self.interp_bilinear(pred, val[ctx][0])

                        if pred.dim() == 4:
                            suffix = self.suffix_single(ctx, i)
                            for mode in self.modes:
                                metrics[f'{key}|{midfix}{mode}{suffix}'] = \
                                    self.compute(
                                        gt=gt,
                                        pred=pred
                                    )
                        elif pred.dim() == 5:
                            for j in range(pred.shape[1]):
                                suffix = self.suffix_multi(ctx, i, j)
                                for mode in self.modes:
                                    metrics[f'{key}|{midfix}{mode}{suffix}'] = \
                                        self.compute(
                                            gt=gt[:, j],
                                            pred=pred[:, j]
                                        )

        return remove_nones_dict(metrics), predictions

