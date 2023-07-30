# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch

from vidar.metrics.base import BaseEvaluation
from vidar.utils.data import remove_nones_dict, get_from_dict
from vidar.utils.distributed import on_rank_0
from vidar.utils.logging import pcolor


class OpticalFlowEvaluation(BaseEvaluation):
    """OPtical flow evaluation class"""
    def __init__(self, cfg):
        super().__init__(cfg,
            name='optical_flow', task='optical_flow',
            metrics=('EPE', '1px', '3px', '5px'),
        )

    @staticmethod
    def reduce_fn(metrics, seen):
        """Reduce function for distributed evaluation"""
        valid = seen.view(-1) > 0
        return (metrics[valid] / seen.view(-1, 1)[valid]).mean(0)

    def populate_metrics_dict(self, metrics, metrics_dict, prefix):
        """Populate metrics dictionary"""
        for metric in metrics:
            if metric.startswith(self.name):
                name, suffix = metric.split('|')
                for i, key in enumerate(self.metrics):
                    metrics_dict[f'{prefix}-{name}|{key}_{suffix}'] = \
                        metrics[metric][i].item()

    @on_rank_0
    def print(self, reduced_data, prefixes, task):
        """Print evaluation results"""
        print()
        print(self.horz_line)
        print(self.metr_line.format(*((task.upper(),) + self.metrics)))
        for n, metrics in enumerate(reduced_data):
            if sum([self.name in key for key in metrics.keys()]) == 0:
                continue
            print(self.horz_line)
            print(self.wrap(pcolor('*** {:<70}'.format(prefixes[n]), **self.font1)))
            print(self.horz_line)
            for key, metric in metrics.items():
                if self.name in key and task.count('|') == key.count('|') - 1 and task.split('|')[-1] in key:
                    print(self.wrap(pcolor(self.outp_line.format(
                        *((key.upper(),) + tuple(metric.tolist()))), **self.font2)))
        print(self.horz_line)
        print()

    def compute(self, gt, pred, mask=None):
        """Compute optical flow metrics"""
        metrics = []
        for i, (pred_i, gt_i) in enumerate(zip(pred, gt)):
            mask_i = mask[i] if mask is not None else None

            epe = torch.sum((pred_i - gt_i) ** 2, dim=0).sqrt().view(-1)

            px1 = (epe > 1).float().mean()
            px3 = (epe > 3).float().mean()
            px5 = (epe > 5).float().mean()
            epe = epe.mean()

            metrics.append([epe, px1, px3, px5])

        return torch.tensor(metrics, dtype=gt.dtype)

    def evaluate(self, batch, output, task, flipped_output=None):
        """Evaluate optical flow metrics"""
        metrics, predictions = {}, {}
        if self.name not in batch:
            return metrics, predictions
        # For each output item
        for key, val in output.items():
            # If it corresponds to this task
            if key.startswith(self.name) and 'debug' not in key:
                # Loop over every target
                for tgt in val.keys():
                    gt_tgt = get_from_dict(batch[self.name], tgt)
                    if gt_tgt is not None:
                        # Loop over every context
                        for ctx in val[tgt].keys():
                            gt_tgt_ctx = get_from_dict(gt_tgt, ctx)
                            if gt_tgt_ctx is not None:
                                # Loop over every scale
                                for i in range(1 if self.only_first else len(val[tgt][ctx])):
                                    pred = val[tgt][ctx][i]
                                    suffix = self.suffix_ctx(tgt, ctx, i)
                                    metrics[f'{key}|{suffix}'] = \
                                        self.compute(
                                            gt=gt_tgt_ctx,
                                            pred=pred,
                                            mask=None,
                                        )
        return remove_nones_dict(metrics), predictions
