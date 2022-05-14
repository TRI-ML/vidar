# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as F
from torch import nn

from vidar.arch.networks.layers.depthformer.context_adjustment import ContextAdjustmentLayer
from vidar.utils.volume import compute_depth_bin


class RegressionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cal = ContextAdjustmentLayer(cfg.context_adjustment)
        self.phi = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.monocular = True

    @staticmethod
    def _compute_unscaled_pos_shift(w, device):
        return torch.linspace(0, w - 1, w)[None, None, None, :].to(device)

    @staticmethod
    def _compute_low_res_depth(pos_shift, attn_weight):
        high_response = torch.argmax(attn_weight, dim=-1)  # NxHxW
        response_range = torch.stack([high_response - 1, high_response, high_response + 1], dim=-1)
        attn_weight_pad = F.pad(attn_weight, [1, 1], value=0.0)
        attn_weight_rw = torch.gather(attn_weight_pad, -1, response_range + 1)

        norm = attn_weight_rw.sum(-1, keepdim=True)
        norm[norm < 0.1] = 1.0

        attn_weight_rw = attn_weight_rw / norm
        pos_pad = F.pad(pos_shift, [1, 1]).expand_as(attn_weight_pad).clone()
        pos_pad[..., -1] = pos_shift[..., -1] + 1
        pos_rw = torch.gather(pos_pad, -1, response_range + 1)
        depth_pred_low_res = (attn_weight_rw * pos_rw)

        depth_pred_low_res = depth_pred_low_res.sum(-1)

        return depth_pred_low_res, norm, high_response

    def upsample(self, x, depth_pred, scale=1.0):
        _, _, h, w = x.size()
        depth_pred_attn = depth_pred * scale
        depth_pred = F.interpolate(depth_pred_attn[None,], size=(h, w), mode='nearest')
        depth_pred_final = self.cal(depth_pred, x)
        return depth_pred_final.squeeze(1), depth_pred_attn.squeeze(1)

    def softmax(self, attn):
        bs, h, w, d = attn.shape
        similarity_matrix = torch.cat([attn, self.phi.expand(bs, h, w, 1).to(attn.device)], -1)
        attn_softmax = F.softmax(similarity_matrix, dim=-1)
        return attn_softmax

    def forward(self, attn_weight, target, context, sampled_rows, sampled_cols, min_depth, max_depth, num_bins):

        stride = [1]

        outputs = []
        for s in stride:
            output = self.forward2(
                attn_weight, target, sampled_cols, min_depth, max_depth, num_bins, s)
            outputs.append(output)
        final_output = {}
        for key in outputs[0].keys():
            final_output[key] = [o[key] for o in outputs]
        return final_output

    def forward2(self, attn_weight, target, sampled_cols, min_depth, max_depth, num_bins, stride=1):

        bs, _, h, w = target.size()
        output = {}

        if stride > 1:
            shape = list(attn_weight.shape)
            shape[-1] = shape[-1] // stride
            attn_weight_tmp = torch.zeros(shape, dtype=attn_weight.dtype, device=attn_weight.device)
            for i in range(0, shape[-1]):
                attn_weight_tmp[..., i] = attn_weight[..., i * stride:(i + 1) * stride].mean(-1)
            attn_weight = attn_weight_tmp

        attn_ot = self.softmax(attn_weight)
        attn_ot = attn_ot[..., :-1]
        output['attn_weight_softmax'] = attn_ot

        pos_shift = self._compute_unscaled_pos_shift(attn_weight.shape[3], attn_weight.device)

        depth_pred_low_res1, matched_attn1, high_response1 = self._compute_low_res_depth(pos_shift, attn_ot)
        depth_pred_low_res2, matched_attn2, high_response2 = self._compute_low_res_depth(pos_shift, attn_ot)

        output['high_response'] = high_response1

        if sampled_cols is not None:
            output['depth_pred1'], output['depth_pred1_low'] = \
                self.upsample(target, depth_pred_low_res1)
            output['depth_pred2'], output['depth_pred2_low'] = \
                self.upsample(target, depth_pred_low_res2)
        else:
            output['depth_pred_low'] = depth_pred_low_res1
            output['depth_pred'] = depth_pred_low_res1

        if self.monocular:

            num_bins = num_bins // stride

            depth1 = compute_depth_bin(min_depth, max_depth, num_bins, output['depth_pred1'])
            depth1_low = compute_depth_bin(min_depth, max_depth, num_bins, output['depth_pred1_low'])

            depth2 = compute_depth_bin(min_depth, max_depth, num_bins, output['depth_pred2'])
            depth2_low = compute_depth_bin(min_depth, max_depth, num_bins, output['depth_pred2_low'])

            output['depth1'] = depth1
            output['depth1_low'] = depth1_low

            output['depth2'] = depth2
            output['depth2_low'] = depth2_low

        return output
