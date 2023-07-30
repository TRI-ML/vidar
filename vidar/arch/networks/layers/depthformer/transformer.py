# Copyright 2023 Toyota Research Institute.  All rights reserved.

import copy
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from vidar.arch.losses.SSIMLoss import SSIMLoss
from vidar.utils.tensor import grid_sample
from vidar.utils.volume import compute_depth_bins, compute_depth_bin


def get_clones(module, N):
    """Create clones of a module"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def prepareA(feat):
    """Reorganize features in one way"""
    return feat.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)


def prepareB(x):
    """Reorganize features in another way"""
    d, c, h, w = x.shape
    return x.permute(1, 2, 3, 0).reshape(c, h * w, d).permute(2, 1, 0)


def unprepare(feat, shape):
    """Return features back to original shape"""
    b, c, h, w = shape
    return feat.permute(2, 0, 1).reshape(c, w, h, b).permute(3, 0, 2, 1)


class Transformer(nn.Module):
    """
    Transformer network for Feature Matching (https://arxiv.org/abs/2204.07616)

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.channel_dim
        self.num_attn_layers = cfg.num_attn_layers

        self_attn_layer = TransformerSelfAttnLayer(self.hidden_dim, cfg.nheads)
        self.self_attn_layers = get_clones(self_attn_layer, self.num_attn_layers)

        cross_attn_layer = TransformerCrossAttnLayer(self.hidden_dim, cfg.nheads)
        self.cross_attn_layers = get_clones(cross_attn_layer, self.num_attn_layers)

        self.norm = nn.LayerNorm(self.hidden_dim)

        self.grid_sample = partial(
            grid_sample, padding_mode='zeros', mode='bilinear')

        self.grid_sample_nearest = partial(
            grid_sample, padding_mode='zeros', mode='nearest')

    def _alternating_attn(self, feat1, feat2,
                          cam=None, min_depth=None, max_depth=None, num_bins=None):
        """Perform self- and cross-attention between two feature maps"""
        device = feat1.device
        cam = cam.to(device)
        h, w = cam.hw

        depth_bins = compute_depth_bins(min_depth, max_depth, num_bins, 'sid').to(device)

        ones = torch.ones((1, h, w), dtype=feat1.dtype, device=device)
        warped_depth = torch.stack([depth * ones for depth in depth_bins], 1)
        coords = cam.coords_from_cost_volume(warped_depth)[0]

        coords[coords < -1] = -2
        coords[coords > +1] = +2

        repeated_feat2 = feat2.repeat([num_bins, 1, 1, 1])
        warped = self.grid_sample(repeated_feat2, coords.type(repeated_feat2.dtype))

        repeated_ones = ones.repeat([num_bins, 1, 1, 1])
        warped_mask = self.grid_sample_nearest(repeated_ones, coords.type(repeated_ones.dtype))

        with torch.no_grad():
            ssim_volume = SSIMLoss()(feat1, warped)['loss'].mean(1).unsqueeze(0)
            lowest_cost = 1. / compute_depth_bin(min_depth, max_depth, num_bins, torch.min(ssim_volume, 1)[1])

        feat1 = prepareB(feat1)
        feat2 = prepareB(warped)

        attn_weight = None
        for idx, (self_attn, cross_attn) in \
                enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
            feat1 = self_attn(feat1)
            feat1, feat2, attn_weight = cross_attn(feat1, feat2)

        return {
            'attn_weight': attn_weight,
            'warped_mask': warped_mask,
            'ssim_lowest_cost': lowest_cost,
            'ssim_cost_volume': ssim_volume,
        }

    def forward(self, feat1, feat2, cam=None, min_depth=None, max_depth=None, num_bins=None):
        """Network forward pass"""

        bs, c, hn, w = feat1.shape

        transformer_output = self._alternating_attn(
            feat1, feat2, cam=cam, min_depth=min_depth, max_depth=max_depth, num_bins=num_bins)
        transformer_output['attn_weight'] = \
            transformer_output['attn_weight'].view(bs, hn, w, num_bins)

        return transformer_output


class TransformerSelfAttnLayer(nn.Module):
    """Self-attention layer for transformers"""
    def __init__(self, hidden_dim, nheads):
        super().__init__()
        self.self_attn = MultiheadAttentionRelative(hidden_dim, nheads)

        self.norm1 = nn.LayerNorm(hidden_dim)

    def forward(self, feat):
        feat_out = self.norm1(feat)
        feat_out, _, _ = self.self_attn(query=feat_out, key=feat_out, value=feat_out)
        return feat + feat_out


class TransformerCrossAttnLayer(nn.Module):
    """Cross-attention layer for transformers"""
    def __init__(self, hidden_dim, nheads):
        super().__init__()
        self.cross_attn = MultiheadAttentionRelative(hidden_dim, nheads)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, feat1, feat2):

        feat1_2 = self.norm1(feat1)
        feat2_2 = self.norm1(feat2)

        feat1_2, attn_weight, raw_attn = self.cross_attn(query=feat1_2, key=feat2_2, value=feat2_2)
        feat1 = feat1 + feat1_2

        return feat1, feat2, raw_attn

    @torch.no_grad()
    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask[mask == 1] = float('-inf')
        return mask


class MultiheadAttentionRelative(nn.MultiheadAttention):
    """Multi-head attention layer"""
    def __init__(self, embed_dim, num_heads):
        super().__init__(
            embed_dim, num_heads, dropout=0.0, bias=True,
            add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)

    def forward(self, query, key, value):
        """Network forward pass for attention calculation and feature reorganization"""

        w2, bsz2, embed_dim2 = key.size()
        w, bsz, embed_dim = query.size()

        head_dim = embed_dim // self.num_heads

        if torch.equal(query, key) and torch.equal(key, value):
            q, k, v = F.linear(
                query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        elif torch.equal(key, value):
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = self.in_proj_bias
                _start = embed_dim
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)
        else:
            raise ValueError('Invalid key/query/value')

        scaling = float(head_dim) ** - 0.5
        q = q * scaling

        q = q.contiguous().view(w, bsz, self.num_heads, head_dim)
        if k is not None:
            k = k.contiguous().view(-1, bsz, self.num_heads, head_dim)
        if v is not None:
            v = v.contiguous().view(-1, bsz, self.num_heads, head_dim)

        attn = torch.einsum('wnec,vnec->newv', q, k)
        raw_attn = attn
        attn = F.softmax(attn, dim=-1)

        v_out = torch.bmm(attn.view(bsz * self.num_heads, w, w2),
                        v.permute(1, 2, 0, 3).view(bsz * self.num_heads, w2, head_dim))
        v_out = v_out.reshape(bsz, self.num_heads, w, head_dim).permute(2, 0, 1, 3).reshape(w, bsz, embed_dim)
        v_out = F.linear(v_out, self.out_proj.weight, self.out_proj.bias)

        attn = attn.sum(dim=1) / self.num_heads
        raw_attn = raw_attn.sum(dim=1)

        return v_out, attn, raw_attn
