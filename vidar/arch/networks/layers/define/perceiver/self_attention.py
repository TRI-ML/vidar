# Copyright 2023 Toyota Research Institute.  All rights reserved.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverSelfAttention(nn.Module):
    """Perceiver Self-Attention module, used as part of the Perceiver IO model."""
    def __init__(
        self,
        is_cross_attention,
        qk_channels,
        v_channels,
        num_heads,
        q_dim,
        kv_dim,
        attention_dropout,
    ):
        super().__init__()

        self.num_heads = num_heads

        if qk_channels is None:
            qk_channels = q_dim
        if v_channels is None:
            v_channels = qk_channels

        if qk_channels % num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by num_heads ({num_heads}).")
        if v_channels % num_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by num_heads ({num_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads

        self.layernorm1 = nn.LayerNorm(q_dim)
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Key/query/value projections
        self.query = nn.Linear(q_dim, qk_channels)
        self.key = nn.Linear(kv_dim, qk_channels)
        self.value = nn.Linear(kv_dim, v_channels)

        # Optional dropout
        self.dropout = nn.Dropout(attention_dropout)

    def transpose_for_scores(self, x, channels_per_head):
        """Reshape input for self-attention"""
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
    ):
        """ Forward pass of the Perceiver Self-Attention module"""
        hidden_states = self.layernorm1(hidden_states)
        inputs = self.layernorm2(inputs)

        is_cross_attention = inputs is not None
        queries = self.query(hidden_states)

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
            attention_mask = inputs_mask
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)

        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        _, _, _, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hidden = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hidden,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return {
            'context': context_layer,
            'attention_prob': attention_probs,
        }
