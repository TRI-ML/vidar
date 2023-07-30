# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch.nn as nn

from vidar.arch.networks.layers.define.perceiver.attention import PerceiverAttention
from vidar.arch.networks.layers.define.perceiver.mlp import PerceiverMLP


class PerceiverLayer(nn.Module):
    """Perceiver Layer with attention and MLP"""
    def __init__(
        self,
        is_cross_attention,
        use_query_residual,
        qk_channels,
        v_channels,
        num_heads,
        q_dim,
        kv_dim,
        widening_factor,
        hidden_activ,
        attention_dropout,
        use_flash_attention,
        cross_attention_shape=None,
    ):
        super().__init__()
        self.attention = PerceiverAttention(
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
            cross_attention_shape=cross_attention_shape,
            attention_dropout=attention_dropout,
            use_flash_attention=use_flash_attention,
        )
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(
            input_size=q_dim,
            hidden_activ=hidden_activ,
            widening_factor=widening_factor
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
    ):
        """ Forward pass of the Perceiver Layer"""

        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
        )

        attention_output = attention_outputs['mlp']

        mlp_output = self.layernorm(attention_output)
        mlp_output = self.mlp(mlp_output)
        mlp_output = mlp_output + attention_output

        return {
            'attention': attention_output,
            'mlp': mlp_output,
        }
