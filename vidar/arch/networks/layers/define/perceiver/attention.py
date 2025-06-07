# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch.nn as nn

from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.self_attention import PerceiverSelfAttention
from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.self_output import PerceiverSelfOutput


class PerceiverAttention(nn.Module):
    """Perceiver attention module, used for self-attention and cross-attention."""
    def __init__(
        self,
        is_cross_attention,
        qk_channels,
        v_channels,
        num_heads,
        q_dim,
        kv_dim,
        use_query_residual,
        cross_attention_shape,
        attention_dropout,
        use_flash_attention,
    ):
        super().__init__()
        self.use_query_residual = use_query_residual
        self.is_cross_attention = is_cross_attention
        self.with_flash_attention = use_flash_attention

        if is_cross_attention and qk_channels is None:
            if cross_attention_shape == "q":
                qk_channels = q_dim
            elif cross_attention_shape == "kv":
                qk_channels = kv_dim
            else:
                raise ValueError(
                    f"Unknown value {cross_attention_shape} for "
                    "cross_attention_shape_for_attention."
                )
        else:
            if qk_channels is None:
                qk_channels = q_dim
            if v_channels is None:
                v_channels = qk_channels

        if not self.is_cross_attention and self.with_flash_attention:
            from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.flash import FlashMHA
            self.self = FlashMHA(
                embed_dim=q_dim, num_heads=num_heads,
                attention_dropout=attention_dropout,
            )
        else:
            self.self = PerceiverSelfAttention(
                is_cross_attention=is_cross_attention,
                qk_channels=qk_channels, v_channels=v_channels,
                num_heads=num_heads, q_dim=q_dim, kv_dim=kv_dim,
                attention_dropout=attention_dropout,
            )
            self.output = PerceiverSelfOutput(
                in_channels=self.self.v_channels,
                out_channels=q_dim if is_cross_attention else v_channels
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
    ):
        """Forward pass for the attention module, returns attention values and projection values"""
        if not self.is_cross_attention and self.with_flash_attention:
            attention_output = self.self(hidden_states)[0]
            if self.use_query_residual:
                attention_output = attention_output + hidden_states
            return {
                'mlp': attention_output,
            }

        self_output = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
        )

        attention_output = self.output(self_output['context'])
        if self.use_query_residual:
            attention_output = attention_output + hidden_states

        return {
            'attention': self_output,
            'mlp': attention_output,
        }
