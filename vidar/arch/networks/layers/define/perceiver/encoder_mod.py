# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch.nn as nn

from vidar.arch.networks.layers.define.perceiver.layer import PerceiverLayer
from vidar.utils.types import is_list


class PerceiverEncoder(nn.Module):
    """
    Class in charge of encoding information in the Perceiver network (modified from original)
    
    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.d_latents % cfg.self_attention.num_heads != 0:
            raise ValueError(
                f"num_z_channels ({cfg.d_latents}) must be divisible by"
                f" num_self_attend_heads ({cfg.self_attention.num_heads})."
            )
        if cfg.d_latents % cfg.cross_attention.num_heads != 0:
            raise ValueError(
                f"num_z_channels ({cfg.d_latents}) must be divisible by"
                f" num_cross_attend_heads ({cfg.cross_attention.num_heads})."
            )

        # Cross-attention layer
        if cfg.d_model > 0:
            self.cross_attention = PerceiverLayer(
                is_cross_attention=True,
                use_query_residual=cfg.use_query_residual,
                qk_channels=cfg.has('qk_channels', None),
                v_channels=cfg.has('v_channels', None),
                num_heads=cfg.cross_attention.num_heads,
                q_dim=cfg.d_latents,
                kv_dim=cfg.d_model,
                widening_factor=cfg.cross_attention.widening_factor,
                cross_attention_shape=cfg.cross_attention.has('shape', 'kv'),
                attention_dropout=cfg.attention_dropout,
                hidden_activ=cfg.hidden_activ,
                use_flash_attention=cfg.has('use_flash_attention', False),
            )
        else:
            self.cross_attention = None

        # Self-attention layers
        self_attention_layers = []
        for _ in range(cfg.num_self_attends_per_block):
            layer = PerceiverLayer(
                is_cross_attention=False,
                use_query_residual=cfg.use_query_residual,
                qk_channels=cfg.has('qk_channels', None),
                v_channels=cfg.has('v_channels', None),
                num_heads=cfg.self_attention.num_heads,
                q_dim=cfg.d_latents,
                kv_dim=cfg.d_latents,
                widening_factor=cfg.self_attention.widening_factor,
                attention_dropout=cfg.attention_dropout,
                hidden_activ=cfg.hidden_activ,
                use_flash_attention=cfg.has('use_flash_attention', False),
            )
            self_attention_layers.append(layer)

        self.self_attends = nn.ModuleList(self_attention_layers)

    def forward(
        self,
        hidden_state,
        attention_mask=None,
        head_mask=None,
        inputs=None,
        inputs_mask=None,
    ):
        """ Forward pass for the Perceiver encoder"""
        all_hidden_state = [hidden_state]
        all_self_attention = []
        all_cross_attention = []

        # For each block
        for j in range(self.cfg.num_blocks):

            # If inputs are available, use cross-attention
            if inputs is not None:
                cross_output = self.cross_attention(
                    hidden_state,
                    attention_mask=attention_mask,
                    head_mask=None,
                    inputs=inputs[j] if is_list(inputs) else inputs,
                    inputs_mask=inputs_mask,
                )

                all_cross_attention.append(cross_output['attention'])
                hidden_state = cross_output['mlp']

            all_hidden_state.append(hidden_state)

            # Self-attention always happens, if there are modules
            for i, layer_module in enumerate(self.self_attends):
                self_outputs = layer_module(
                    hidden_state,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i] if head_mask is not None else None,
                )
                hidden_state = self_outputs['mlp']

                all_hidden_state.append(hidden_state)
                all_self_attention.append(self_outputs['attention'])

        return {
            'last_hidden_state': hidden_state,
            'all_hidden_state': all_hidden_state,
            'all_self_attention': all_self_attention,
            'all_cross_attention': all_cross_attention,
        }
