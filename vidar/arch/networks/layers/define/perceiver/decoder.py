# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch.nn as nn

from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.layer import PerceiverLayer


class PerceiverBasicDecoder(nn.Module):
    """
    Basic decoder network for Perceiver, with functionalities shared across specific decoders
    
    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """    
    def __init__(self, cfg, output_num_channels=None):
        super().__init__()

        output_num_channels = output_num_channels if output_num_channels is not None else \
            cfg.has('output_num_channels', output_num_channels)

        self.decoding_cross_attention = PerceiverLayer(
            is_cross_attention=True,
            use_query_residual=cfg.use_query_residual,
            qk_channels=cfg.has('qk_channels', None),
            v_channels=cfg.has('v_channels', None),
            num_heads=cfg.num_heads,
            q_dim=cfg.num_channels,
            kv_dim=cfg.d_latents,
            widening_factor=cfg.widening_factor,
            hidden_activ=cfg.hidden_activ,
            cross_attention_shape=cfg.has('cross_attention_shape', 'kv'),
            attention_dropout=cfg.attention_dropout,
            use_flash_attention=cfg.has('use_flash_attention', False),
        )

        self.mlp_type = cfg.has('mlp_type', 'single')
        if self.mlp_type == 'single':
            self.final_layer = nn.Linear(cfg.num_channels, output_num_channels)
            # self.final_layer.weight.data.normal_(mean=0.0, std=0.02)
            # self.final_layer.bias.data.fill_(0.0)
        elif self.mlp_type == 'double':
            self.final_layer = nn.Sequential(
                nn.Linear(cfg.num_channels, 2 * cfg.num_channels),
                nn.Linear(2 * cfg.num_channels, output_num_channels),
            )
        elif self.mlp_type == 'implicit':
            from knk_vision.vidar.vidar.arch.networks.layers.define.decoders.utils.implicit_net import ImplicitNet
            self.final_layer = ImplicitNet(d_out=output_num_channels, num_blocks=3)
        elif self.mlp_type == 'residual':
            from knk_vision.vidar.vidar.arch.networks.layers.define.decoders.utils.res_mlp import ResMLP
            self.final_layer = ResMLP(input_dim=cfg.num_channels, output_dim=output_num_channels,
                                      depth=44, channels=256)
        elif self.mlp_type == 'residual_upsample':
            from knk_vision.vidar.vidar.arch.networks.layers.define.decoders.utils.res_mlp_upsample import ResMLPUpsample
            self.final_layer = ResMLPUpsample(input_dim=cfg.num_channels, output_dim=output_num_channels,
                                      depth=44, channels=256)
        else:
            from knk_vision.vidar.vidar.arch.networks.layers.define.decoders.utils.mlp import ImplicitNet
            self.final_layer = ImplicitNet(d_in=cfg.num_channels, d_out=output_num_channels, dims=self.mlp_type)

    def forward(self, query, z, shape=None, query_mask=None, extra=None):
        """Forward pass of the decoder network, returns attention values and predictions"""

        cross_output = self.decoding_cross_attention(
            query,
            attention_mask=query_mask,
            head_mask=None,
            inputs=z,
            inputs_mask=None,
        )

        if self.mlp_type in ['single']:
            predictions = self.final_layer(cross_output['mlp'])
        elif self.mlp_type in ['implicit']:
            predictions = self.final_layer(cross_output['mlp'], shape=shape, extra=extra)
        else:
            predictions = self.final_layer(cross_output['mlp'], shape=shape)

        return {
            'predictions': predictions,
            'cross_output': cross_output,
        }
