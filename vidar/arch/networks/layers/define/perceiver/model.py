# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.encoder_mod import PerceiverEncoder
from knk_vision.vidar.vidar.utils.data import get_from_list
from knk_vision.vidar.vidar.utils.types import is_int


class PerceiverModel(nn.Module, ABC):
    """
    Class to create and store the Perceiver model
    
    Parameters
    ----------
    config : Config
        Configuration with parameters
    """
    def __init__(self, config, is_variational):
        super().__init__()

        if is_variational:
            config = deepcopy(config)
            config.latent.dim *= 2

        config.layer_norm_eps = 1e-12
        config.initializer_range = 0.02
        config.is_encoder_decoder = False
        config.pruned_heads = False
        config.encoder.d_latents = config.latent.dim

        self.dropout = config.latent.has('dropout', None)
        self.detach_recurrent = config.has('detach_recurrent', False)

        self.config = config

        self.latent_mode = config.latent.has('mode', 'single')
        if self.latent_mode == 'single':
            from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.embeddings import PerceiverEmbeddings
            self.embeddings = PerceiverEmbeddings(config.latent)
        elif self.latent_mode == 'multi':
            from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.multi_embeddings import PerceiverMultiEmbeddings
            self.embeddings = PerceiverMultiEmbeddings(config.latent)
        elif self.latent_mode == 'token':
            from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.token_embeddings import PerceiverTokenEmbeddings
            self.embeddings = PerceiverTokenEmbeddings(config.latent)
        elif self.latent_mode == 'grid':
            from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.grid_embeddings import PerceiverGridEmbeddings
            self.embeddings = PerceiverGridEmbeddings(config.latent)
        else:
            raise ValueError('Invalid latent mode')

        self.use_deq = config.has('use_deq', False)
        self.encoder = PerceiverEncoder
        if config.encoder.has('multiple'):
            self.encoder = torch.nn.ModuleList(
                [self.encoder(config.encoder) for _ in range(config.encoder.multiple)])
        else:
            self.encoder = self.encoder(config.encoder)
        self.apply(self.init_weights)


        self.mode = config.mode

        if self.mode == 'gru':
            self.rnn = torch.nn.GRU(
                input_size=config.latent.dim,
                hidden_size=config.latent.dim,
                num_layers=4,
                batch_first=True,
            )

        self.with_clip = config.has('with_clip', False)
        if self.with_clip:
            from knk_vision.vidar.vidar.arch.networks.language.CLIPNet import CLIPNet
            from knk_vision.vidar.vidar.utils.config import Config
            clip_cfg = Config(
                linear_out=config.latent.dim
            )
            self.clip = CLIPNet(clip_cfg)

    def invert_attention_mask(self, encoder_attention_mask):
        """ Invert an attention mask (e.g., switches 0. and 1.)."""
        dtype = encoder_attention_mask.dtype
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=dtype)
        if dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif dtype in [torch.bfloat16, torch.float32]:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(f"{dtype} not recognized.")
        return encoder_extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        """ Prepare head mask if needed."""
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif hasattr(module, "latents"):
            module.latents.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.ParameterDict):
            for modality in module.keys():
                module[modality].data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def default(self, data, embeddings, scene, attention_mask, head_mask):
        """Forward pass to retrieve the Perceiver IO latent space (default)"""
        # Stack all embeddings

        if embeddings is None:
            return self.encoder(
                self.embeddings(batch_size=1),
                inputs=None, inputs_mask=None,
                head_mask=None, attention_mask=None,
            )

        embeddings = torch.cat([emb for emb in embeddings.values()], 1)
        if embeddings.dim() == 4:
            b, n, k, d = embeddings.shape
            embeddings = embeddings.view(b, n * k, d)

        # Get dimensions and device information

        batch_size, seq_length, _ = embeddings.size()
        device = embeddings.device

        # Masks

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        extended_attention_mask = self.invert_attention_mask(attention_mask)
        head_mask = self.get_head_mask(
            head_mask, self.config.encoder.num_blocks * self.config.encoder.num_self_attends_per_block)

        # Latent representation

        latent = self.embeddings(batch_size=batch_size, scene=scene)

        # Apply dropout if requested

        if self.training and self.dropout is not None:
            tot = self.dropout if is_int(self.dropout) else int(self.dropout * latent.shape[1])
            idx = [torch.randperm(latent.shape[1])[:tot] for _ in range(latent.shape[0])]
            latent = torch.stack([latent[i, idx[i]] for i in range(len(idx))], 0)

        if self.with_clip:
            clip = {key: self.clip(val['gt']['rgb'])['rgb'].unsqueeze(1) for key, val in data.items()}
            latent = torch.cat([latent] + [val for val in clip.values()], 1)

        # Encode inputs

        encoded = self.encoder(
            latent,
            inputs=embeddings,
            inputs_mask=extended_attention_mask,
            head_mask=head_mask,
            attention_mask=None,
        )

        return encoded

    def recurrent(self, data, scene, attention_mask, head_mask):
        """Forward pass to retrieve the Perceiver IO latent space (recurrent)"""

        keys = list(data.keys())
        tsteps = sorted(list(set([k[0] for k in keys])))
        cams = sorted(list(set([k[1] for k in keys])))

        data0 = data[keys[0]]
        batch_size, seq_length, _ = data0.size()
        device = data0.device

        seq_length *= len(cams)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        extended_attention_mask = self.invert_attention_mask(attention_mask)
        head_mask = self.get_head_mask(
            head_mask, self.config.encoder.num_blocks * self.config.encoder.num_self_attends_per_block)

        latent = self.embeddings(batch_size=batch_size, scene=scene)

        encodeds = {}
        for i, t in enumerate(tsteps):

            keys_t = [key for key in keys if key[0] == t]
            data_t = torch.cat([val for key, val in data.items() if key in keys_t], 1)

            encoded = get_from_list(self.encoder, i)(
                latent,
                inputs=data_t,
                inputs_mask=extended_attention_mask,
                head_mask=head_mask,
                attention_mask=None,
            )

            latent = encoded['last_hidden_state']
            encodeds[t] = encoded

            if self.detach_recurrent:
                latent = latent.detach()

        encoded = {}
        for key1 in encodeds[tsteps[0]].keys():
            encoded[key1] = {key2: val[key1] for key2, val in encodeds.items()}

        return encoded

    def gru(self, data, scene, attention_mask, head_mask):
        """Forward pass to retrieve the Perceiver IO latent space (gated recurrent)"""

        keys = list(data.keys())
        tsteps = sorted(list(set([k[0] for k in keys])))
        cams = sorted(list(set([k[1] for k in keys])))

        data0 = data[keys[0]]
        batch_size, seq_length, _ = data0.size()
        device = data0.device

        seq_length *= len(cams)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        extended_attention_mask = self.invert_attention_mask(attention_mask)
        head_mask = self.get_head_mask(
            head_mask, self.config.encoder.num_blocks * self.config.encoder.num_self_attends_per_block)

        latent = self.embeddings(batch_size=batch_size, scene=scene)

        encodeds = {}
        for i, t in enumerate(tsteps):

            keys_t = [key for key in keys if key[0] == t]
            data_t = torch.cat([val for key, val in data.items() if key in keys_t], 1)

            encoded = get_from_list(self.encoder, i)(
                latent,
                inputs=data_t,
                inputs_mask=extended_attention_mask,
                head_mask=head_mask,
                attention_mask=None,
            )

            latent = encoded['last_hidden_state']
            encodeds[t] = encoded

            b, n, d = latent.shape
            latent = self.rnn(latent.view(b * n, 1, d))[0].view(b, n, d)

        encoded = {}
        for key1 in encodeds[tsteps[0]].keys():
            encoded[key1] = {key2: val[key1] for key2, val in encodeds.items()}

        return encoded

    def separated(self, data, scene, attention_mask, head_mask):
        """Forward pass to retrieve the Perceiver IO latent space (separated)"""

        keys = list(data.keys())
        tsteps = sorted(list(set([k[0] for k in keys])))
        cams = sorted(list(set([k[1] for k in keys])))

        data0 = data[keys[0]]
        batch_size, seq_length, _ = data0.size()
        device = data0.device

        seq_length *= len(cams)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        extended_attention_mask = self.invert_attention_mask(attention_mask)
        head_mask = self.get_head_mask(
            head_mask, self.config.encoder.num_blocks * self.config.encoder.num_self_attends_per_block)

        latent = self.embeddings(batch_size=batch_size, scene=scene)

        encodeds = {}
        for i, t in enumerate(tsteps):

            keys_t = [key for key in keys if key[0] == t]
            data_t = torch.cat([val for key, val in data.items() if key in keys_t], 1)

            encoded = get_from_list(self.encoder, i)(
                latent,
                inputs=data_t,
                inputs_mask=extended_attention_mask,
                head_mask=head_mask,
                attention_mask=None,
            )
            encodeds[t] = encoded

        encoded = {}
        for key1 in encodeds[tsteps[0]].keys():
            encoded[key1] = {key2: val[key1] for key2, val in encodeds.items()}

        return encoded

    def forward(
        self,
        data, embeddings, scene,
        attention_mask=None,
        head_mask=None,
    ):
        """Forward pass that switches between modes"""
        if self.mode == 'default':
            return self.default(data, embeddings, scene, attention_mask, head_mask)
        elif self.mode == 'recurrent':
            return self.recurrent(data, embeddings, scene, attention_mask, head_mask)
        elif self.mode == 'gru':
            return self.gru(data, embeddings, scene, attention_mask, head_mask)
        elif self.mode == 'separated':
            return self.separated(data, embeddings, scene, attention_mask, head_mask)
        else:
            raise ValueError('Invalid Perceiver Model')
