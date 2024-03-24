# Copyright 2023 Toyota Research Institute.  All rights reserved.

import torch.nn as nn

from knk_vision.vidar.vidar.arch.networks.layers.define.perceiver.activations import ACT2FN


class PerceiverMLP(nn.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self, hidden_activ, input_size, widening_factor):
        super().__init__()
        self.dense1 = nn.Linear(input_size, widening_factor * input_size)
        if isinstance(hidden_activ, str):
            self.intermediate_act_fn = ACT2FN[hidden_activ]
        else:
            self.intermediate_act_fn = hidden_activ
        self.dense2 = nn.Linear(input_size, input_size)

    def forward(self, hidden_states):
        """Forward pass of the Perceiver MLP"""
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states
