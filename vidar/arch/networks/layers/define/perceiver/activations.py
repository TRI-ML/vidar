# Copyright 2023 Toyota Research Institute.  All rights reserved.

import math

import torch
from packaging import version
from torch import nn

from knk_vision.vidar.vidar.arch.networks.layers.activations import GaussianActivation


def sine(x):
    """Sine activation function"""
    return torch.sin(x)


def gelu_python(x):
    """Generalized ELU activation function"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """Modified Generalized ELU activation function"""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


if version.parse(torch.__version__) < version.parse("1.4"):
    gelu = gelu_python
else:
    gelu = nn.functional.gelu


def gelu_fast(x):
    """Fast Generalized ELU activation function"""
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def quick_gelu(x):
    """Quick ELU activation function"""
    return x * torch.sigmoid(1.702 * x)


def silu_python(x):
    """SILU activation function"""
    return x * torch.sigmoid(x)


if version.parse(torch.__version__) < version.parse("1.7"):
    silu = silu_python
else:
    silu = nn.functional.silu


def mish_python(x):
    """MISH activation function"""
    return x * torch.tanh(nn.functional.softplus(x))


if version.parse(torch.__version__) < version.parse("1.9"):
    mish = mish_python
else:
    mish = nn.functional.mish


def linear_act(x):
    """Linear activation function"""
    return x


ACT2FN = {
    "relu": nn.functional.relu,
    "silu": silu,
    "swish": silu,
    "gelu": gelu,
    "tanh": torch.tanh,
    "gelu_python": gelu_python,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "quick_gelu": quick_gelu,
    "mish": mish,
    "linear": linear_act,
    "sigmoid": torch.sigmoid,
    "sine": sine,
    "gaussian": GaussianActivation(a=1, trainable=True)
}


def get_activation(activation_string):
    """Get activation function from string"""
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
