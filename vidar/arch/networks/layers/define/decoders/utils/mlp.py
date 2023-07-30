import torch
from torch import nn
import numpy as np


class ImplicitNet(nn.Module):
    """
    Represents a MLP;
    Original code from IGR
    """

    def __init__(
        self, d_in, dims,
        skip_in=(), d_out=4,
        dim_excludes_skip=False,
        combine_layer=1000,
        combine_type="average",
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out]
        if dim_excludes_skip:
            for i in range(1, len(dims) - 1):
                if i in skip_in:
                    dims[i] += d_in

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.dims = dims
        self.combine_layer = combine_layer
        self.combine_type = combine_type

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            nn.init.constant_(lin.bias, 0.0)
            nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")

            setattr(self, "lin" + str(layer), lin)

        self.activation = nn.ReLU()

    def forward(self, x, shape=None):
        x_init = x
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer < self.combine_layer and layer in self.skip_in:
                x = torch.cat([x, x_init], -1) / np.sqrt(2)

            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)

        return x
