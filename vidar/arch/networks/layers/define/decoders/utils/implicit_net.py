import torch
from torch import nn


def combine_interleaved(t, inner_dims=(1,), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


class ResnetBlockFC(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()

        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        net = self.fc_0(self.activation(x))
        dx = self.fc_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


class ImplicitNet(nn.Module):
    def __init__(self, d_out, num_blocks):
        super().__init__()

        self.d_in = 198
        self.d_out = d_out

        self.n_blocks = num_blocks
        self.d_hidden = 256
        self.d_latent = 0 # 64
        self.combine_layer = 0 # 3

        if self.d_in > 0:
            self.lin_in = nn.Linear(self.d_in, self.d_hidden)
            nn.init.constant_(self.lin_in.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        self.lin_out = nn.Linear(self.d_hidden, self.d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(self.d_hidden) for i in range(self.n_blocks)]
        )

        if self.d_latent != 0:
            n_lin_z = min(self.combine_layer, self.n_blocks)
            self.lin_z = nn.ModuleList(
                [nn.Linear(self.d_latent, self.d_hidden) for i in range(n_lin_z)]
            )
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

        self.activation = nn.ReLU()

    def forward(self, x, extra=None):
        x = self.lin_in(x)
        for block_id in range(self.n_blocks):
            if self.d_latent > 0 and block_id < self.combine_layer:
                tz = self.lin_z[block_id](extra)
                x = x + tz
            x = self.blocks[block_id](x)
        return self.lin_out(self.activation(x))


# class ImplicitNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.d_in = 153
#         self.d_out = 4
#
#         self.n_blocks = 4
#         self.d_hidden = 512
#         self.d_latent = 64
#         self.combine_layer = 3
#
#         if self.d_in > 0:
#             self.lin_in = nn.Linear(self.d_in, self.d_hidden)
#             nn.init.constant_(self.lin_in.bias, 0.0)
#             nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")
#
#         self.lin_out = nn.Linear(self.d_hidden, self.d_out)
#         nn.init.constant_(self.lin_out.bias, 0.0)
#         nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")
#
#         self.blocks = nn.ModuleList(
#             [ResnetBlockFC(self.d_hidden) for i in range(self.n_blocks)]
#         )
#
#         if self.d_latent != 0:
#             n_lin_z = min(self.combine_layer, self.n_blocks)
#             self.lin_z = nn.ModuleList(
#                 [nn.Linear(self.d_latent, self.d_hidden) for i in range(n_lin_z)]
#             )
#             for i in range(n_lin_z):
#                 nn.init.constant_(self.lin_z[i].bias, 0.0)
#                 nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")
#
#         self.activation = nn.ReLU()
#
#     def forward(self, x, extra=None):
#         x = self.lin_in(x)
#         for block_id in range(self.n_blocks):
#             if self.d_latent > 0 and block_id < self.combine_layer:
#                 tz = self.lin_z[block_id](extra)
#                 x = x + tz
#             x = self.blocks[block_id](x)
#         return self.lin_out(self.activation(x))