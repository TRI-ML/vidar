import torch.nn as nn


class ResMLP2(nn.Module):

    def __init__(self,
                 width,
                 inact=nn.ReLU(True),
                 outact=None,
                 res_scale=1,
                 n_learnable=2):
        '''inact is the activation func within block. outact is the activation func right before output'''
        super().__init__()
        m = [nn.Linear(width, width)]
        for _ in range(n_learnable - 1):
            if inact is not None: m += [inact]
            m += [nn.Linear(width, width)]
        if inact is not None: m += [inact]
        self.body = nn.Sequential(*m)
        # self.final = nn.Linear(d_in, d_out)
        self.res_scale = res_scale
        self.outact = outact

    def forward(self, x):
        x = self.body(x).mul(self.res_scale) + x
        # x = self.final(x)
        if self.outact is not None:
            x = self.outact(x)
        return x


def get_activation(act):
    if act.lower() == 'relu':
        func = nn.ReLU(inplace=True)
    elif act.lower() == 'lrelu':
        func = nn.LeakyReLU(inplace=True)
    elif act.lower() == 'none':
        func = None
    else:
        raise NotImplementedError
    return func


class ResMLP(nn.Module):
    '''Based on NeRF_v3, move positional embedding out'''

    def __init__(self, input_dim, output_dim, depth, channels):
        super().__init__()
        # self.args = args
        # D, W = args.netdepth, args.netwidth
        D, W = depth, channels

        # get network width
        # if args.layerwise_netwidths:
        #     Ws = [int(x) for x in args.layerwise_netwidths.split(',')] + [3]
        #     print('Layer-wise widths are given. Overwrite args.netwidth')
        # else:
        Ws = [W] * (D - 1) + [3]

        # the main non-linear activation func
        act = get_activation('relu')

        # input_dim = 236
        # output_dim = 4

        # head
        self.input_dim = input_dim
        self.head = nn.Sequential(*[nn.Linear(input_dim, Ws[0]), act])

        # body
        body = []
        for i in range(1, D - 1):
            body += [nn.Linear(Ws[i - 1], Ws[i]), act]

        # >>> new implementation of the body. Will replace the above
        inact = get_activation('relu')
        outact = get_activation('none')

        n_block = (
            D - 2
        ) // 2  # 2 layers in a ResMLP, deprecated since there can be >2 layers in a block, use --trial.n_block

        body = [
            ResMLP2(W,
                   inact=inact,
                   outact=outact,
                   res_scale=1,
                   n_learnable=2)
            for _ in range(n_block)
        ]

        self.body = nn.Sequential(*body)

        # tail
        self.tail = nn.Linear(
            W, output_dim)

    def forward(self, x, shape=None):  # x: embedded position coordinates
        # if x.shape[-1] != self.input_dim:  # [N, C, H, W]
        #     x = x.permute(0, 2, 3, 1)
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return x
