import torch
from torch import nn
import numpy as np
from einops import rearrange


class MLP(nn.Module):
    """MLP class with different activation functions"""
    def __init__(self, n_in,
                 n_layers=4, n_hidden_units=256,
                 act='relu', act_trainable=False,
                 **kwargs):
        super().__init__()

        layers = []
        for i in range(n_layers):

            if i == 0:
                l = nn.Linear(n_in, n_hidden_units)
            elif 0 < i < n_layers - 1:
                l = nn.Linear(n_hidden_units, n_hidden_units)

            if act == 'relu':
                act_ = nn.ReLU(True)
            elif act == 'gaussian':
                act_ = GaussianActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'quadratic':
                act_ = QuadraticActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'multi-quadratic':
                act_ = MultiQuadraticActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'laplacian':
                act_ = LaplacianActivation(a=kwargs['a'], trainable=act_trainable)
            elif act == 'super-gaussian':
                act_ = SuperGaussianActivation(a=kwargs['a'], b=kwargs['b'],
                                               trainable=act_trainable)
            elif act == 'expsin':
                act_ = ExpSinActivation(a=kwargs['a'], trainable=act_trainable)

            if i < n_layers - 1:
                layers += [l, act_]
            else:
                layers += [nn.Linear(n_hidden_units, 3), nn.Sigmoid()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, 2) # pixel uv (normalized)
        """
        return self.net(x)  # (B, 3) rgb


class PE(nn.Module):
    """
    perform positional encoding
    """

    def __init__(self, P):
        """
        P: (2, F) encoding matrix
        """
        super().__init__()
        self.register_buffer("P", P)

    @property
    def out_dim(self):
        return self.P.shape[1] * 2

    def forward(self, x):
        """
        x: (B, 2)
        """
        x_ = 2 * np.pi * x @ self.P  # (B, F)
        return torch.cat([torch.sin(x_), torch.cos(x_)], 1)  # (B, 2*F)


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    def __init__(self, in_features=2, out_features=3,
                 hidden_features=256, hidden_layers=4, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


# different activation functions
class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x ** 2 / (2 * self.a ** 2))


class QuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x):
        return 1 / (1 + (self.a * x) ** 2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x):
        return 1 / (1 + (self.a * x) ** 2) ** 0.5


class LaplacianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x) / self.a)


class SuperGaussianActivation(nn.Module):
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b * torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x ** 2 / (2 * self.a ** 2)) ** self.b


class ExpSinActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a * x))
