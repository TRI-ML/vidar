# Copyright 2023 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
import torch.nn as nn


def build_position_encoding(
        position_encoding_type,
        out_channels=None,
        project_pos_dim=-1,
        fourier_position_encoding_kwargs=None,
):
    """ Build position encoding module and projection layer"""
    if position_encoding_type == "fourier":
        output_pos_enc = PerceiverFourierPositionEncoding(**fourier_position_encoding_kwargs)
    else:
        raise ValueError(f"Unknown position encoding type: {position_encoding_type}.")
    positions_projection = nn.Linear(out_channels, project_pos_dim) if project_pos_dim > 0 else nn.Identity()
    return output_pos_enc, positions_projection


def generate_fourier_features(pos, num_bands=None, max_resolution=None, freq_bands=None,
                              concat_pos=True, sine_only=False, freq_sampling='linear'):
    """Generate fourier features from a given set of positions and frequencies"""
    b, n = pos.shape[:2]
    device = pos.device

    if freq_sampling == 'linear':
        min_freq = 1.0
        freq_bands = torch.stack(
            [torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=device)
             for res in max_resolution], dim=0
        )
    elif freq_sampling == 'log':
        freq_bands = torch.stack(
            [2. ** torch.linspace(0., max_res, steps=num_bands)
             for max_res in max_resolution], dim=0
        ).to(pos.device)
    else:
        raise ValueError('Invalid freq_sampling')

    per_pos_features = torch.stack([pos[i, :, :][:, :, None] * freq_bands[None, :, :] for i in range(b)], 0)
    per_pos_features = per_pos_features.reshape(b, n, -1)

    if sine_only:
        per_pos_features = torch.sin(np.pi * per_pos_features)
    else:
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )

    if concat_pos:
        per_pos_features = torch.cat([pos, per_pos_features], dim=-1)

    return per_pos_features


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    """Build linear positions for a given set of dimensions"""
    def _linspace(n_xels_per_dim):
        return torch.linspace(start=output_range[0], end=output_range[1], steps=n_xels_per_dim, dtype=torch.float32)
    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = torch.meshgrid(*dim_ranges)
    return torch.stack(array_index_grid, dim=-1)


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """Check or build spatial positions"""
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = torch.broadcast_to(pos[None], (batch_size,) + pos.shape)
        pos = torch.reshape(pos, [batch_size, np.prod(index_dims), -1])
    return pos


class PerceiverFourierPositionEncoding(nn.Module):
    """
    Class used to generate Fourier positional encodings for a Perceiver network
    
    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self,cfg, n):
        super().__init__()
        self.num_bands = cfg.num_bands
        self.max_resolution = [cfg.max_resolution] * n
        self.concat_pos = cfg.has('concat_pos', True)
        self.sine_only = cfg.has('sine_only', False)
        self.freq_sampling = cfg.has('freq_sampling', 'linear')

        self.increase_resolution = cfg.has('increase_resolution', None)
        self.current = 1

    @property
    def dims(self):
        """Number of dimensions of the positional encoding"""
        return len(self.max_resolution)

    @property
    def channels(self):
        """Number of channels of the positional encoding"""
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += self.dims

        return encoding_size

    def forward(self, index_dims, batch_size, device, pos=None):
        """Forward pass to generate Fourier positional encodings"""
        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
            freq_sampling=self.freq_sampling,
        ).to(device)
        return fourier_pos_enc
