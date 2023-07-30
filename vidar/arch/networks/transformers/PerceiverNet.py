# Copyright 2023 Toyota Research Institute.  All rights reserved.

from abc import ABC

from externals.transformers.adapters.inputs.image import ImageInputAdapter
from externals.transformers.adapters.outputs.image import ImageOutputAdapter
from externals.transformers.networks.perceiver_io import PerceiverIO
from externals.transformers.networks.perceiver_io_decoder import PerceiverIODecoder
from vidar.arch.networks.BaseNet import BaseNet


class PerceiverNet(BaseNet, ABC):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.input_dim = 256
        self.latent_dim = 512 # 1024
        self.num_latents = 256 # 512

        self.image_shape = (3, *cfg.image_shape)

        self.image_adapter = ImageInputAdapter(
            num_frequency_bands=20,
            max_frequencies=[1, 1, 1],
            input_dim=self.input_dim,
        )

        self.encoder = PerceiverIO(
            input_dim=self.input_dim,
            depth=2,
            num_latents=self.num_latents,
            latent_dim=self.latent_dim,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
        )

        if self.decode_rgb:
            self.decoder_rgb = PerceiverIODecoder(
                output_adapter=ImageOutputAdapter(self.input_dim),
                latent_dim=self.latent_dim,
                cross_heads=1,
                cross_dim_head=64,
            )
        else:
            self.decoder_rgb = None

    def encode(self, rgb=None, cam=None):
        encodings = self.image_adapter(rgb=rgb, cam=cam)
        return self.encoder(encodings.reshape(1, -1, self.input_dim))

    def decode_rgb(self, cam, latent):
        encodings = self.image_adapter(cam=cam)
        latent = latent.repeat(len(encodings), 1, 1)
        rgb = self.decoder_rgb(latent, encodings)
        rgb = rgb.permute(0, 2, 1).reshape(len(rgb), *self.image_shape)

        return {
            'rgb': rgb,
        }
