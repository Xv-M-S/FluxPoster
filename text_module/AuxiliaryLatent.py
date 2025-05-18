import torch.nn as nn
from abc import abstractmethod
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
import torch
import yaml
from ldm.models.autoencoder import AutoencoderKL

"""
ldm 链接: https://github.com/CompVis/latent-diffusion/tree/main/ldm
"""

def get_autoencoder():
    # Load the YAML configuration file
    with open('/home/sxm/Poster/TextGeneration/Method/AnyText2/models_yaml/anytext2_sd15.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Extract the first_stage_config
    first_stage_config = config['model']['params']['first_stage_config']

    # Instantiate the AutoencoderKL
    autoencoder = AutoencoderKL(
        ddconfig=first_stage_config['params']['ddconfig'],
        lossconfig=first_stage_config['params']['lossconfig'],
        embed_dim=first_stage_config['params']['embed_dim'],
        monitor=first_stage_config['params'].get('monitor', None)
    )

    print("Autoencoder initialized successfully.")

    return autoencoder

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, attnx_scale=1.0):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, attnx_scale)
            else:
                x = layer(x)
        return x


class AuxiliaryLatentMoudle(nn.Module):
    def __init__(
            self,
            glyph_scale= 1,
            glyph_channels = 1,
            dims=2,
            position_channels = 1,
            model_channels = 320
    ):
        super().__init__()
        self.autoencoder = get_autoencoder()
        self.glyph_scale = glyph_scale

        if self.glyph_scale == 2:
            self.glyph_block = TimestepEmbedSequential(
                conv_nd(dims, glyph_channels, 8, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 8, 8, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 8, 16, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
            )
        elif self.glyph_scale == 1:
            self.glyph_block = TimestepEmbedSequential(
                conv_nd(dims, glyph_channels, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2),
                nn.SiLU(),
            )

        self.position_block = TimestepEmbedSequential(
            conv_nd(dims, position_channels, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 8, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 8, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
        )
        self.fuse_block_za = zero_module(conv_nd(dims, 256+64+4, model_channels, 3, padding=1))

    def encode(self, text_info):
        # 位置和字形合并
        glyphs = text_info['glyphs']
        positions = text_info['positions']
        masked_x = text_info['masked_x']

        # Encode the input - vae
        x = masked_x.unsqueeze(0) 
        posterior = self.autoencoder.encode(x)
        # Sample from the posterior distribution
        latent_sample = posterior.sample().bfloat16()

        enc_glyph = self.glyph_block(glyphs, None, None)
        enc_glyph = enc_glyph.unsqueeze(0)
       
        enc_pos = self.position_block(positions, None, None)
        enc_pos = enc_pos.unsqueeze(0)

        guided_hint = self.fuse_block_za(torch.cat([enc_glyph, enc_pos, latent_sample], dim=1))
        
        return guided_hint

if __name__ == '__main__':
    pass


