"""
OceanEmbed model definitions.

    surface patch [B, C, H, W]
        -> OceanEncoder (CNN + global average pool + FC)
        -> Ocean Embedding [B, EMBEDDING_DIM]
        -> OceanDecoder (MLP)
        -> temperature profile [B, N_DEPTHS]

C = len(config.INPUT_VARIABLES) is read from config, so adding SSH/current/
wind channels later only requires changing config.INPUT_VARIABLES (and
re-running preprocessing) - no architecture code needs to change, since the
first conv layer's in_channels is derived from config at construction time.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ml import config


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> (optional) downsample via stride-2 conv."""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        stride = 2 if downsample else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OceanEncoder(nn.Module):
    """
    CNN baseline that turns a small surface patch [B, C, H, W] into a fixed
    length Ocean Embedding [B, embedding_dim].

    Architecture: stacked conv blocks with increasing channel width and
    spatial downsampling -> global average pooling (so it works for any
    PATCH_SIZE) -> a fully connected projection to embedding_dim.
    """

    def __init__(
        self,
        in_channels: int = None,
        conv_channels: List[int] = None,
        embedding_dim: int = None,
    ):
        super().__init__()
        in_channels = in_channels or len(config.INPUT_VARIABLES)
        conv_channels = conv_channels or config.CONV_CHANNELS
        embedding_dim = embedding_dim or config.EMBEDDING_DIM

        blocks = []
        prev_c = in_channels
        for i, out_c in enumerate(conv_channels):
            # don't downsample on the very first block if the patch is
            # already small (keeps at least a couple of spatial cells alive
            # for tiny PATCH_SIZE values such as 5 or 7)
            downsample = True
            blocks.append(ConvBlock(prev_c, out_c, downsample=downsample))
            prev_c = out_c
        self.conv_blocks = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> [B, C, 1, 1], any H,W
        self.fc = nn.Sequential(
            nn.Linear(prev_c, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        feats = self.conv_blocks(x)          # [B, C', H', W']
        pooled = self.global_pool(feats)     # [B, C', 1, 1]
        pooled = pooled.flatten(1)           # [B, C']
        embedding = self.fc(pooled)          # [B, embedding_dim]
        return embedding


class OceanDecoder(nn.Module):
    """MLP: Ocean Embedding [B, embedding_dim] -> temperature profile [B, n_depths]."""

    def __init__(
        self,
        embedding_dim: int = None,
        hidden_dims: List[int] = None,
        n_depths: int = None,
    ):
        super().__init__()
        embedding_dim = embedding_dim or config.EMBEDDING_DIM
        hidden_dims = hidden_dims or config.DECODER_HIDDEN_DIMS
        n_depths = n_depths or config.N_DEPTHS

        layers = []
        prev_dim = embedding_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_depths))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(embedding)  # [B, n_depths]


class OceanEmbedModel(nn.Module):
    """Full pipeline: surface patch -> Ocean Embedding -> temperature profile."""

    def __init__(
        self,
        in_channels: int = None,
        conv_channels: List[int] = None,
        embedding_dim: int = None,
        decoder_hidden_dims: List[int] = None,
        n_depths: int = None,
    ):
        super().__init__()
        self.encoder = OceanEncoder(in_channels, conv_channels, embedding_dim)
        self.decoder = OceanDecoder(self.encoder.embedding_dim, decoder_hidden_dims, n_depths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        temperature_profile = self.decoder(embedding)
        return temperature_profile

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the Ocean Embedding, without decoding to temperature."""
        with torch.no_grad():
            return self.encoder(x)


def build_model() -> OceanEmbedModel:
    """Construct a model sized from the current config.py values."""
    return OceanEmbedModel(
        in_channels=len(config.INPUT_VARIABLES),
        conv_channels=config.CONV_CHANNELS,
        embedding_dim=config.EMBEDDING_DIM,
        decoder_hidden_dims=config.DECODER_HIDDEN_DIMS,
        n_depths=config.N_DEPTHS,
    )


if __name__ == "__main__":
    # quick shape smoke test
    model = build_model()
    dummy = torch.randn(4, len(config.INPUT_VARIABLES), config.PATCH_SIZE, config.PATCH_SIZE)
    out = model(dummy)
    emb = model.encode(dummy)
    print(f"Input shape:     {tuple(dummy.shape)}")
    print(f"Embedding shape: {tuple(emb.shape)}")
    print(f"Output shape:    {tuple(out.shape)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
