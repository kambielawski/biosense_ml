"""Convolutional autoencoder for 512x512 RGB images."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

# Fixed intermediate channel progression regardless of depth
_INTERMEDIATE_CHANNELS = [32, 64, 128, 256]


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder: 512x512x3 -> spatial bottleneck -> 512x512x3.

    Encoder: N stride-2 conv blocks controlled by cfg.num_encoder_blocks.
    Decoder: mirrors encoder in reverse.
    No final activation — output is unbounded to match ImageNet-normalized
    input range (approximately [-2.1, 2.6]).

    Bottleneck spatial size = 512 / (2 ** num_encoder_blocks), unless
    bottleneck_spatial is set, in which case an adaptive pool resizes
    the encoder output to that target spatial size.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the autoencoder.

        Args:
            cfg: Model config node (cfg.model). Uses:
                 - cfg.latent_channels: bottleneck channel depth (default 512)
                 - cfg.num_encoder_blocks: number of stride-2 blocks (default 5)
                 - cfg.bottleneck_spatial: optional target spatial size (e.g. 12
                   for 12x12). When set, adds AdaptiveAvgPool2d after encoder
                   and bilinear upsample before decoder.
        """
        super().__init__()
        latent_channels: int = getattr(cfg, "latent_channels", 512)
        num_encoder_blocks: int = getattr(cfg, "num_encoder_blocks", 5)
        bottleneck_spatial: int | None = getattr(cfg, "bottleneck_spatial", None)

        self._natural_spatial = 512 // (2 ** num_encoder_blocks)
        self._bottleneck_spatial = bottleneck_spatial

        # Build channel list: [3, <intermediates>, latent_channels]
        intermediates = _INTERMEDIATE_CHANNELS[: num_encoder_blocks - 1]
        enc_channels = [3] + intermediates + [latent_channels]

        # ------------------------------------------------------------------
        # Encoder: each block halves spatial dims
        # ------------------------------------------------------------------
        encoder_blocks: list[nn.Module] = []
        for i in range(len(enc_channels) - 1):
            encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        enc_channels[i],
                        enc_channels[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(enc_channels[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )
        self.encoder = nn.Sequential(*encoder_blocks)

        # Optional adaptive pooling to non-power-of-2 spatial size
        if bottleneck_spatial is not None:
            self.adaptive_pool = nn.AdaptiveAvgPool2d(bottleneck_spatial)
        else:
            self.adaptive_pool = None

        # ------------------------------------------------------------------
        # Decoder: each block doubles spatial dims (mirrors encoder)
        # ------------------------------------------------------------------
        dec_channels = [latent_channels] + list(reversed(intermediates)) + [3]
        decoder_blocks: list[nn.Module] = []
        for i in range(len(dec_channels) - 1):
            is_last = i == len(dec_channels) - 2
            decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        dec_channels[i],
                        dec_channels[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(dec_channels[i + 1]),
                    nn.Identity() if is_last else nn.ReLU(inplace=True),
                )
            )
        self.decoder = nn.Sequential(*decoder_blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder then decoder.

        Args:
            x: Input images of shape (B, 3, 512, 512), ImageNet-normalized.

        Returns:
            Tuple of:
                reconstruction: Reconstructed images (B, 3, 512, 512), unbounded,
                    same scale as input (ImageNet-normalized).
                bottleneck: Encoder output (B, latent_channels, H_b, W_b).
        """
        bottleneck = self.encoder(x)
        if self.adaptive_pool is not None:
            bottleneck = self.adaptive_pool(bottleneck)

        # Decoder expects the natural spatial size from stride-2 blocks
        if self._bottleneck_spatial is not None:
            dec_input = F.interpolate(
                bottleneck,
                size=self._natural_spatial,
                mode="bilinear",
                align_corners=False,
            )
        else:
            dec_input = bottleneck

        reconstruction = self.decoder(dec_input)
        return reconstruction, bottleneck
