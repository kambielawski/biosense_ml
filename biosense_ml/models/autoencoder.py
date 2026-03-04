"""Convolutional autoencoder for 512x512 RGB images."""

import torch
import torch.nn as nn
from omegaconf import DictConfig


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder: 512x512x3 -> 16x16x512 bottleneck -> 512x512x3.

    Encoder: 5 stride-2 conv blocks, channels 3->32->64->128->256->512.
    Decoder: 5 stride-2 transposed conv blocks, channels mirror encoder;
             final activation is Sigmoid so output is in [0, 1].
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the autoencoder.

        Args:
            cfg: Model config node (cfg.model). Uses cfg.latent_channels
                 to set the bottleneck channel depth (default 512).
        """
        super().__init__()
        latent_channels: int = getattr(cfg, "latent_channels", 512)

        # ------------------------------------------------------------------
        # Encoder: each block halves spatial dims
        # Input:  (B,   3, 512, 512)
        # After 1:(B,  32, 256, 256)
        # After 2:(B,  64, 128, 128)
        # After 3:(B, 128,  64,  64)
        # After 4:(B, 256,  32,  32)
        # After 5:(B, 512,  16,  16)
        # ------------------------------------------------------------------
        enc_channels = [3, 32, 64, 128, 256, latent_channels]
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

        # ------------------------------------------------------------------
        # Decoder: each block doubles spatial dims
        # Input:  (B, 512,  16,  16)
        # After 1:(B, 256,  32,  32)
        # After 2:(B, 128,  64,  64)
        # After 3:(B,  64, 128, 128)
        # After 4:(B,  32, 256, 256)
        # After 5:(B,   3, 512, 512)
        # ------------------------------------------------------------------
        dec_channels = [latent_channels, 256, 128, 64, 32, 3]
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
                    nn.Sigmoid() if is_last else nn.ReLU(inplace=True),
                )
            )
        self.decoder = nn.Sequential(*decoder_blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder then decoder.

        Args:
            x: Input images of shape (B, 3, 512, 512), values in [0, 1].

        Returns:
            Tuple of:
                reconstruction: Reconstructed images (B, 3, 512, 512) in [0, 1].
                bottleneck: Encoder output (B, latent_channels, 16, 16).
        """
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return reconstruction, bottleneck
