
import torch
import torch.nn as nn

from src.models.blocks import (
    ConvBlock3D,
    ResidualBlock3D,
    Downsample3D,
)


class SeverityEncoder3D(nn.Module):
    """
    Learns a compact representation of CBCT
    degradation and artifact characteristics.

    Input:
        volume: [B, 1, D, H, W]

    Output:
        severity_embedding: [B, embed_dim]
        severity_features:  [B, C, D', H', W']

    Note:
        The embedding is learned through downstream
        training objectives. It is not automatically
        an interpretable or calibrated artifact score.
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        embed_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        # ----------------------------------
        # Initial feature extraction
        # ----------------------------------

        self.stem = ConvBlock3D(
            in_channels=in_channels,
            out_channels=base_channels,
        )

        # ----------------------------------
        # Multi-scale degradation features
        # ----------------------------------

        self.stage1 = nn.Sequential(
            ResidualBlock3D(
                base_channels,
                base_channels,
                dropout=dropout,
            ),
            ResidualBlock3D(
                base_channels,
                base_channels,
                dropout=dropout,
            ),
        )

        self.down1 = Downsample3D(
            base_channels,
            base_channels * 2,
        )

        self.stage2 = nn.Sequential(
            ResidualBlock3D(
                base_channels * 2,
                base_channels * 2,
                dropout=dropout,
            ),
            ResidualBlock3D(
                base_channels * 2,
                base_channels * 2,
                dropout=dropout,
            ),
        )

        self.down2 = Downsample3D(
            base_channels * 2,
            base_channels * 4,
        )

        self.stage3 = nn.Sequential(
            ResidualBlock3D(
                base_channels * 4,
                base_channels * 4,
                dropout=dropout,
            ),
            ResidualBlock3D(
                base_channels * 4,
                base_channels * 4,
                dropout=dropout,
            ),
        )

        # ----------------------------------
        # Global severity representation
        # ----------------------------------

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, volume):

        if volume.ndim != 5:
            raise ValueError(
                "Expected [B, C, D, H, W], "
                f"received {tuple(volume.shape)}"
            )

        x = self.stem(volume)

        x = self.stage1(x)

        x = self.down1(x)
        x = self.stage2(x)

        x = self.down2(x)
        x = self.stage3(x)

        severity_features = x

        pooled = self.global_pool(x)

        severity_embedding = self.embedding_head(pooled)

        return {
            "severity_embedding": severity_embedding,
            "severity_features": severity_features,
        }
