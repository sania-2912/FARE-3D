import torch
import torch.nn as nn

from src.models.blocks import (
    ConvBlock3D,
    ResidualBlock3D,
    Downsample3D,
    Upsample3D,
)


class AnchorUNet3D(nn.Module):
    """
    3D Residual U-Net backbone for FARE-3D Stage A.

    Responsibilities:
        1. Extract multi-scale anatomical features.
        2. Reconstruct spatially detailed feature maps.
        3. Produce a direct CT prediction for baseline experiments.
        4. Expose decoder features for frequency-aware heads.

    Input:
        CBCT: [B, 1, D, H, W]

    Output:
        features:     [B, base_channels, D, H, W]
        prediction:   [B, 1, D, H, W]
        residual:     [B, 1, D, H, W]
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
    ):
        super().__init__()

        if len(channel_multipliers) < 2:
            raise ValueError(
                "At least two channel levels are required."
            )

        channels = [
            base_channels * multiplier
            for multiplier in channel_multipliers
        ]

        self.channels = channels

        # ----------------------------------
        # Initial feature extraction
        # ----------------------------------

        self.stem = ConvBlock3D(
            in_channels=in_channels,
            out_channels=channels[0],
        )

        # ----------------------------------
        # Encoder
        # ----------------------------------

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for level in range(len(channels) - 1):

            blocks = []

            for block_index in range(num_res_blocks):
                input_channels = (
                    channels[level]
                    if block_index == 0
                    else channels[level]
                )

                blocks.append(
                    ResidualBlock3D(
                        in_channels=input_channels,
                        out_channels=channels[level],
                        dropout=dropout,
                    )
                )

            self.encoder_blocks.append(
                nn.Sequential(*blocks)
            )

            self.downsamples.append(
                Downsample3D(
                    in_channels=channels[level],
                    out_channels=channels[level + 1],
                )
            )

        # ----------------------------------
        # Bottleneck
        # ----------------------------------

        self.bottleneck = nn.Sequential(
            ResidualBlock3D(
                channels[-1],
                channels[-1],
                dropout=dropout,
            ),
            ResidualBlock3D(
                channels[-1],
                channels[-1],
                dropout=dropout,
            ),
        )

        # ----------------------------------
        # Decoder
        # ----------------------------------

        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(len(channels) - 1)):

            self.upsamples.append(
                Upsample3D(
                    in_channels=channels[level + 1],
                    skip_channels=channels[level],
                    out_channels=channels[level],
                )
            )

            blocks = [
                ResidualBlock3D(
                    channels[level],
                    channels[level],
                    dropout=dropout,
                )
                for _ in range(num_res_blocks)
            ]

            self.decoder_blocks.append(
                nn.Sequential(*blocks)
            )

        # ----------------------------------
        # Baseline prediction head
        # ----------------------------------

        self.output_head = nn.Conv3d(
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        """
        Forward pass.

        Returns a dictionary containing:
            features
            prediction
            residual
        """

        if x.ndim != 5:
            raise ValueError(
                "Expected input shape [B, C, D, H, W], "
                f"received {tuple(x.shape)}"
            )

        input_volume = x

        # Initial convolution
        x = self.stem(x)

        # Encoder with skip connections
        skip_features = []

        for encoder, downsample in zip(
            self.encoder_blocks,
            self.downsamples,
        ):
            x = encoder(x)

            skip_features.append(x)

            x = downsample(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for upsample, decoder, skip in zip(
            self.upsamples,
            self.decoder_blocks,
            reversed(skip_features),
        ):
            x = upsample(x, skip)
            x = decoder(x)

        # Final high-resolution feature map
        features = x

        # Direct prediction for baseline training
        prediction = self.output_head(features)

        # Difference between prediction and input.
        # This is diagnostic, not an assumption that
        # CBCT and CT have identical intensity scales.
        residual = prediction - input_volume

        return {
            "features": features,
            "prediction": prediction,
            "residual": residual,
        }
