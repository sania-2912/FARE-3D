
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomyEncoder3D(nn.Module):
    """
    3D Vision Transformer for global anatomical context.

    Input:
        volume: [B, 1, D, H, W]

    Output:
        tokens:      [B, N, embed_dim]
        feature_map: [B, embed_dim, D', H', W']

    The token grid is reduced when necessary to control
    the computational cost of global self-attention.
    """

    def __init__(
        self,
        in_channels=1,
        embed_dim=256,
        depth=6,
        num_heads=8,
        patch_size=(4, 8, 8),
        max_token_grid=(8, 8, 8),
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        self.patch_size = tuple(patch_size)
        self.max_token_grid = tuple(max_token_grid)
        self.embed_dim = embed_dim

        # ----------------------------------
        # 3D patch embedding
        # ----------------------------------

        self.patch_embedding = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Learnable 3D positional embedding.
        # It will be interpolated to the current token grid.
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                embed_dim,
                *self.max_token_grid,
            )
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

        # ----------------------------------
        # Transformer encoder
        # ----------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
        )

        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, volume):
        """
        Extract global anatomical representations.
        """

        if volume.ndim != 5:
            raise ValueError(
                "Expected [B, C, D, H, W], "
                f"received {tuple(volume.shape)}"
            )

        # Convert volume into 3D patch embeddings.
        x = self.patch_embedding(volume)

        # Current patch grid.
        _, _, depth, height, width = x.shape

        # ----------------------------------
        # Limit token count
        # ----------------------------------

        max_d, max_h, max_w = self.max_token_grid

        target_grid = (
            min(depth, max_d),
            min(height, max_h),
            min(width, max_w),
        )

        if (depth, height, width) != target_grid:
            x = F.adaptive_avg_pool3d(
                x,
                output_size=target_grid,
            )

        # ----------------------------------
        # Positional embedding interpolation
        # ----------------------------------

        position = F.interpolate(
            self.position_embedding,
            size=target_grid,
            mode="trilinear",
            align_corners=False,
        )

        x = x + position

        # ----------------------------------
        # Convert feature map to tokens
        # ----------------------------------

        batch_size, channels, d, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2)

        # Global anatomical attention.
        tokens = self.transformer(tokens)

        tokens = self.output_norm(tokens)

        # ----------------------------------
        # Restore spatial feature map
        # ----------------------------------

        feature_map = (
            tokens.transpose(1, 2)
            .reshape(batch_size, channels, d, h, w)
        )

        return {
            "tokens": tokens,
            "feature_map": feature_map,
        }
