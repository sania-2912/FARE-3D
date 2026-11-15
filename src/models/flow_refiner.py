
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Converts continuous flow time into an embedding."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, time):
        if time.ndim == 0:
            time = time.unsqueeze(0)

        time = time.reshape(-1).float()

        half_dim = self.embedding_dim // 2

        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half_dim,
                device=time.device,
                dtype=time.dtype,
            )
            / max(half_dim - 1, 1)
        )

        angles = time[:, None] * frequencies[None, :]

        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=-1,
        )

        if embedding.shape[-1] < self.embedding_dim:
            embedding = F.pad(
                embedding,
                (0, self.embedding_dim - embedding.shape[-1]),
            )

        return embedding


class FlowDiTBlock3D(nn.Module):
    """
    Transformer block with:
        1. Spatial self-attention
        2. Conditional cross-attention
        3. Feed-forward network
    """

    def __init__(
        self,
        hidden_dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        self.self_norm = nn.LayerNorm(hidden_dim)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.condition_norm = nn.LayerNorm(hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.mlp_norm = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(
                hidden_dim,
                int(hidden_dim * mlp_ratio),
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                int(hidden_dim * mlp_ratio),
                hidden_dim,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, tokens, condition_tokens):

        # Spatial self-attention
        normalized = self.self_norm(tokens)

        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )

        tokens = tokens + attended

        # Cross-attention to anatomy and severity
        query = self.cross_norm(tokens)
        condition = self.condition_norm(condition_tokens)

        attended, _ = self.cross_attention(
            query,
            condition,
            condition,
            need_weights=False,
        )

        tokens = tokens + attended

        # Feed-forward network
        tokens = tokens + self.mlp(
            self.mlp_norm(tokens)
        )

        return tokens


class HighFrequencyProjection3D(nn.Module):
    """
    Removes local low-frequency content.

    P_H(x) = x - LowPass(x)

    This is a differentiable spatial high-pass projection.
    The cutoff is determined by the smoothing kernel size.
    """

    def __init__(self, kernel_size=5):
        super().__init__()

        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be an odd integer >= 3."
            )

        self.kernel_size = kernel_size

    def forward(self, x):

        padding = self.kernel_size // 2

        # Replicate padding avoids artificial zero-valued
        # boundaries when computing the local average.
        padded = F.pad(
            x,
            (
                padding, padding,
                padding, padding,
                padding, padding,
            ),
            mode="replicate",
        )

        low_frequency = F.avg_pool3d(
            padded,
            kernel_size=self.kernel_size,
            stride=1,
        )

        return x - low_frequency


class FlowRefiner3D(nn.Module):
    """
    Conditional 3D flow-matching refiner.

    Inputs:
        noisy_residual:
            [B, 1, D, H, W]

        time:
            [B] continuous values in [0, 1]

        anatomy_tokens:
            [B, N, anatomy_dim]

        severity_embedding:
            [B, severity_dim]

    Output:
        velocity:
            [B, 1, D, H, W]

        raw_velocity:
            Unprojected velocity for diagnostics.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        anatomy_dim=256,
        severity_dim=128,
        hidden_dim=256,
        depth=6,
        num_heads=8,
        patch_size=(2, 4, 4),
        time_embedding_dim=128,
        mlp_ratio=4.0,
        dropout=0.0,
        frequency_kernel_size=5,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        self.patch_size = tuple(patch_size)

        # ----------------------------------
        # Residual patch embedding
        # ----------------------------------

        self.patch_embedding = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # ----------------------------------
        # Time embedding
        # ----------------------------------

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ----------------------------------
        # Conditioning projections
        # ----------------------------------

        self.anatomy_projection = nn.Linear(
            anatomy_dim,
            hidden_dim,
        )

        self.severity_projection = nn.Linear(
            severity_dim,
            hidden_dim,
        )

        # ----------------------------------
        # Conditional DiT blocks
        # ----------------------------------

        self.blocks = nn.ModuleList([
            FlowDiTBlock3D(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

        # ----------------------------------
        # Velocity prediction head
        # ----------------------------------

        self.output_head = nn.Conv3d(
            hidden_dim,
            out_channels,
            kernel_size=1,
        )

        # Hard high-frequency projection
        self.frequency_projection = (
            HighFrequencyProjection3D(
                kernel_size=frequency_kernel_size
            )
        )

    def forward(
        self,
        noisy_residual,
        time,
        anatomy_tokens,
        severity_embedding,
    ):

        if noisy_residual.ndim != 5:
            raise ValueError(
                "Expected noisy residual [B, C, D, H, W]."
            )

        batch_size, _, original_d, original_h, original_w = (
            noisy_residual.shape
        )

        if anatomy_tokens.ndim != 3:
            raise ValueError(
                "anatomy_tokens must have shape [B, N, C]."
            )

        if severity_embedding.ndim != 2:
            raise ValueError(
                "severity_embedding must have shape [B, C]."
            )

        if (
            anatomy_tokens.shape[0] != batch_size
            or severity_embedding.shape[0] != batch_size
        ):
            raise ValueError(
                "Input and conditioning batch sizes must match."
            )

        # ----------------------------------
        # Patch embedding
        # ----------------------------------

        x = self.patch_embedding(noisy_residual)

        _, channels, d, h, w = x.shape

        # Convert spatial map to tokens
        tokens = x.flatten(2).transpose(1, 2)

        # ----------------------------------
        # Time conditioning
        # ----------------------------------

        time_embedding = self.time_embedding(time)

        if time_embedding.shape[0] == 1 and batch_size > 1:
            time_embedding = time_embedding.expand(
                batch_size, -1
            )

        if time_embedding.shape[0] != batch_size:
            raise ValueError(
                "Time batch size must match input batch size."
            )

        tokens = tokens + time_embedding.unsqueeze(1)

        # ----------------------------------
        # Anatomy and severity conditioning
        # ----------------------------------

        anatomy = self.anatomy_projection(
            anatomy_tokens
        )

        severity = self.severity_projection(
            severity_embedding
        ).unsqueeze(1)

        condition_tokens = torch.cat(
            [anatomy, severity],
            dim=1,
        )

        # ----------------------------------
        # Conditional transformer
        # ----------------------------------

        for block in self.blocks:
            tokens = block(
                tokens,
                condition_tokens,
            )

        tokens = self.output_norm(tokens)

        # Restore spatial feature map
        x = (
            tokens.transpose(1, 2)
            .reshape(batch_size, channels, d, h, w)
        )

        # Upsample to original volume dimensions
        x = F.interpolate(
            x,
            size=(original_d, original_h, original_w),
            mode="trilinear",
            align_corners=False,
        )

        # Raw predicted velocity
        raw_velocity = self.output_head(x)

        # Enforce high-frequency-only output
        velocity = self.frequency_projection(
            raw_velocity
        )

        return {
            "velocity": velocity,
            "raw_velocity": raw_velocity,
        }
