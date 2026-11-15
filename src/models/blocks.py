
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# NORMALIZATION
# ==========================================

def get_norm(channels, norm_type="group"):
    """
    Create a normalization layer.

    GroupNorm is preferred for 3D medical imaging
    because batch sizes are often small.
    """

    if norm_type == "group":
        # Select the largest valid group count <= 8.
        groups = min(8, channels)

        while channels % groups != 0:
            groups -= 1

        return nn.GroupNorm(
            num_groups=groups,
            num_channels=channels
        )

    elif norm_type == "instance":
        return nn.InstanceNorm3d(
            channels,
            affine=True
        )

    elif norm_type == "batch":
        return nn.BatchNorm3d(channels)

    else:
        raise ValueError(
            f"Unsupported normalization: {norm_type}"
        )


# ==========================================
# ACTIVATION
# ==========================================

def get_activation(activation="silu"):
    """
    Create activation functions.
    """

    if activation == "silu":
        return nn.SiLU(inplace=True)

    elif activation == "relu":
        return nn.ReLU(inplace=True)

    elif activation == "gelu":
        return nn.GELU()

    elif activation == "leaky_relu":
        return nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True
        )

    else:
        raise ValueError(
            f"Unsupported activation: {activation}"
        )


# ==========================================
# 3D CONVOLUTION BLOCK
# ==========================================

class ConvBlock3D(nn.Module):
    """
    Basic 3D convolution block.

    Architecture:
        Conv3D -> Normalization -> Activation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        norm_type="group",
        activation="silu",
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.norm = get_norm(
            out_channels,
            norm_type
        )

        self.activation = get_activation(
            activation
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


# ==========================================
# RESIDUAL BLOCK
# ==========================================

class ResidualBlock3D(nn.Module):
    """
    3D residual block.

    Architecture:

        Input
          |
        Conv3D
          |
        Norm + SiLU
          |
        Conv3D
          |
        Norm
          |
        Add Input (Skip Connection)
          |
        SiLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_type="group",
        dropout=0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm1 = get_norm(
            out_channels,
            norm_type
        )

        self.activation1 = nn.SiLU(inplace=True)

        self.dropout = nn.Dropout3d(
            p=dropout
        ) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = get_norm(
            out_channels,
            norm_type
        )

        # Match input and output channels if necessary.
        if in_channels != out_channels:
            self.skip = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )
        else:
            self.skip = nn.Identity()

        self.activation2 = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + residual

        return self.activation2(out)


# ==========================================
# DOWNSAMPLING BLOCK
# ==========================================

class Downsample3D(nn.Module):
    """
    Reduce spatial dimensions by a factor of 2.

    Example:
        [B, C, 64, 128, 128]
        ->
        [B, C_out, 32, 64, 64]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_type="group",
    ):
        super().__init__()

        self.block = ConvBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_type=norm_type,
        )

    def forward(self, x):
        return self.block(x)


# ==========================================
# UPSAMPLING BLOCK
# ==========================================

class Upsample3D(nn.Module):
    """
    Upsample decoder features and combine them
    with encoder skip connections.

    Interpolation is used to support odd spatial
    dimensions without requiring exact doubling.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_type="group",
    ):
        super().__init__()

        self.conv = ConvBlock3D(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            norm_type=norm_type,
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        return self.conv(x)
