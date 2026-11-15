
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothFieldHead3D(nn.Module):
    """
    Predicts a smooth 3D intensity correction field
    using a learned combination of low-order DCT bases.

    Input:
        features: [B, C, D, H, W]

    Output:
        smooth_field: [B, 1, D, H, W]
        coefficients: [B, Kd, Kh, Kw]

    The DCT basis is fixed. Only its coefficients are learned.
    """

    def __init__(
        self,
        in_channels,
        basis_order=(4, 4, 4),
        hidden_dim=128,
    ):
        super().__init__()

        self.basis_order = tuple(basis_order)

        if len(self.basis_order) != 3:
            raise ValueError(
                "basis_order must contain (Kd, Kh, Kw)."
            )

        if any(order < 1 for order in self.basis_order):
            raise ValueError(
                "Every DCT basis order must be positive."
            )

        num_coefficients = math.prod(self.basis_order)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.coefficient_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_coefficients),
        )

        # Start with zero correction.
        # The model learns the correction progressively.
        nn.init.zeros_(
            self.coefficient_predictor[-1].weight
        )
        nn.init.zeros_(
            self.coefficient_predictor[-1].bias
        )

    @staticmethod
    def create_dct_basis(
        length,
        num_basis,
        device,
        dtype,
    ):
        """
        Construct an orthonormal DCT-II basis.

        Returns:
            [length, num_basis]
        """

        if num_basis > length:
            raise ValueError(
                f"Number of basis functions ({num_basis}) "
                f"cannot exceed spatial length ({length})."
            )

        positions = torch.arange(
            length,
            device=device,
            dtype=dtype,
        )

        frequencies = torch.arange(
            num_basis,
            device=device,
            dtype=dtype,
        )

        basis = torch.cos(
            math.pi
            * (positions[:, None] + 0.5)
            * frequencies[None, :]
            / length
        )

        normalization = torch.full(
            (num_basis,),
            math.sqrt(2.0 / length),
            device=device,
            dtype=dtype,
        )

        normalization[0] = math.sqrt(1.0 / length)

        basis = basis * normalization.unsqueeze(0)

        return basis

    def forward(self, features):
        """
        Generate the smooth correction field.
        """

        if features.ndim != 5:
            raise ValueError(
                "Expected [B, C, D, H, W], "
                f"received {tuple(features.shape)}"
            )

        batch_size, _, depth, height, width = features.shape

        kd, kh, kw = self.basis_order

        # Predict DCT coefficients from global features.
        pooled = self.pool(features)

        coefficients = self.coefficient_predictor(pooled)

        coefficients = coefficients.reshape(
            batch_size,
            kd,
            kh,
            kw,
        )

        # Construct orthonormal DCT bases.
        # Use the feature dtype and device.
        basis_d = self.create_dct_basis(
            depth,
            kd,
            features.device,
            features.dtype,
        )

        basis_h = self.create_dct_basis(
            height,
            kh,
            features.device,
            features.dtype,
        )

        basis_w = self.create_dct_basis(
            width,
            kw,
            features.device,
            features.dtype,
        )

        # Reconstruct the smooth 3D field:
        #
        # field[b,d,h,w] =
        #   sum(coeff[b,i,j,k] *
        #       basis_d[d,i] *
        #       basis_h[h,j] *
        #       basis_w[w,k])
        #
        smooth_field = torch.einsum(
            "bijk,di,hj,wk->bdhw",
            coefficients,
            basis_d,
            basis_h,
            basis_w,
        )

        smooth_field = smooth_field.unsqueeze(1)

        return {
            "smooth_field": smooth_field,
            "coefficients": coefficients,
        }
