import torch
import torch.nn as nn

from src.models.anchor import AnchorUNet3D
from src.models.smooth_field import SmoothFieldHead3D
from src.models.anatomy_encoder import AnatomyEncoder3D
from src.models.severity_encoder import SeverityEncoder3D
from src.models.flow_refiner import (
    FlowRefiner3D,
    HighFrequencyProjection3D,
)


class DetailHead3D(nn.Module):
    """
    Predicts high-frequency anatomical corrections
    from Stage A decoder features.

    The high-pass projection is applied in the
    forward path to prevent low-frequency leakage.
    """

    def __init__(
        self,
        in_channels,
        out_channels=1,
        kernel_size=5,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                min(8, in_channels // 2),
                in_channels // 2,
            ),
            nn.SiLU(),

            nn.Conv3d(
                in_channels // 2,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.high_pass = HighFrequencyProjection3D(
            kernel_size=kernel_size
        )

        # Begin with zero detail correction.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):

        raw_detail = self.network(features)

        detail = self.high_pass(raw_detail)

        return {
            "detail": detail,
            "raw_detail": raw_detail,
        }


class FARE3D(nn.Module):
    """
    Frequency-Anchored Residual Evolution (FARE-3D).

    Stage A:
        3D Residual U-Net
        + DCT smooth-field head
        + High-frequency detail head

    Conditioning:
        3D ViT anatomy encoder
        + CBCT severity encoder

    Stage B:
        Conditional 3D flow-matching refiner.

    Input:
        cbct: [B, 1, D, H, W]

    Optional Stage B inputs:
        noisy_residual: [B, 1, D, H, W]
        time: [B]

    Output:
        anchor: [B, 1, D, H, W]
        velocity: [B, 1, D, H, W] when Stage B is enabled
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        smooth_basis_order=(4, 4, 4),
        anatomy_embed_dim=256,
        anatomy_depth=6,
        anatomy_heads=8,
        anatomy_patch_size=(4, 8, 8),
        severity_embed_dim=128,
        refiner_hidden_dim=256,
        refiner_depth=6,
        refiner_heads=8,
        refiner_patch_size=(2, 4, 4),
        use_refiner=True,
        dropout=0.0,
    ):
        super().__init__()

        self.use_refiner = use_refiner

        # ----------------------------------
        # Stage A: Anchor backbone
        # ----------------------------------

        self.anchor_network = AnchorUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

        feature_channels = base_channels * channel_multipliers[0]

        # ----------------------------------
        # Stage A: Smooth-field head
        # ----------------------------------

        self.smooth_field_head = SmoothFieldHead3D(
            in_channels=feature_channels,
            basis_order=smooth_basis_order,
        )

        # ----------------------------------
        # Stage A: Detail head
        # ----------------------------------

        self.detail_head = DetailHead3D(
            in_channels=feature_channels,
            out_channels=out_channels,
        )

        # ----------------------------------
        # Anatomy conditioning
        # ----------------------------------

        self.anatomy_encoder = AnatomyEncoder3D(
            in_channels=in_channels,
            embed_dim=anatomy_embed_dim,
            depth=anatomy_depth,
            num_heads=anatomy_heads,
            patch_size=anatomy_patch_size,
        )

        # ----------------------------------
        # Artifact severity conditioning
        # ----------------------------------

        self.severity_encoder = SeverityEncoder3D(
            in_channels=in_channels,
            embed_dim=severity_embed_dim,
        )

        # ----------------------------------
        # Stage B: Flow refiner
        # ----------------------------------

        if self.use_refiner:

            self.flow_refiner = FlowRefiner3D(
                in_channels=out_channels,
                out_channels=out_channels,
                anatomy_dim=anatomy_embed_dim,
                severity_dim=severity_embed_dim,
                hidden_dim=refiner_hidden_dim,
                depth=refiner_depth,
                num_heads=refiner_heads,
                patch_size=refiner_patch_size,
                dropout=dropout,
            )

    def forward(
        self,
        cbct,
        noisy_residual=None,
        time=None,
    ):

        if cbct.ndim != 5:
            raise ValueError(
                "Expected CBCT shape [B, C, D, H, W]."
            )

        # ==================================
        # STAGE A: ANCHOR RECONSTRUCTION
        # ==================================

        anchor_output = self.anchor_network(cbct)

        base_prediction = anchor_output["prediction"]

        decoder_features = anchor_output["features"]

        # Low-frequency correction
        smooth_output = self.smooth_field_head(
            decoder_features
        )

        smooth_field = smooth_output["smooth_field"]

        # High-frequency detail correction
        detail_output = self.detail_head(
            decoder_features
        )

        detail = detail_output["detail"]

        # Final Stage A anchor
        anchor = (
            base_prediction
            + smooth_field
            + detail
        )

        # ==================================
        # CONDITIONING ENCODERS
        # ==================================

        anatomy_output = self.anatomy_encoder(cbct)

        severity_output = self.severity_encoder(cbct)

        anatomy_tokens = anatomy_output["tokens"]

        severity_embedding = severity_output[
            "severity_embedding"
        ]

        # ==================================
        # OUTPUT DICTIONARY
        # ==================================

        outputs = {
            "base_prediction": base_prediction,
            "smooth_field": smooth_field,
            "detail": detail,
            "anchor": anchor,

            "anchor_features": decoder_features,

            "smooth_coefficients": smooth_output[
                "coefficients"
            ],

            "anatomy_tokens": anatomy_tokens,
            "anatomy_feature_map": anatomy_output[
                "feature_map"
            ],

            "severity_embedding": severity_embedding,
            "severity_features": severity_output[
                "severity_features"
            ],
        }

        # ==================================
        # STAGE B: FLOW REFINEMENT
        # ==================================

        if self.use_refiner:

            if (noisy_residual is None) != (time is None):
                raise ValueError(
                    "Provide both noisy_residual and time, "
                    "or neither."
                )

            if noisy_residual is not None:

                refinement_output = self.flow_refiner(
                    noisy_residual=noisy_residual,
                    time=time,
                    anatomy_tokens=anatomy_tokens,
                    severity_embedding=severity_embedding,
                )

                outputs["velocity"] = refinement_output[
                    "velocity"
                ]

                outputs["raw_velocity"] = refinement_output[
                    "raw_velocity"
                ]

        return outputs

    @staticmethod
    def integrate_residual(anchor, residual):

        """
        Combine an anchor with a high-frequency residual.

        The residual is projected again before addition,
        preserving the low-frequency anchor component.
        """

        high_pass = HighFrequencyProjection3D(
            kernel_size=5
        ).to(
            device=residual.device,
            dtype=residual.dtype,
        )

        projected_residual = high_pass(residual)

        return anchor + projected_residual
