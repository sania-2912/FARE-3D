
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecomposition3D(nn.Module):
    """
    Approximate 3D low/high-frequency decomposition.

    Low frequency:
        Local average of the volume.

    High frequency:
        Original volume - low-frequency component.

    Uses replicate padding to preserve spatial dimensions.
    """

    def __init__(self, kernel_size=5):
        super().__init__()

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number.")

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def low_pass(self, volume):

        p = self.padding

        padded = F.pad(
            volume,
            (p, p, p, p, p, p),
            mode="replicate"
        )

        return F.avg_pool3d(
            padded,
            kernel_size=self.kernel_size,
            stride=1
        )

    def high_pass(self, volume):

        low_frequency = self.low_pass(volume)

        return volume - low_frequency

    def forward(self, volume):

        low_frequency = self.low_pass(volume)
        high_frequency = volume - low_frequency

        return low_frequency, high_frequency


class FrequencyLoss3D(nn.Module):
    """
    Frequency-aware reconstruction objective for FARE-3D.

    Combines:
        1. Low-frequency reconstruction loss.
        2. High-frequency reconstruction loss.
        3. Low-frequency leakage penalty for Stage B.
    """

    def __init__(
        self,
        low_weight=1.0,
        high_weight=1.0,
        leakage_weight=0.1,
        kernel_size=5
    ):
        super().__init__()

        self.low_weight = low_weight
        self.high_weight = high_weight
        self.leakage_weight = leakage_weight

        self.decomposition = FrequencyDecomposition3D(
            kernel_size=kernel_size
        )

    def forward(
        self,
        prediction,
        target,
        refiner_residual=None
    ):

        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target must have identical shapes."
            )

        # Decompose prediction and ground truth.
        pred_low, pred_high = self.decomposition(prediction)
        target_low, target_high = self.decomposition(target)

        # Preserve broad intensity structure.
        low_loss = F.l1_loss(
            pred_low,
            target_low
        )

        # Preserve fine anatomical details.
        high_loss = F.l1_loss(
            pred_high,
            target_high
        )

        # Penalize low-frequency changes from Stage B.
        if refiner_residual is not None:

            residual_low, _ = self.decomposition(
                refiner_residual
            )

            leakage_loss = residual_low.abs().mean()

        else:
            leakage_loss = prediction.new_zeros(())

        total_loss = (
            self.low_weight * low_loss
            + self.high_weight * high_loss
            + self.leakage_weight * leakage_loss
        )

        return {
            "total": total_loss,
            "low_frequency": low_loss,
            "high_frequency": high_loss,
            "frequency_leakage": leakage_loss
        }
