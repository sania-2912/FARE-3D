
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighFrequencyProjection3D(nn.Module):
    """
    Projects a 3D volume into the approximate
    high-frequency subspace.

    Must remain consistent with the projection
    used inside FlowRefiner3D.
    """

    def __init__(self, kernel_size=5):
        super().__init__()

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd number."
            )

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, volume):

        p = self.padding

        padded = F.pad(
            volume,
            (p, p, p, p, p, p),
            mode="replicate"
        )

        low_frequency = F.avg_pool3d(
            padded,
            kernel_size=self.kernel_size,
            stride=1
        )

        return volume - low_frequency


class FlowMatchingLoss3D(nn.Module):
    """
    Conditional flow-matching objective for FARE-3D.

    Both noise and target residual are projected
    into the same high-frequency subspace.
    """

    def __init__(self, kernel_size=5):
        super().__init__()

        self.projection = HighFrequencyProjection3D(
            kernel_size=kernel_size
        )

    def forward(
        self,
        model,
        anchor,
        target,
        anatomy_tokens,
        severity_embedding
    ):

        if anchor.shape != target.shape:
            raise ValueError(
                "Anchor and target must have identical shapes."
            )

        batch_size = anchor.shape[0]
        device = anchor.device

        # 1. Ground-truth high-frequency residual.
        with torch.no_grad():

            residual = target - anchor

            x1 = self.projection(residual)

            # 2. Random noise in the same subspace.
            noise = torch.randn_like(x1)

            x0 = self.projection(noise)

        # 3. Sample continuous time.
        t = torch.rand(
            batch_size,
            device=device
        )

        t_volume = t.view(
            batch_size, 1, 1, 1, 1
        )

        # 4. Interpolate noise and target.
        xt = (
            (1.0 - t_volume) * x0
            + t_volume * x1
        )

        # 5. Constant velocity target.
        target_velocity = x1 - x0

        # 6. Predict velocity using conditional DiT.
        outputs = model(
            noisy_residual=xt,
            time=t,
            anatomy_tokens=anatomy_tokens,
            severity_embedding=severity_embedding
        )

        predicted_velocity = outputs["velocity"]

        # 7. Flow-matching objective.
        loss = F.mse_loss(
            predicted_velocity,
            target_velocity
        )

        return {
            "total": loss,
            "flow_matching": loss,
            "time": t.detach()
        }
