
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Robust reconstruction loss.
    """

    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        difference = prediction - target

        loss = torch.sqrt(
            difference ** 2 + self.epsilon ** 2
        )

        return loss.mean()


class GradientLoss3D(nn.Module):
    """
    3D gradient consistency loss.

    Compares spatial intensity gradients along
    depth, height and width.
    """

    def forward(self, prediction, target):

        pred_d = prediction[:, :, 1:, :, :] - prediction[:, :, :-1, :, :]
        pred_h = prediction[:, :, :, 1:, :] - prediction[:, :, :, :-1, :]
        pred_w = prediction[:, :, :, :, 1:] - prediction[:, :, :, :, :-1]

        target_d = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        target_h = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        target_w = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

        loss_d = F.l1_loss(pred_d, target_d)
        loss_h = F.l1_loss(pred_h, target_h)
        loss_w = F.l1_loss(pred_w, target_w)

        return (loss_d + loss_h + loss_w) / 3.0


class ReconstructionLoss3D(nn.Module):
    """
    Combined reconstruction objective for Stage A.
    """

    def __init__(
        self,
        l1_weight=1.0,
        charbonnier_weight=0.0,
        gradient_weight=0.1
    ):
        super().__init__()

        self.l1_weight = l1_weight
        self.charbonnier_weight = charbonnier_weight
        self.gradient_weight = gradient_weight

        self.l1 = nn.L1Loss()
        self.charbonnier = CharbonnierLoss()
        self.gradient = GradientLoss3D()

    def forward(self, prediction, target):

        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: prediction {prediction.shape}, "
                f"target {target.shape}"
            )

        l1_loss = self.l1(prediction, target)

        charbonnier_loss = self.charbonnier(
            prediction, target
        )

        gradient_loss = self.gradient(
            prediction, target
        )

        total_loss = (
            self.l1_weight * l1_loss
            + self.charbonnier_weight * charbonnier_loss
            + self.gradient_weight * gradient_loss
        )

        return {
            "total": total_loss,
            "l1": l1_loss,
            "charbonnier": charbonnier_loss,
            "gradient": gradient_loss
        }
