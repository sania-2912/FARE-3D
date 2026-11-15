
import numpy as np
from skimage.metrics import structural_similarity


def _prepare_arrays(prediction, target, mask=None):
    """
    Convert inputs to NumPy arrays and validate dimensions.
    """

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: {prediction.shape} vs {target.shape}"
        )

    if prediction.ndim != 3:
        raise ValueError(
            "Expected 3D volumes with shape [D, H, W]."
        )

    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or Inf.")

    if not np.isfinite(target).all():
        raise ValueError("Target contains NaN or Inf.")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != prediction.shape:
            raise ValueError("Mask shape must match the volumes.")

        if not mask.any():
            raise ValueError("Evaluation mask is empty.")

        prediction = prediction[mask]
        target = target[mask]

    return prediction, target


def calculate_mae(prediction, target, mask=None):
    """Mean Absolute Error."""

    prediction, target = _prepare_arrays(
        prediction, target, mask
    )

    return float(np.mean(np.abs(prediction - target)))


def calculate_rmse(prediction, target, mask=None):
    """Root Mean Squared Error."""

    prediction, target = _prepare_arrays(
        prediction, target, mask
    )

    return float(
        np.sqrt(np.mean((prediction - target) ** 2))
    )


def calculate_psnr(
    prediction,
    target,
    data_range,
    mask=None
):
    """
    Peak Signal-to-Noise Ratio.

    data_range must be specified explicitly.
    """

    if data_range <= 0:
        raise ValueError("data_range must be positive.")

    prediction, target = _prepare_arrays(
        prediction, target, mask
    )

    mse = np.mean((prediction - target) ** 2)

    if mse == 0:
        return float("inf")

    return float(
        10.0 * np.log10((data_range ** 2) / mse)
    )


def calculate_ssim(
    prediction,
    target,
    data_range,
    mask=None,
    window_size=7
):
    """
    Structural Similarity Index for 3D volumes.

    SSIM is calculated on the complete volume.
    Masked SSIM is not implemented because masking
    individual voxels changes the local-window calculation.
    """

    if mask is not None:
        raise NotImplementedError(
            "SSIM masking requires a dedicated masked-window implementation."
        )

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ.")

    if prediction.ndim != 3:
        raise ValueError("Expected 3D volumes.")

    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or Inf.")

    if not np.isfinite(target).all():
        raise ValueError("Target contains NaN or Inf.")

    if data_range <= 0:
        raise ValueError("data_range must be positive.")

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    if min(prediction.shape) < window_size:
        raise ValueError(
            "Volume dimensions must be at least window_size."
        )

    score = structural_similarity(
        prediction,
        target,
        data_range=data_range,
        win_size=window_size,
        gaussian_weights=True,
        sigma=1.5
    )

    return float(score)


def calculate_all_metrics(
    prediction,
    target,
    data_range,
    mask=None
):
    """
    Calculate all supported metrics.

    Use original HU volumes for clinically meaningful
    HU-based MAE and RMSE reporting.
    """

    return {
        "MAE": calculate_mae(
            prediction, target, mask
        ),
        "RMSE": calculate_rmse(
            prediction, target, mask
        ),
        "PSNR": calculate_psnr(
            prediction, target, data_range, mask
        ),
        "SSIM": calculate_ssim(
            prediction, target, data_range
        )
    }
