from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ==========================================
# 1. LOAD MEDICAL VOLUMES
# ==========================================

def load_volume(path):
    """
    Load a medical image using SimpleITK.

    Returns:
        image: SimpleITK image object
        volume: NumPy array in [D, H, W] order
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = sitk.ReadImage(str(path))

    volume = sitk.GetArrayFromImage(image)
    volume = volume.astype(np.float32)

    return image, volume


# ==========================================
# 2. SAVE MEDICAL VOLUMES
# ==========================================

def save_volume(volume, reference_image, output_path):
    """
    Save a NumPy volume while preserving the
    physical metadata of a reference image.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    volume = np.asarray(volume, dtype=np.float32)

    if tuple(volume.shape) != tuple(
        reversed(reference_image.GetSize())
    ):
        raise ValueError(
            "Volume shape does not match reference image geometry."
        )

    output_image = sitk.GetImageFromArray(volume)
    output_image.CopyInformation(reference_image)

    sitk.WriteImage(output_image, str(output_path))

    return output_image


# ==========================================
# 3. RESAMPLE TO REFERENCE GEOMETRY
# ==========================================

def resample_to_reference(
    moving_image,
    reference_image,
    interpolator=sitk.sitkLinear,
    default_value=0.0
):
    """
    Resample a moving image onto the physical grid
    of a reference image.

    This aligns voxel grids. It does NOT perform
    deformable or rigid registration.
    """

    resampler = sitk.ResampleImageFilter()

    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)

    resampler.SetTransform(
        sitk.Transform(3, sitk.sitkIdentity)
    )

    resampler.SetOutputPixelType(sitk.sitkFloat32)

    return resampler.Execute(moving_image)


# ==========================================
# 4. INTENSITY STATISTICS
# ==========================================

def calculate_intensity_statistics(volume):
    """
    Calculate robust intensity statistics.

    Original values are preserved.
    No clipping or normalization is performed.
    """

    volume = np.asarray(volume, dtype=np.float32)

    valid_values = volume[np.isfinite(volume)]

    if valid_values.size == 0:
        raise ValueError("Volume contains no finite intensity values.")

    return {
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "mean": float(np.mean(valid_values)),
        "std": float(np.std(valid_values)),
        "percentiles": {
            "p0_5": float(np.percentile(valid_values, 0.5)),
            "p1": float(np.percentile(valid_values, 1)),
            "p5": float(np.percentile(valid_values, 5)),
            "p25": float(np.percentile(valid_values, 25)),
            "p50": float(np.percentile(valid_values, 50)),
            "p75": float(np.percentile(valid_values, 75)),
            "p95": float(np.percentile(valid_values, 95)),
            "p99": float(np.percentile(valid_values, 99)),
            "p99_5": float(np.percentile(valid_values, 99.5)),
        }
    }


# ==========================================
# 5. ADAPTIVE PERCENTILE WINDOW
# ==========================================

def estimate_intensity_window(
    volume,
    lower_percentile=0.5,
    upper_percentile=99.5
):
    """
    Estimate an intensity window independently
    for a volume.

    Returns:
        lower_bound, upper_bound

    These values can later be used as supervision
    or input features for the HU-range hypernetwork.
    """

    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Invalid percentile configuration.")

    volume = np.asarray(volume, dtype=np.float32)
    valid_values = volume[np.isfinite(volume)]

    if valid_values.size == 0:
        raise ValueError("No finite intensity values available.")

    lower_bound, upper_bound = np.percentile(
        valid_values,
        [lower_percentile, upper_percentile]
    )

    if upper_bound <= lower_bound:
        raise ValueError("Invalid intensity window.")

    return float(lower_bound), float(upper_bound)


# ==========================================
# 6. NORMALIZATION
# ==========================================

def normalize_volume(
    volume,
    lower_bound,
    upper_bound
):
    """
    Normalize a volume to [-1, 1].

    Returns:
        normalized_volume
        normalization_parameters

    Original intensity values can be recovered
    using denormalize_volume().
    """

    if upper_bound <= lower_bound:
        raise ValueError("upper_bound must exceed lower_bound.")

    volume = np.asarray(volume, dtype=np.float32)

    clipped = np.clip(
        volume,
        lower_bound,
        upper_bound
    )

    normalized = (
        2.0 * (clipped - lower_bound)
        / (upper_bound - lower_bound)
    ) - 1.0

    parameters = {
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }

    return normalized.astype(np.float32), parameters


# ==========================================
# 7. REVERSE NORMALIZATION
# ==========================================

def denormalize_volume(normalized_volume, parameters):
    """
    Convert normalized values from [-1, 1]
    back to the original intensity window.
    """

    lower_bound = parameters["lower_bound"]
    upper_bound = parameters["upper_bound"]

    normalized_volume = np.asarray(
        normalized_volume,
        dtype=np.float32
    )

    volume = (
        (normalized_volume + 1.0)
        * 0.5
        * (upper_bound - lower_bound)
    ) + lower_bound

    return volume.astype(np.float32)


# ==========================================
# 8. GEOMETRY INFORMATION
# ==========================================

def get_geometry_information(image):
    """
    Extract physical geometry information.
    """

    return {
        "size": image.GetSize(),
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
    }
