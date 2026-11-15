from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import (
    load_volume,
    resample_to_reference,
    estimate_intensity_window,
    normalize_volume,
)


class SynthRADDataset(Dataset):
    """
    Paired 3D CBCT-to-CT dataset for FARE-3D.

    Expected input:
        cases = [
            {
                "case_id": "case_001",
                "cbct": "/path/to/CBCT/case_001.nii.gz",
                "ct": "/path/to/CT/case_001.nii.gz"
            }
        ]

    Output:
        cbct: [1, D, H, W]
        ct:   [1, D, H, W]

    Intensity windows and metadata are also returned.
    """

    def __init__(
        self,
        cases,
        patch_size=(64, 128, 128),
        training=True,
        cbct_percentiles=(0.5, 99.5),
        ct_percentiles=(0.5, 99.5),
        patches_per_volume=4,
        normalize=True,
        seed=42,
    ):
        self.cases = cases
        self.patch_size = tuple(patch_size)
        self.training = training

        self.cbct_percentiles = cbct_percentiles
        self.ct_percentiles = ct_percentiles

        self.patches_per_volume = patches_per_volume
        self.normalize = normalize
        self.seed = seed

        if len(self.patch_size) != 3:
            raise ValueError("patch_size must contain (D, H, W).")

        if any(size <= 0 for size in self.patch_size):
            raise ValueError("All patch dimensions must be positive.")

        if self.patches_per_volume < 1:
            raise ValueError("patches_per_volume must be at least 1.")

        self.rng = np.random.default_rng(seed)

        # Training samples multiple patches per volume.
        # Validation uses one deterministic center patch.
        self.samples_per_case = (
            self.patches_per_volume if training else 1
        )

    def __len__(self):
        return len(self.cases) * self.samples_per_case

    def _load_pair(self, case):
        cbct_image, cbct_volume = load_volume(case["cbct"])
        ct_image, ct_volume = load_volume(case["ct"])

        # Align CT to CBCT physical geometry if needed.
        same_geometry = (
            cbct_image.GetSize() == ct_image.GetSize()
            and np.allclose(
                cbct_image.GetSpacing(),
                ct_image.GetSpacing(),
                atol=1e-4
            )
            and np.allclose(
                cbct_image.GetOrigin(),
                ct_image.GetOrigin(),
                atol=1e-4
            )
            and np.allclose(
                cbct_image.GetDirection(),
                ct_image.GetDirection(),
                atol=1e-5
            )
        )

        if not same_geometry:
            ct_image = resample_to_reference(
                moving_image=ct_image,
                reference_image=cbct_image,
            )

            ct_volume = __import__("SimpleITK").GetArrayFromImage(
                ct_image
            ).astype(np.float32)

        if cbct_volume.shape != ct_volume.shape:
            raise ValueError(
                f"Pair shape mismatch for {case['case_id']}: "
                f"{cbct_volume.shape} vs {ct_volume.shape}"
            )

        return (
            cbct_image,
            cbct_volume,
            ct_volume,
        )

    def _get_patch_start(self, volume_shape, patch_index):
        """
        Select a valid patch location.

        Training:
            Random sampling.

        Validation:
            Deterministic center sampling.
        """

        starts = []

        for axis, (volume_dim, patch_dim) in enumerate(
            zip(volume_shape, self.patch_size)
        ):
            if volume_dim < patch_dim:
                raise ValueError(
                    f"Volume dimension {volume_dim} is smaller "
                    f"than patch dimension {patch_dim}. "
                    "Resampling or padding is required."
                )

            max_start = volume_dim - patch_dim

            if self.training:
                start = self.rng.integers(0, max_start + 1)
            else:
                start = max_start // 2

            starts.append(int(start))

        return tuple(starts)

    def _extract_patch(self, volume, start):
        d, h, w = start
        pd, ph, pw = self.patch_size

        return volume[
            d:d + pd,
            h:h + ph,
            w:w + pw
        ]

    def _normalize_pair(self, cbct, ct):
        cbct_low, cbct_high = estimate_intensity_window(
            cbct,
            *self.cbct_percentiles
        )

        ct_low, ct_high = estimate_intensity_window(
            ct,
            *self.ct_percentiles
        )

        cbct, cbct_params = normalize_volume(
            cbct,
            cbct_low,
            cbct_high
        )

        ct, ct_params = normalize_volume(
            ct,
            ct_low,
            ct_high
        )

        return cbct, ct, {
            "cbct": cbct_params,
            "ct": ct_params,
        }

    def __getitem__(self, index):
        case_index = index // self.samples_per_case
        case = self.cases[case_index]

        cbct_image, cbct_volume, ct_volume = (
            self._load_pair(case)
        )

        # Select exactly the same spatial patch for both.
        start = self._get_patch_start(
            cbct_volume.shape,
            index
        )

        cbct_patch = self._extract_patch(
            cbct_volume,
            start
        )

        ct_patch = self._extract_patch(
            ct_volume,
            start
        )

        if self.normalize:
            (
                cbct_patch,
                ct_patch,
                intensity_params,
            ) = self._normalize_pair(
                cbct_patch,
                ct_patch
            )
        else:
            intensity_params = None

        # Add channel dimension:
        # [D, H, W] -> [1, D, H, W]
        cbct_tensor = torch.from_numpy(
            np.ascontiguousarray(cbct_patch[None])
        ).float()

        ct_tensor = torch.from_numpy(
            np.ascontiguousarray(ct_patch[None])
        ).float()

        return {
            "cbct": cbct_tensor,
            "ct": ct_tensor,
            "case_id": str(case["case_id"]),
            "patch_start": start,
            "spacing": cbct_image.GetSpacing(),
            "intensity_params": intensity_params,
        }
