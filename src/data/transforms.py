import torch


class PairedVolumeTransform:
    """
    Paired 3D augmentation for CBCT-to-sCT training.

    Input tensors:
        CBCT: [C, D, H, W]
        CT:   [C, D, H, W]

    Spatial transformations are always synchronized.

    CBCT-only intensity augmentations are optional and
    disabled by default to preserve intensity supervision.
    """

    def __init__(
        self,
        training=True,
        flip_probability=0.5,
        rotation_probability=0.5,
        noise_probability=0.0,
        noise_std=0.01,
        intensity_probability=0.0,
        intensity_scale_range=(0.95, 1.05),
        intensity_shift_range=(-0.02, 0.02),
    ):
        self.training = training

        self.flip_probability = flip_probability
        self.rotation_probability = rotation_probability

        self.noise_probability = noise_probability
        self.noise_std = noise_std

        self.intensity_probability = intensity_probability
        self.intensity_scale_range = intensity_scale_range
        self.intensity_shift_range = intensity_shift_range

    @staticmethod
    def _random_probability(probability):
        return torch.rand(1).item() < probability

    def _spatial_augmentation(self, cbct, ct):
        """
        Apply identical spatial transformations to both volumes.
        """

        # Random flips along depth, height and width.
        for axis in (1, 2, 3):
            if self._random_probability(self.flip_probability):
                cbct = torch.flip(cbct, dims=(axis,))
                ct = torch.flip(ct, dims=(axis,))

        # Rotate within the H-W plane.
        # Only use 90-degree rotations when H == W,
        # ensuring the patch dimensions remain unchanged.
        if (
            cbct.shape[2] == cbct.shape[3]
            and self._random_probability(self.rotation_probability)
        ):
            k = int(torch.randint(1, 4, (1,)).item())

            cbct = torch.rot90(
                cbct, k=k, dims=(2, 3)
            )

            ct = torch.rot90(
                ct, k=k, dims=(2, 3)
            )

        return cbct, ct

    def _cbct_intensity_augmentation(self, cbct):
        """
        Optional CBCT-only intensity augmentation.

        This can help simulate acquisition variability,
        but should be enabled only after examining the
        actual intensity distributions.
        """

        if self._random_probability(self.intensity_probability):
            scale = torch.empty(1).uniform_(
                *self.intensity_scale_range
            ).item()

            shift = torch.empty(1).uniform_(
                *self.intensity_shift_range
            ).item()

            cbct = cbct * scale + shift

        if self._random_probability(self.noise_probability):
            noise = torch.randn_like(cbct) * self.noise_std
            cbct = cbct + noise

        return cbct

    def __call__(self, cbct, ct):
        """
        Apply transformations to a paired volume.
        """

        if cbct.shape != ct.shape:
            raise ValueError(
                f"CBCT and CT shapes differ: "
                f"{cbct.shape} vs {ct.shape}"
            )

        if not self.training:
            return cbct, ct

        cbct, ct = self._spatial_augmentation(
            cbct, ct
        )

        cbct = self._cbct_intensity_augmentation(cbct)

        return cbct, ct


def get_train_transforms():
    """
    Conservative training augmentation.

    Intensity augmentation remains disabled until
    dataset intensity statistics have been reviewed.
    """

    return PairedVolumeTransform(
        training=True,
        flip_probability=0.5,
        rotation_probability=0.5,
        noise_probability=0.0,
        intensity_probability=0.0,
    )


def get_validation_transforms():
    """
    No augmentation during validation.
    """

    return PairedVolumeTransform(
        training=False
    )
