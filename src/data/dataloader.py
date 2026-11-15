import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SynthRADDataset
from src.data.transforms import (
    get_train_transforms,
    get_validation_transforms,
)


def seed_worker(worker_id):
    """
    Initialize each DataLoader worker with a reproducible seed.
    """

    worker_seed = torch.initial_seed() % (2 ** 32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_generator(seed):
    """
    Create a reproducible PyTorch random generator.
    """

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def create_dataloaders(
    train_cases,
    val_cases,
    patch_size=(64, 128, 128),
    batch_size=1,
    patches_per_volume=4,
    num_workers=4,
    seed=42,
    pin_memory=True,
):
    """
    Construct training and validation DataLoaders.

    Args:
        train_cases: List of training CBCT-CT pairs.
        val_cases: List of validation CBCT-CT pairs.
        patch_size: 3D patch dimensions (D, H, W).
        batch_size: Number of patches per batch.
        patches_per_volume: Training patches sampled per case.
        num_workers: Number of loading workers.
        seed: Reproducibility seed.

    Returns:
        train_loader, val_loader
    """

    train_dataset = SynthRADDataset(
        cases=train_cases,
        patch_size=patch_size,
        training=True,
        patches_per_volume=patches_per_volume,
        normalize=True,
        seed=seed,
    )

    val_dataset = SynthRADDataset(
        cases=val_cases,
        patch_size=patch_size,
        training=False,
        patches_per_volume=1,
        normalize=True,
        seed=seed,
    )

    # Attach augmentation pipelines.
    train_dataset.transform = get_train_transforms()
    val_dataset.transform = get_validation_transforms()

    train_generator = create_generator(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=train_generator,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        drop_last=False,
    )

    return train_loader, val_loader
