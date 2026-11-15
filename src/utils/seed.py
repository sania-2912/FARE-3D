
import os
import random

# Set before CUDA libraries are initialized.
os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def set_seed(seed=42, deterministic=True):
    """
    Set random seeds across Python, NumPy and PyTorch.

    Args:
        seed: Global random seed.
        deterministic: Enable deterministic operations
                       where supported.
    """

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Backend configuration
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(
            True,
            warn_only=True
        )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"Random seed set to: {seed}")
    print(f"Deterministic mode: {deterministic}")


def seed_worker(worker_id):
    """
    Initialize random seeds for individual DataLoader workers.
    """

    worker_seed = torch.initial_seed() % (2 ** 32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    """
    Automatically select the available compute device.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    else:
        device = torch.device("cpu")
        print("CUDA unavailable. Using CPU.")

    return device
