
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SynthRADDataset
from src.models.fare3d import FARE3D
from src.losses import ReconstructionLoss3D, FrequencyLoss3D
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def load_splits(split_path):
    with open(split_path, "r") as file:
        splits = json.load(file)

    if "train" not in splits or "val" not in splits:
        raise ValueError(
            "Split JSON must contain 'train' and 'val' lists."
        )

    return splits["train"], splits["val"]


def create_loaders(config, train_cases, val_cases):

    data_cfg = config["data"]

    common_args = {
        "patch_size": tuple(data_cfg["patch_size"]),
        "normalize": True,
        "seed": config["project"]["seed"],
    }

    train_dataset = SynthRADDataset(
        train_cases,
        training=True,
        patches_per_volume=data_cfg.get(
            "patches_per_volume", 4
        ),
        **common_args
    )

    val_dataset = SynthRADDataset(
        val_cases,
        training=False,
        patches_per_volume=1,
        **common_args
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        persistent_workers=(
            data_cfg["persistent_workers"]
            and data_cfg["num_workers"] > 0
        ),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        persistent_workers=(
            data_cfg["persistent_workers"]
            and data_cfg["num_workers"] > 0
        )
    )

    return train_loader, val_loader


def configure_stage_a(model):

    # Only Stage A components are trainable.
    trainable_modules = [
        model.anchor,
        model.smooth_field,
        model.detail_head
    ]

    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    return model


def build_step_fn(reconstruction_loss, frequency_loss,
                  frequency_weight, device):

    def step_fn(model, batch):

        cbct = batch["cbct"].to(device)
        ct = batch["ct"].to(device)

        outputs = model(cbct)

        prediction = outputs["anchor"]

        reconstruction = reconstruction_loss(
            prediction, ct
        )

        frequency = frequency_loss(
            prediction, ct
        )

        total_loss = (
            reconstruction["total"]
            + frequency_weight * frequency["total"]
        )

        return {
            "total": total_loss,
            "reconstruction": reconstruction["total"],
            "frequency": frequency["total"],
            "l1": reconstruction["l1"],
            "gradient": reconstruction["gradient"]
        }

    return step_fn


def main(args):

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    set_seed(config["project"]["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    train_cases, val_cases = load_splits(args.splits)

    train_loader, val_loader = create_loaders(
        config, train_cases, val_cases
    )

    # Initialize FARE-3D.
    model = FARE3D()
    model = configure_stage_a(model)

    optimizer_cfg = config["training"]["anchor"]

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=optimizer_cfg["learning_rate"],
        weight_decay=optimizer_cfg["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=optimizer_cfg["epochs"]
    )

    loss_cfg = config["loss"]

    reconstruction_loss = ReconstructionLoss3D(
        l1_weight=loss_cfg["anchor_l1"],
        gradient_weight=loss_cfg["edge"]
    )

    frequency_loss = FrequencyLoss3D(
        low_weight=1.0,
        high_weight=1.0,
        leakage_weight=0.0
    )

    step_fn = build_step_fn(
        reconstruction_loss,
        frequency_loss,
        frequency_weight=loss_cfg["high_frequency"],
        device=device
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config["output"]["checkpoint_dir"]
        + "/anchor",
        mixed_precision=config["training"]["mixed_precision"],
        gradient_accumulation_steps=config["training"][
            "gradient_accumulation_steps"
        ],
        gradient_clip=config["training"]["gradient_clip"]
    )

    epochs = optimizer_cfg["epochs"]
    save_every = config["training"]["save_every"]

    for epoch in range(epochs):

        train_metrics = trainer.train_epoch(
            train_loader, step_fn
        )

        val_metrics = trainer.validate_epoch(
            val_loader, step_fn
        )

        scheduler.step()

        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train: {train_loss:.6f} | "
            f"Validation: {val_loss:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        trainer.save_checkpoint(
            epoch=epoch,
            val_loss=val_loss,
            filename="latest.pt"
        )

        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(
                epoch=epoch,
                val_loss=val_loss,
                filename=f"epoch_{epoch + 1}.pt"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train FARE-3D Stage A anchor."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml"
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="data/splits/split.json"
    )

    args = parser.parse_args()
    main(args)
