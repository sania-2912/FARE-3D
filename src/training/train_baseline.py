
import argparse
import json

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SynthRADDataset
from src.models.anchor import AnchorUNet3D
from src.losses import ReconstructionLoss3D
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def load_splits(path):

    with open(path, "r") as file:
        splits = json.load(file)

    return splits["train"], splits["val"]


def create_loaders(config, train_cases, val_cases):

    data_cfg = config["data"]

    common_args = {
        "patch_size": tuple(data_cfg["patch_size"]),
        "normalize": True,
        "seed": config["project"]["seed"]
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

    loader_args = {
        "num_workers": data_cfg["num_workers"],
        "pin_memory": data_cfg["pin_memory"],
        "persistent_workers": (
            data_cfg["persistent_workers"]
            and data_cfg["num_workers"] > 0
        )
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=False,
        **loader_args
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        **loader_args
    )

    return train_loader, val_loader


def build_step_fn(loss_function, device):

    def step_fn(model, batch):

        cbct = batch["cbct"].to(device)
        ct = batch["ct"].to(device)

        outputs = model(cbct)

        prediction = outputs["prediction"]

        loss_dict = loss_function(
            prediction,
            ct
        )

        return loss_dict

    return step_fn


def main(args):

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    set_seed(config["project"]["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Baseline training device: {device}")

    train_cases, val_cases = load_splits(args.splits)

    train_loader, val_loader = create_loaders(
        config,
        train_cases,
        val_cases
    )

    # Plain 3D U-Net baseline.
    model = AnchorUNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=config["model"]["anchor"]["base_channels"],
        channel_multipliers=tuple(
            config["model"]["anchor"]["channel_multipliers"]
        ),
        num_res_blocks=config["model"]["anchor"]["num_res_blocks"]
    )

    optimizer_cfg = config["training"]["anchor"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_cfg["learning_rate"],
        weight_decay=optimizer_cfg["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=optimizer_cfg["epochs"]
    )

    loss_function = ReconstructionLoss3D(
        l1_weight=1.0,
        gradient_weight=0.1
    )

    step_fn = build_step_fn(
        loss_function,
        device
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=(
            config["output"]["checkpoint_dir"] + "/baseline"
        ),
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
            train_loader,
            step_fn
        )

        val_metrics = trainer.validate_epoch(
            val_loader,
            step_fn
        )

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Validation Loss: {val_metrics['loss']:.6f}"
        )

        trainer.save_checkpoint(
            epoch=epoch,
            val_loss=val_metrics["loss"],
            filename="latest.pt"
        )

        if (epoch + 1) % save_every == 0:

            trainer.save_checkpoint(
                epoch=epoch,
                val_loss=val_metrics["loss"],
                filename=f"epoch_{epoch + 1}.pt"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train the FARE-3D 3D U-Net baseline."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml"
    )

    parser.add_argument(
        "--splits",
        default="data/splits/split.json"
    )

    args = parser.parse_args()

    main(args)
