
import argparse
import torch
import yaml

from torch.utils.data import DataLoader

from src.data.dataset import SynthRADDataset
from src.models.fare3d import FARE3D
from src.losses import FlowMatchingLoss3D
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def load_splits(path):

    import json

    with open(path, "r") as file:
        splits = json.load(file)

    if "train" not in splits or "val" not in splits:
        raise ValueError(
            "Split file must contain train and val cases."
        )

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


def configure_stage_b(model, anchor_checkpoint, device):

    checkpoint = torch.load(
        anchor_checkpoint,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Freeze Stage A.
    frozen_modules = [
        model.anchor,
        model.smooth_field,
        model.detail_head
    ]

    for module in frozen_modules:
        module.eval()

        for parameter in module.parameters():
            parameter.requires_grad = False

    # Train conditioning encoders and flow refiner.
    trainable_modules = [
        model.anatomy_encoder,
        model.severity_encoder,
        model.refiner
    ]

    for module in trainable_modules:
        if module is None:
            raise RuntimeError(
                "Anatomy, severity and refiner modules must be enabled."
            )

        for parameter in module.parameters():
            parameter.requires_grad = True

    return model


def build_step_fn(flow_loss, device):

    def step_fn(model, batch):

        cbct = batch["cbct"].to(device)
        ct = batch["ct"].to(device)

        # Generate anchor and conditioning representations.
        # Anchor parameters are frozen, but conditioning
        # encoders remain trainable.
        outputs = model(cbct)

        anchor = outputs["anchor"].detach()

        loss_dict = flow_loss(
            model=model.refiner,
            anchor=anchor,
            target=ct,
            anatomy_tokens=outputs["anatomy_tokens"],
            severity_embedding=outputs["severity_embedding"]
        )

        return {
            "total": loss_dict["total"],
            "flow_matching": loss_dict["flow_matching"]
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
        config,
        train_cases,
        val_cases
    )

    model = FARE3D()

    model = configure_stage_b(
        model,
        args.anchor_checkpoint,
        device
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer_cfg = config["training"]["refiner"]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=optimizer_cfg["learning_rate"],
        weight_decay=optimizer_cfg["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=optimizer_cfg["epochs"]
    )

    flow_loss = FlowMatchingLoss3D(
        kernel_size=5
    )

    step_fn = build_step_fn(
        flow_loss,
        device
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=(
            config["output"]["checkpoint_dir"] + "/refiner"
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

        # Keep Stage A frozen and in evaluation mode.
        model.anchor.eval()
        model.smooth_field.eval()
        model.detail_head.eval()

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
            f"Train FM: {train_metrics['loss']:.6f} | "
            f"Validation FM: {val_metrics['loss']:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
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
        description="Train FARE-3D Stage B flow refiner."
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml"
    )

    parser.add_argument(
        "--splits",
        default="data/splits/split.json"
    )

    parser.add_argument(
        "--anchor-checkpoint",
        required=True
    )

    args = parser.parse_args()

    main(args)
