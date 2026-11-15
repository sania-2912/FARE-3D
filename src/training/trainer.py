
from pathlib import Path
import torch
from torch.amp import autocast, GradScaler


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        device,
        checkpoint_dir="checkpoints",
        scheduler=None,
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        gradient_clip=1.0
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = torch.device(device)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.gradient_accumulation_steps = max(
            1, gradient_accumulation_steps
        )

        self.gradient_clip = gradient_clip

        self.mixed_precision = mixed_precision.lower()

        if self.mixed_precision not in [
            "bf16", "fp16", "none"
        ]:
            raise ValueError(
                "mixed_precision must be bf16, fp16 or none."
            )

        if self.mixed_precision == "bf16":
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "BF16 requested but unsupported on this device."
                )

        self.amp_enabled = (
            self.device.type == "cuda"
            and self.mixed_precision != "none"
        )

        self.amp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16
        }.get(self.mixed_precision)

        self.scaler = GradScaler(
            "cuda",
            enabled=(
                self.amp_enabled
                and self.mixed_precision == "fp16"
            )
        )

        self.global_step = 0
        self.best_val_loss = float("inf")

        self.model.to(self.device)

    def _autocast_context(self):

        if not self.amp_enabled:
            return autocast(
                device_type=self.device.type,
                enabled=False
            )

        return autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=True
        )

    def train_epoch(self, dataloader, step_fn):

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        num_batches = len(dataloader)

        if num_batches == 0:
            raise RuntimeError("Training dataloader is empty.")

        for batch_index, batch in enumerate(dataloader):

            batch = self._move_to_device(batch)

            with self._autocast_context():
                loss_dict = step_fn(
                    self.model,
                    batch
                )

                loss = loss_dict["total"]

                if not torch.isfinite(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError(
                        f"Non-finite training loss at batch {batch_index}"
                    )

                scaled_loss = (
                    loss / self.gradient_accumulation_steps
                )

            self.scaler.scale(scaled_loss).backward()

            is_accumulation_step = (
                (batch_index + 1)
                % self.gradient_accumulation_steps == 0
            )

            is_last_batch = (
                batch_index + 1 == num_batches
            )

            if is_accumulation_step or is_last_batch:

                # Correct the final partial accumulation group.
                if is_last_batch and not is_accumulation_step:
                    remainder = (
                        num_batches
                        % self.gradient_accumulation_steps
                    )

                    if remainder:
                        correction = (
                            self.gradient_accumulation_steps
                            / remainder
                        )

                        for parameter in self.model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(correction)

                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1

            total_loss += loss.detach().float().item()

        return {
            "loss": total_loss / num_batches,
            "global_step": self.global_step
        }

    @torch.no_grad()
    def validate_epoch(self, dataloader, step_fn):

        self.model.eval()

        total_loss = 0.0
        num_batches = len(dataloader)

        if num_batches == 0:
            raise RuntimeError("Validation dataloader is empty.")

        for batch in dataloader:

            batch = self._move_to_device(batch)

            with self._autocast_context():
                loss_dict = step_fn(
                    self.model,
                    batch
                )

            loss = loss_dict["total"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite validation loss detected."
                )

            total_loss += loss.float().item()

        return {
            "loss": total_loss / num_batches
        }

    def _move_to_device(self, batch):

        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)

        if isinstance(batch, dict):
            return {
                key: self._move_to_device(value)
                for key, value in batch.items()
            }

        if isinstance(batch, (list, tuple)):
            return type(batch)(
                self._move_to_device(value)
                for value in batch
            )

        return batch

    def save_checkpoint(
        self,
        epoch,
        val_loss,
        filename="latest.pt"
    ):

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "scaler_state_dict": self.scaler.state_dict()
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = (
                self.scheduler.state_dict()
            )

        path = self.checkpoint_dir / filename

        torch.save(checkpoint, path)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

            best_path = self.checkpoint_dir / "best.pt"

            torch.save(checkpoint, best_path)

        return path

    def load_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if (
            self.scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        self.global_step = checkpoint.get(
            "global_step", 0
        )

        self.best_val_loss = checkpoint.get(
            "best_val_loss", float("inf")
        )

        return checkpoint.get("epoch", 0) + 1
