"""
Trainer: encapsulates the training and validation loops.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.config import TrainingConfig

if TYPE_CHECKING:
    from src.logger import CometLogger


class Trainer:
    """Manages the training loop, validation, and checkpointing.

    Args:
        model: PyTorch model to train.
        optimizer: Optimizer instance.
        loss_fn: Loss function (criterion).
        config: Training-specific configuration.
        logger: Optional :class:`~src.logger.CometLogger` instance.  When
            provided, hyper-parameters are logged at construction time and
            train/val losses are logged after every epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        config: TrainingConfig,
        logger: CometLogger | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.logger = logger
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)

        if self.logger is not None:
            self.logger.log_params(
                {
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "device": str(self.device),
                }
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for the training split.
            val_loader: DataLoader for the validation split.
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            print(f"Epoch [{epoch}/{self.config.num_epochs}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if self.logger is not None:
                self.logger.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss},
                    epoch=epoch,
                )

            self.save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt", epoch)

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run one training epoch.

        Args:
            loader: DataLoader for the training split.
            epoch: Current epoch number (used for logging).

        Returns:
            Average training loss over all batches.
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, masks) in enumerate(loader):
            images: Tensor = images.to(self.device)
            masks: Tensor = masks.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss: Tensor = self.loss_fn(predictions, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % self.config.log_interval == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch} | batch {batch_idx + 1}/{len(loader)} | loss={avg:.4f}")

        return total_loss / len(loader)

    def validate(self, loader: DataLoader) -> float:
        """Evaluate the model on a validation or test split.

        Args:
            loader: DataLoader for the validation/test split.

        Returns:
            Average loss over all batches.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, masks in loader:
                images: Tensor = images.to(self.device)
                masks: Tensor = masks.to(self.device)
                predictions = self.model(images)
                loss: Tensor = self.loss_fn(predictions, masks)
                total_loss += loss.item()

        return total_loss / len(loader)

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        """Save model and optimizer state to disk.

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch (saved for reference).
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        """Load model and optimizer state from a checkpoint file.

        Args:
            path: File path of the checkpoint to load.

        Returns:
            Epoch number stored in the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve the device string to a :class:`torch.device`.

        Args:
            device_str: One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.

        Returns:
            Resolved :class:`torch.device`.
        """
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)
