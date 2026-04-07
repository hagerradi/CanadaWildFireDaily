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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.segmentation import MeanIoU, DiceScore
from torchmetrics.classification import (
    BinaryPrecision, 
    BinaryRecall, 
    BinaryAUROC, 
    BinaryAveragePrecision # AUC-PR (Area Under the Precision-Recall Curve)
)
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        scheduler: ReduceLROnPlateau,
        loss_fn: nn.Module,
        config: TrainingConfig,
        logger: CometLogger | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.logger = logger
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)

        # --- DYNAMIC METRICS SETUP ---
        # Read the flag to determine if we are in 3-class or 2-class mode
        num_classes = 3 if self.config.use_cumuarea else 2

        self.iou_metric = MeanIoU(
            num_classes=num_classes, 
            per_class=True, 
            input_format='index',
            include_background=True
        ).to(self.device)

        self.dice_metric = DiceScore(
            num_classes=num_classes, 
            average="none",
            input_format="index",
            include_background=True
        ).to(self.device)

        # Binary Classification Metrics (Only if 2-class/binary mode)
        if not self.config.use_cumuarea:
            self.precision_metric = BinaryPrecision().to(self.device)
            self.recall_metric = BinaryRecall().to(self.device)
            self.auroc_metric = BinaryAUROC().to(self.device)
            self.aucpr_metric = BinaryAveragePrecision().to(self.device)

        if self.logger is not None:
            self.logger.log_params(
                {
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "device": str(self.device),
                    "notes":"With satellite - dynamic flexible classes",
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

        # Initialize the tracker
        best_val_loss = float('inf')

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_iou_metrics = self.validate(val_loader, epoch)

            self.scheduler.step(val_loss)
            
            print(f"Epoch [{epoch}/{self.config.num_epochs}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if self.logger is not None:
                self.logger.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss},
                    epoch=epoch,
                )
                self.logger.log_metrics(val_iou_metrics, epoch=epoch)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_metrics({"learning_rate": current_lr}, epoch=epoch)

            self.save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt", epoch)

            # Check if this is the best model so far
            if val_loss < best_val_loss:
                print(f"*** Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best model! ***")
                best_val_loss = val_loss
                
                # Overwrite the best_checkpoint.pt with the new best weights
                best_ckpt_path = checkpoint_dir / "best_checkpoint.pt"
                self.save_checkpoint(best_ckpt_path, epoch)

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

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Epoch {epoch}")

        for batch_idx, (images, masks) in pbar:

            images: Tensor = images.to(self.device)
            masks: Tensor = masks.long().to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss: Tensor = self.loss_fn(predictions, masks)
            
            # --- Catch NaNs ---
            if torch.isnan(loss):
                print(f"\n[!] NaN Loss at batch {batch_idx}. Skipping parameter update to save the model.")
                continue
                
            loss.backward()
            
            # --- Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % self.config.log_interval == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch} | batch {batch_idx + 1}/{len(loader)} | loss={avg:.4f}")

        return total_loss / len(loader)

    def validate(self, loader: DataLoader, epoch: int | None = None) -> tuple[float, dict[str, float]]:
        """Evaluate the model on a validation or test split and calculate IoU."""
        self.model.eval()
        total_loss = 0.0

        # Make sure the metric is empty before starting the validation epoch
        self.iou_metric.reset()
        self.dice_metric.reset()

        if not self.config.use_cumuarea:
            self.precision_metric.reset()
            self.recall_metric.reset()
            self.auroc_metric.reset()
            self.aucpr_metric.reset()

        pbar = tqdm(loader, total=len(loader), desc="Validating", leave=False)
        
        # We only want to log ONE image per epoch, so we use a flag
        logged_image_this_epoch = False

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images, masks = batch[0], batch[1]
                
                images: Tensor = images.to(self.device)
                masks: Tensor = masks.long().to(self.device)
                
                predictions = self.model(images)
                loss: Tensor = self.loss_fn(predictions, masks)
                total_loss += loss.item()

                # --- METRICS CALCULATION ---
                pred_probs = None

                if predictions.shape[1] == 1:
                    # 1-channel binary: values > 0.0 are class 1 (Fire), otherwise 0 (Background)
                    # Squeeze ensures shape is (B, H, W) to match masks
                    pred_classes = (predictions > 0.0).squeeze(1).long()
                    # Get probabilities using sigmoid for the AUC metrics
                    pred_probs = torch.sigmoid(predictions).squeeze(1)
                else:
                    # Multi-class: argmax across channels
                    pred_classes = torch.argmax(predictions, dim=1)
                
                self.iou_metric.update(pred_classes, masks)
                self.dice_metric.update(pred_classes, masks)

                # Update binary metrics (Only if 2-class setup AND we have probabilities)
                if not self.config.use_cumuarea and pred_probs is not None:
                    self.precision_metric.update(pred_classes, masks)
                    self.recall_metric.update(pred_classes, masks)
                    self.auroc_metric.update(pred_probs, masks)
                    self.aucpr_metric.update(pred_probs, masks)

                # ---  IMAGE LOGGING ---
                if not logged_image_this_epoch and self.logger is not None and epoch is not None:
                    
                    min_pixels = 30
                    selected_idx = -1
                    
                    is_3_class = self.config.use_cumuarea
                    vmax_val = 2 if is_3_class else 1
                    
                    # Search through every image in the current batch
                    for i in range(masks.size(0)):
                        current_mask = masks[i]
                        
                        if is_3_class:
                            # 3-Class: Need both Old Fire (1) and New Fire (2)
                            c1_count = (current_mask == 1).sum().item()
                            c2_count = (current_mask == 2).sum().item()
                            is_interesting = (c1_count >= min_pixels) and (c2_count >= min_pixels)
                        else:
                            # 2-Class: Just need enough New Fire (1)
                            c1_count = (current_mask == 1).sum().item()
                            is_interesting = (c1_count >= min_pixels)
                        
                        # If the image has enough of our target classes, pick it and stop searching
                        if is_interesting:
                            selected_idx = i
                            break
                    
                    if selected_idx == -1 and batch_idx == len(loader) - 1:
                        selected_idx = 0

                    # If we found a good image in this batch, generate and log the plot
                    if selected_idx != -1:
                        true_mask = masks[selected_idx].cpu().numpy()
                        pred_mask = pred_classes[selected_idx].cpu().numpy()
                        
                        # Draw a side-by-side plot
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        axes[0].imshow(true_mask, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[0].set_title("Ground Truth")
                        axes[0].axis('off')
                        
                        axes[1].imshow(pred_mask, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[1].set_title(f"Prediction (Epoch {epoch})")
                        axes[1].axis('off')
                        
                        # Save locally, log to Comet, then delete the local file
                        temp_img_path = f"temp_val_epoch_{epoch}.png"
                        plt.savefig(temp_img_path, bbox_inches='tight')
                        plt.close(fig)
                        
                        self.logger.log_image(temp_img_path, name=f"Validation/Epoch_{epoch}", epoch=epoch)
                        os.remove(temp_img_path)
                        
                        # Lock it so we don't log any more images this epoch
                        logged_image_this_epoch = True
                # -------------------------------
                
                pbar.set_postfix({"loss": f"{(total_loss / (batch_idx + 1)):.4f}"})

        # --- COMPUTE Final Epoch Metrics ---
        class_ious = self.iou_metric.compute() 
        class_dices = self.dice_metric.compute()

        # Safe Macro IoU calculation (ignoring -1)
        valid_iou_mask = class_ious != -1
        macro_iou = class_ious[valid_iou_mask].mean().item() if valid_iou_mask.any() else 0.0

        # Safe Macro Dice calculation (ignoring -1)
        valid_dice_mask = class_dices != -1
        macro_dice = class_dices[valid_dice_mask].mean().item() if valid_dice_mask.any() else 0.0
        
        # Start with the global metrics
        final_metrics = {
            "Macro_IoU": macro_iou,
            "Macro_Dice": macro_dice
        }

        # Dynamically add per-class metrics based on the number of output classes
        if len(class_ious) == 3:
            final_metrics.update({
                "IoU_0_Background": class_ious[0].item(),
                "IoU_1_Old_Fire": class_ious[1].item(),
                "IoU_2_New_Fire": class_ious[2].item(),
                
                "Dice_0_Background": class_dices[0].item(),
                "Dice_1_Old_Fire": class_dices[1].item(),
                "Dice_2_New_Fire": class_dices[2].item(),
            })
        
        elif len(class_ious) == 2:

            binary_metrics = {
                "IoU_0_Background": class_ious[0].item(),
                "IoU_1_New_Fire": class_ious[1].item(),
                
                "Dice_0_Background": class_dices[0].item(),
                "Dice_1_New_Fire": class_dices[1].item(),
            }

            metrics_updated = all(
                getattr(metric, "update_called", False)
                for metric in (
                    self.precision_metric,
                    self.recall_metric,
                    self.auroc_metric,
                    self.aucpr_metric,
                )
            )

            if metrics_updated:
                binary_metrics.update({
                    "Precision": self.precision_metric.compute().item(),
                    "Recall": self.recall_metric.compute().item(),
                    "AUC_ROC": self.auroc_metric.compute().item(),
                    "AUC_PR": self.aucpr_metric.compute().item(),
                })

            final_metrics.update(binary_metrics)
            
        avg_loss = total_loss / len(loader)

        return avg_loss, final_metrics

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        """Save model and optimizer state to disk."""
        """Save model, optimizer, and scheduler state to disk."""
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        # Only save scheduler if it exists
        if self.scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = self.scheduler.state_dict()
            
        torch.save(checkpoint_data, path)

    def load_checkpoint(self, path: str | Path) -> int:
        """Load model and optimizer state from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Restore scheduler if it was saved and exists in the current trainer
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)