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
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchmetrics.segmentation import MeanIoU, DiceScore
from torchmetrics.classification import (
    BinaryJaccardIndex,    # Binary IoU
    BinaryF1Score,         # Binary Dice
    BinaryPrecision, 
    BinaryRecall, 
    BinaryAUROC, 
    BinaryAveragePrecision # AUC-PR
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import uuid

from src.config import TrainingConfig

if TYPE_CHECKING:
    from src.logger import CometLogger


class Trainer:
    """Manages the training loop, validation, and checkpointing.

    Args:
        model: PyTorch model to train.
        optimizer: Optimizer instance.
        scheduler: Scheduler instqnce.
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
        scheduler: LRScheduler,
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

        # Enable automatic mixed precision only on CUDA
        self._use_amp = self.device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        self.binary_classification_threshold = 0.5

        # Initialize separate metrics for train and validation phases
        self.train_metrics = self._init_metrics()
        self.val_metrics = self._init_metrics()
        # TEMP
        self.val_metrics_filtered = self._init_metrics()

        if self.logger is not None:
            self.logger.log_params(
                {
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "device": str(self.device),
                    "notes":"With satellite - dynamic flexible classes (Train & Val Metrics tracked)",
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

        # Initialize tracking for MAX IoU instead of MIN Loss
        best_val_iou = -float('inf')

        # Determine the target key to track based on our config
        target_metric_key = "Val_Macro_IoU" if self.config.use_cumuarea else "Val_IoU_1_New_Fire"

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss, train_metrics_results = self.train_epoch(train_loader, epoch)
            val_loss, val_metrics_results = self.validate(val_loader, epoch)
            
            current_val_iou = val_metrics_results[target_metric_key]

            print(f"Epoch [{epoch}/{self.config.num_epochs}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_iou={current_val_iou:.4f}")

            if self.logger is not None:
                self.logger.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss},
                    step=epoch,
                    epoch=epoch,
                )
                self.logger.log_metrics(train_metrics_results, step=epoch, epoch=epoch)
                self.logger.log_metrics(val_metrics_results, step=epoch, epoch=epoch)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_metrics({"learning_rate": current_lr}, step=epoch, epoch=epoch)

            # Check if this is the best model so far based on IoU
            if current_val_iou > best_val_iou:
                print(f"*** Validation IoU improved from {best_val_iou:.4f} to {current_val_iou:.4f}. Saving best model! ***")
                best_val_iou = current_val_iou
                
                best_ckpt_path = checkpoint_dir / "best_checkpoint.pt"
                self.save_checkpoint(best_ckpt_path, epoch)

    def train_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, dict[str, float]]:
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

        for batch_idx, batch_data in pbar:

            if len(batch_data) == 3:
                images, masks, delta_t = batch_data
                delta_t = delta_t.to(self.device, non_blocking=True)
            else:
                images, masks = batch_data
           
            images: Tensor = images.to(self.device, non_blocking=True)
            masks: Tensor = masks.long().to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(self.device.type, enabled=self._use_amp):
                if len(batch_data) == 3:
                    predictions = self.model(images, delta_t)
                else:
                    predictions = self.model(images)

                loss: Tensor = self.loss_fn(predictions, masks)

            self._scaler.scale(loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()

            total_loss += loss.item()

            self.scheduler.step()

            # Update train metrics without tracking gradients for the metric computation
            with torch.no_grad():
                self._update_metrics(self.train_metrics, predictions.detach(), masks)

            if (batch_idx + 1) % self.config.log_interval == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch} | batch {batch_idx + 1}/{len(loader)} | loss={avg:.4f}")

        # Compute final train metrics and reset states for the next epoch
        train_metric_results = self._compute_and_reset_metrics(self.train_metrics, prefix="Train_")
        return total_loss / len(loader), train_metric_results

    def validate(self, loader: DataLoader, epoch: int | None = None) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(loader, total=len(loader), desc="Validating", leave=False)
        logged_image_this_epoch = False

        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):
                
                images, masks = batch[0], batch[1]
                
                images: Tensor = images.to(self.device, non_blocking=True)
                masks: Tensor = masks.long().to(self.device, non_blocking=True)

                if len(batch) == 3:
                    delta_t = batch[2]
                    delta_t = delta_t.to(self.device, non_blocking=True)
                
                if len(batch) == 3:
                    predictions = self.model(images, delta_t)
                else:
                    predictions = self.model(images)

                loss: Tensor = self.loss_fn(predictions, masks)
                total_loss += loss.item()

                # Update validation metrics
                self._update_metrics(self.val_metrics, predictions, masks)

                # ==========================================================
                # EXPERIMENTAL COARSE + CLEANED METRICS BLOCK
                # Only runs when use_filtered_metrics=True in config
                # ==========================================================
                if self.config.use_filtered_metrics:
                    downscale = 2  # Set to 2 for 180m, or 4 for 360m

                    # ----------------------------------------------------------
                    # 1. CLEAN THE NOISY GROUND TRUTH (Binary Closing)
                    # ----------------------------------------------------------
                    raw_masks = masks.float().unsqueeze(1)
                    
                    # Dilation: Fills in the accidental gaps in your manual grid
                    dilated_masks = F.max_pool2d(raw_masks, kernel_size=3, stride=1, padding=1)
                    # Erosion (Negative-Max Trick): Shrinks the inflated perimeter back to normal
                    closed_masks = -F.max_pool2d(-dilated_masks, kernel_size=3, stride=1, padding=1)
                    
                    # Downsample the CLEANED masks
                    coarse_masks = F.max_pool2d(closed_masks, kernel_size=downscale, stride=downscale).squeeze(1).long()
                    
                    # ----------------------------------------------------------
                    # 2. CLEAN THE PREDICTIONS (Binary Closing)
                    # ----------------------------------------------------------
                    # Dilation: Fills in the model's "salt-and-pepper" holes
                    dilated_preds = F.max_pool2d(predictions, kernel_size=3, stride=1, padding=1)
                    # Erosion: Shrinks the inflated perimeter back to normal
                    closed_preds = -F.max_pool2d(-dilated_preds, kernel_size=3, stride=1, padding=1)

                    # Downsample the CLEANED predictions
                    coarse_preds = F.max_pool2d(closed_preds, kernel_size=downscale, stride=downscale)

                    # ----------------------------------------------------------
                    # 3. CALCULATE METRICS
                    # ----------------------------------------------------------
                    self._update_metrics(self.val_metrics_filtered, coarse_preds, coarse_masks)
                # ==========================================================

                # --- UPGRADED IMAGE LOGGING LOGIC ---
                if not logged_image_this_epoch and self.logger is not None and epoch is not None:
                    min_pixels = 30
                    selected_idx = -1
                    is_3_class = self.config.use_cumuarea
                    vmax_val = 2 if is_3_class else 1
                    
                    for i in range(masks.size(0)):
                        current_mask = masks[i]
                        
                        if is_3_class:
                            c1_count = (current_mask == 1).sum().item()
                            c2_count = (current_mask == 2).sum().item()
                            is_interesting = (c1_count >= min_pixels) and (c2_count >= min_pixels)
                        else:
                            c1_count = (current_mask == 1).sum().item()
                            is_interesting = (c1_count >= min_pixels)
                        
                        if is_interesting:
                            selected_idx = i
                            break
                    
                    if selected_idx == -1 and batch_idx == len(loader) - 1:
                        selected_idx = 0

                    if selected_idx != -1:
                        # Grab predictions for the selected image to plot
                        if predictions.shape[1] == 1:
                            pred_probs = torch.sigmoid(predictions[selected_idx]).squeeze(0)
                            pred_mask = (pred_probs > self.binary_classification_threshold).long().cpu().numpy()
                        else:
                            pred_mask = torch.argmax(predictions[selected_idx], dim=0).cpu().numpy()

                        true_mask = masks[selected_idx].cpu().numpy()
                        
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        axes[0].imshow(true_mask, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[0].set_title("Ground Truth")
                        axes[0].axis('off')
                        
                        axes[1].imshow(pred_mask, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[1].set_title(f"Prediction (Epoch {epoch})")
                        axes[1].axis('off')
                        
                        # temp_img_path = f"temp_val_epoch_{epoch}.png"
                        random_id = uuid.uuid4().hex[:6]
                        temp_img_path = f"temp_val_epoch_{epoch}_{random_id}.png"
                        plt.savefig(temp_img_path, bbox_inches='tight')
                        plt.close(fig)
                        
                        self.logger.log_image(temp_img_path, name=f"Validation/Epoch_{epoch}", epoch=epoch)
                        os.remove(temp_img_path)
                        
                        logged_image_this_epoch = True
                
                pbar.set_postfix({"loss": f"{(total_loss / (batch_idx + 1)):.4f}"})

        # Compute final val metrics and reset states for the next epoch
        val_raw_results = self._compute_and_reset_metrics(self.val_metrics, prefix="Val_")
        
        if self.config.use_filtered_metrics:
            val_filtered_results = self._compute_and_reset_metrics(self.val_metrics_filtered, prefix="Val_Masked_")
            # Merge both dictionaries into one so Comet logs everything
            val_metric_results = {**val_raw_results, **val_filtered_results}
        else:
            val_metric_results = val_raw_results
        
        return total_loss / len(loader), val_metric_results

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]

    # ------------------------------------------------------------------
    # Metric Helpers
    # ------------------------------------------------------------------

    def _init_metrics(self) -> dict[str, nn.Module]:
        """Initialize and return a dictionary of required metric objects."""
        metrics = {}
        if self.config.use_cumuarea:
            metrics['iou'] = MeanIoU(num_classes=3, per_class=True, input_format='index', include_background=True).to(self.device)
            metrics['dice'] = DiceScore(num_classes=3, average="none", input_format="index", include_background=True).to(self.device)
        else:
            metrics['iou'] = BinaryJaccardIndex(threshold=self.binary_classification_threshold).to(self.device)
            metrics['dice'] = BinaryF1Score(threshold=self.binary_classification_threshold).to(self.device)
            metrics['precision'] = BinaryPrecision().to(self.device)
            metrics['recall'] = BinaryRecall().to(self.device)
            # metrics['auroc'] = BinaryAUROC().to(self.device)
            # metrics['aucpr'] = BinaryAveragePrecision().to(self.device)
        return metrics

    def _update_metrics(self, metrics_dict: dict[str, nn.Module], predictions: Tensor, masks: Tensor) -> None:
        """Process model predictions and update the provided metrics dictionary."""
        pred_probs = None
        
        if predictions.shape[1] == 1:
            pred_probs = torch.sigmoid(predictions).squeeze(1)
            pred_classes = (pred_probs > self.binary_classification_threshold).long()
        else:
            pred_classes = torch.argmax(predictions, dim=1)

        metrics_dict['iou'].update(pred_classes, masks)
        metrics_dict['dice'].update(pred_classes, masks)

        if not self.config.use_cumuarea and pred_probs is not None:
            metrics_dict['precision'].update(pred_classes, masks)
            metrics_dict['recall'].update(pred_classes, masks)
            # metrics_dict['auroc'].update(pred_probs, masks)
            # metrics_dict['aucpr'].update(pred_probs, masks)

    def _compute_and_reset_metrics(self, metrics_dict: dict[str, nn.Module], prefix: str = "") -> dict[str, float]:
        """Compute the final scores from the tracker, map them to a dict, and reset the states."""
        class_ious = metrics_dict['iou'].compute() 
        class_dices = metrics_dict['dice'].compute()

        final_metrics = {}

        if self.config.use_cumuarea:
            valid_iou_mask = class_ious != -1
            macro_iou = class_ious[valid_iou_mask].mean().item() if valid_iou_mask.any() else 0.0

            valid_dice_mask = class_dices != -1
            macro_dice = class_dices[valid_dice_mask].mean().item() if valid_dice_mask.any() else 0.0
            
            final_metrics.update({
                f"{prefix}Macro_IoU": macro_iou,
                f"{prefix}Macro_Dice": macro_dice,
                f"{prefix}IoU_0_Background": class_ious[0].item(),
                f"{prefix}IoU_1_Old_Fire": class_ious[1].item(),
                f"{prefix}IoU_2_New_Fire": class_ious[2].item(),
                f"{prefix}Dice_0_Background": class_dices[0].item(),
                f"{prefix}Dice_1_Old_Fire": class_dices[1].item(),
                f"{prefix}Dice_2_New_Fire": class_dices[2].item(),
            })
        else:
            final_metrics.update({
                f"{prefix}IoU_1_New_Fire": class_ious.item(),
                f"{prefix}Dice_1_New_Fire": class_dices.item(),
                f"{prefix}Precision": metrics_dict['precision'].compute().item(),
                f"{prefix}Recall": metrics_dict['recall'].compute().item(),
                # f"{prefix}AUC_ROC": metrics_dict['auroc'].compute().item(),
                # f"{prefix}AUC_PR": metrics_dict['aucpr'].compute().item(),
            })

        # Reset all metrics for the next pass
        for metric in metrics_dict.values():
            metric.reset()

        return final_metrics

    # ------------------------------------------------------------------
    # Environment Helpers
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