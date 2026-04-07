"""
Training entry point.

Usage:
    python -m src.train
    python -m src.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import time

from src.config import Config
# from src.dataloader import get_dataloaders
from src.dataloader_satellite import get_dataloaders
from src.logger import CometLogger
from src.models.unet import UNet
from src.trainer import Trainer
from src.utils import seed_everything
from src.focal_loss import FocalLoss 
from src.dice_loss import DiceLoss

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super(ComboLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        
        # Add smooth=1.0 back to prevent gradient death on empty masks
        self.dice = DiceLoss(smooth=1.0, apply_sigmoid=True)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
            
        focal_l = self.focal(inputs, targets)
        dice_l = self.dice(inputs, targets)
        
        return (self.focal_weight * focal_l) + (self.dice_weight * dice_l)

def train(config: Config) -> None:
    """Set up all components and run the training loop.

    Args:
        config: Global configuration object.
    """
    # Fix all random seeds (CPU, CUDA, CuDNN, Python)
    seed_everything(config.seed)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(config, cloud_threshold=40)

    # Model
    mc = config.model
    model = UNet(
        input_channels=mc.input_channels,
        num_classes=mc.num_classes,
        hidden_features=mc.hidden_features,
        use_skip_connections=mc.use_skip_connections,
        use_activation_after_upsampling=mc.use_activation_after_upsampling,
    )

    # Optimizer & loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Initialize the Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', 
        factor=0.5,
        patience=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.training.use_cumuarea:
        # --- 3-CLASS SETUP ---
        print("Initializing 3-Class CrossEntropy Loss...")
        weights = torch.tensor([1.0, 10.0, 50.0], dtype=torch.float32).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        # --- 2-CLASS SETUP (Using Custom Focal + Dice) ---
        # 0 = Background + Old Fire, 1 = New Fire
        print("Initializing 2-Class Combo Loss (Focal + Dice)...")
        # Higher gamma = harder focus on difficult pixels.
        loss_fn = ComboLoss(alpha=0.75, gamma=2.0, focal_weight=1.0, dice_weight=1.0).to(device)

    # Optional Comet logger
    logger: CometLogger | None = None
    if config.comet.enabled:
        cc = config.comet

        # Auto-generates names like: unet-baseline-1704124800
        run_name = f"{cc.experiment_name}-{int(time.time())}"

        logger = CometLogger(
            project_name=cc.project_name,
            workspace=cc.workspace,
            experiment_name=run_name,
            experiment_tags=cc.experiment_tags or None,
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer, 
        scheduler=scheduler,
        loss_fn=loss_fn, 
        config=config.training, 
        logger=logger
    )

    print(f"Training on device: {trainer.device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    trainer.fit(train_loader, val_loader)

    return trainer, test_loader