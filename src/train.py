"""
Training entry point.

Usage:
    python -m src.train
    python -m src.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import random

import torch

from src.config import Config
from src.dataloader import get_dataloaders
from src.models.unet import UNet
from src.trainer import Trainer


def train(config: Config) -> None:
    """Set up all components and run the training loop.

    Args:
        config: Global configuration object.
    """
    # Reproducibility
    seed = config.training.seed
    random.seed(seed)
    torch.manual_seed(seed)

    # Data
    train_loader, val_loader, _ = get_dataloaders(config)

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
    loss_fn = torch.nn.MSELoss()

    # Trainer
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, config=config.training)

    print(f"Training on device: {trainer.device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    trainer.fit(train_loader, val_loader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the wildfire segmentation model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    train(config)


if __name__ == "__main__":
    main()
