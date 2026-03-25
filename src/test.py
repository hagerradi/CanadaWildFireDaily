"""
Evaluation entry point.

Loads a trained model checkpoint and evaluates it on the held-out test split.

Usage:
    python -m src.test --checkpoint checkpoints/checkpoint_epoch_010.pt
    python -m src.test --checkpoint checkpoints/checkpoint_epoch_010.pt --config configs/default.yaml
"""
from __future__ import annotations

import argparse

import torch

from src.config import Config
from src.dataloader import get_dataloaders
from src.models.unet import UNet
from src.trainer import Trainer


def test(config: Config, checkpoint_path: str) -> float:
    """Load a checkpoint and evaluate on the test split.

    Args:
        config: Global configuration object.
        checkpoint_path: Path to the model checkpoint file.

    Returns:
        Average test loss.
    """
    _, _, test_loader = get_dataloaders(config)

    mc = config.model
    model = UNet(
        input_channels=mc.input_channels,
        num_classes=mc.num_classes,
        hidden_features=mc.hidden_features,
        use_skip_connections=mc.use_skip_connections,
        use_activation_after_upsampling=mc.use_activation_after_upsampling,
    )

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, config=config.training)
    epoch = trainer.load_checkpoint(checkpoint_path)

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Evaluating on device: {trainer.device}")

    test_loss = trainer.validate(test_loader)
    print(f"Test loss: {test_loss:.4f}")
    return test_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained wildfire segmentation model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    test(config, args.checkpoint)


if __name__ == "__main__":
    main()
