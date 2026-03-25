"""
Configuration dataclasses and loader from YAML files.
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    height: int = 256
    width: int = 256
    channels: int = 3
    num_samples: int = 1000
    num_classes: int = 1


@dataclass
class ModelConfig:
    input_channels: int = 3
    num_classes: int = 1
    hidden_features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_skip_connections: bool = True
    use_activation_after_upsampling: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a Config from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Populated Config instance.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        dataset_cfg = DatasetConfig(**raw.get("dataset", {}))
        model_cfg = ModelConfig(**raw.get("model", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))
        return cls(dataset=dataset_cfg, model=model_cfg, training=training_cfg)
