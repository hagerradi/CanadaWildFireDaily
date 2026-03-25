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
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10


@dataclass
class CometConfig:
    enabled: bool = False
    project_name: str = "mila-wildfires"
    workspace: str = ""
    experiment_name: str = "default-run"
    experiment_tags: list[str] = field(default_factory=list)


@dataclass
class Config:
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    comet: CometConfig = field(default_factory=CometConfig)

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

        seed = raw.get("seed", 42)
        dataset_cfg = DatasetConfig(**raw.get("dataset", {}))
        model_cfg = ModelConfig(**raw.get("model", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))
        comet_cfg = CometConfig(**raw.get("comet", {}))
        return cls(seed=seed, dataset=dataset_cfg, model=model_cfg, training=training_cfg, comet=comet_cfg)
