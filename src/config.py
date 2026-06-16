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
    channels: int = 20
    num_classes: int = 1

@dataclass
class ModelConfig:
    input_channels: int = 20
    num_classes: int = 1
    hidden_features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_skip_connections: bool = True
    use_activation_after_upsampling: bool = False
    architecture: str = "unet"
    utae_decoder_widths: list[int] | None = None
    utae_out_conv_channels: list[int] = field(default_factory=lambda: [32])
    utae_agg_mode: str = "att_group"
    utae_encoder_norm: str = "group"
    utae_n_head: int = 16
    utae_d_model: int = 256
    utae_d_k: int = 4
    utae_pad_value: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    use_cumuarea: bool = False
    use_cumuarea_prev: bool = True
    downscale: bool = False
    downscale_factor: int = 1
    smooth_mask: bool = True
    smooth_kernel: int = 3
    use_cyclical_aspect: bool = True
    use_filtered_metrics: bool = False


@dataclass
class CometConfig:
    enabled: bool = False
    project_name: str = ""
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
