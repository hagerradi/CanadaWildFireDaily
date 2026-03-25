"""
Toy image dataset and DataLoader factory.

The dataset generates random images of shape (C, H, W) paired with binary
segmentation masks of shape (1, H, W).  This is useful for verifying model
architectures and training pipelines before switching to a real dataset.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from src.config import Config


class ToyImageDataset(Dataset):
    """In-memory dataset of randomly generated images and binary masks.

    Args:
        num_samples: Total number of (image, mask) pairs to generate.
        height: Spatial height H of each image.
        width: Spatial width W of each image.
        channels: Number of input channels C.
        num_classes: Number of output segmentation classes.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        height: int = 256,
        width: int = 256,
        channels: int = 3,
        num_classes: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.images: Tensor = torch.randn(num_samples, channels, height, width, generator=generator)
        self.masks: Tensor = torch.randint(0, num_classes + 1, (num_samples, 1, height, width), generator=generator).float()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.images[idx], self.masks[idx]


def get_dataloaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders from the toy dataset.

    Args:
        config: Global configuration object.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    dc = config.dataset
    tc = config.training

    dataset = ToyImageDataset(
        num_samples=dc.num_samples,
        height=dc.height,
        width=dc.width,
        channels=dc.channels,
        num_classes=dc.num_classes,
        seed=tc.seed,
    )

    n_total = len(dataset)
    n_train = int(n_total * tc.train_split)
    n_val = int(n_total * tc.val_split)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(tc.seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_set, batch_size=tc.batch_size, shuffle=True, num_workers=tc.num_workers)
    val_loader = DataLoader(val_set, batch_size=tc.batch_size, shuffle=False, num_workers=tc.num_workers)
    test_loader = DataLoader(test_set, batch_size=tc.batch_size, shuffle=False, num_workers=tc.num_workers)

    return train_loader, val_loader, test_loader
