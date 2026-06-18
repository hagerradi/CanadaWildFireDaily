import os
import torch
from torch.utils.data import Dataset, DataLoader
from configs import settings
from src.config import Config
import numpy as np

class DailyFireDataset(Dataset):
    def __init__(self, data_dir: str, return_sat_age: bool = False, cache_in_memory: bool = False):
        """
            data_dir: Path to the specific split folder (e.g., settings.SAMPLE_FOLDER + "/train")
            cache_in_memory: When True, all samples are loaded and preprocessed once at init
                time and stored in RAM.  Subsequent __getitem__ calls are pure tensor indexing
                with no file I/O or decompression overhead.  Requires enough free RAM to hold
                the entire split (~5 MB per 256×256×20-channel float32 sample).
        """
        
        self.data_dir = data_dir
        self.return_sat_age = return_sat_age
        
        # Sort files to ensure consistent, reproducible ordering across runs
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

        self._cache: list | None = None
        if cache_in_memory:
            self._cache = [self._load(i) for i in range(len(self.files))]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, idx: int):
        """Load, hole-fill, and convert one sample to tensors."""
        file_path = os.path.join(self.data_dir, self.files[idx])

        with np.load(file_path) as data:
            x_np = data['x']
            y_np = data['y']
            delta_t_np = data['delta_t']

        # APPLY MAX_SIZE=1 HOLE FILLING TO THE MASK (remove noise pixels from projection)
        # Equivalent to skimage.morphology.remove_small_holes(area_threshold=1):
        # fills isolated False pixels whose all 4-connected neighbours are True.
        y_bool = y_np.astype(bool)
        padded = np.pad(y_bool, 1, constant_values=False)
        has_false_4neighbor = (
            ~padded[:-2, 1:-1] | ~padded[2:, 1:-1] |
            ~padded[1:-1, :-2] | ~padded[1:-1, 2:]
        )
        y_filled_np = (y_bool | ~has_false_4neighbor).astype(y_np.dtype)

        x_tensor = torch.from_numpy(x_np).float()
        y_tensor = torch.from_numpy(y_filled_np).float()
        delta_t_tensor = torch.from_numpy(delta_t_np).float()

        if self.return_sat_age:
            return x_tensor, y_tensor, delta_t_tensor
        return x_tensor, y_tensor

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        return self._load(idx)


def get_dataloaders(config: Config, is_sat_age: bool) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Instantiates the offline datasets and wraps them in PyTorch DataLoaders.

    Args:
      config: Config: config parameters
      is_sat_age: bool: toggle to choose if the satellite age gap is included in the sample

    Returns: the train, val, and test loaders

    """
    tc = config.training
    
    base_data_path = settings.SAMPLE_FOLDER

    # Define the paths to the precomputed splits
    train_dir = os.path.join(base_data_path, "train")
    val_dir = os.path.join(base_data_path, "val")
    test_dir = os.path.join(base_data_path, "test")

    cache = getattr(tc, 'cache_in_memory', False)

    # Instantiate the datasets, passing the sat_age flag
    train_dataset = DailyFireDataset(train_dir, return_sat_age=is_sat_age, cache_in_memory=cache)
    val_dataset = DailyFireDataset(val_dir, return_sat_age=is_sat_age, cache_in_memory=cache)
    test_dataset = DailyFireDataset(test_dir, return_sat_age=is_sat_age, cache_in_memory=cache)

    print(f'Number of offline training samples   :: {len(train_dataset)}')
    print(f'Number of offline validation samples :: {len(val_dataset)}')
    print(f'Number of offline test samples       :: {len(test_dataset)}')

    # Wrap them in DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=tc.batch_size, 
        shuffle=True, 
        num_workers=tc.num_workers,
        persistent_workers=tc.num_workers > 0,
        pin_memory=True,
        prefetch_factor=4 if tc.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=tc.batch_size, 
        shuffle=False, 
        num_workers=tc.num_workers,
        persistent_workers=tc.num_workers > 0,
        pin_memory=True,
        prefetch_factor=4 if tc.num_workers > 0 else None,
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=tc.batch_size, 
        shuffle=False,
        num_workers=tc.num_workers,
        persistent_workers=tc.num_workers > 0,
        pin_memory=True,
        prefetch_factor=4 if tc.num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader