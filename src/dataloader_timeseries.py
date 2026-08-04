import os
import torch
from torch.utils.data import Dataset, DataLoader
from configs import settings
from src.config import Config
from skimage import morphology
import numpy as np

class TimeSeriesFireDataset(Dataset):
    def __init__(self, data_dir: str, return_positions: bool = False):
        """
            data_dir: Path to the specific split folder (e.g., settings.TIMESERIES_SAMPLE_FOLDER + "/train")
        """
        self.data_dir = data_dir
        self.return_positions = return_positions
        
        # Sort files to ensure consistent, reproducible ordering across runs
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        
        file_path = os.path.join(self.data_dir, self.files[idx])

        with np.load(file_path) as data:
            x_np = data['x']
            y_np = data['y']
            positions_np = data['positions']

        # APPYING MAX_SIZE=1 HOLE FILLING TO THE MASK (remove noise pixels from projection)
        y_bool = y_np > 0 
        # Fill the 1-pixel projection gaps
        y_filled_np = morphology.remove_small_holes(y_bool, area_threshold=1)
        # Convert back to its original integer type (uint8)
        y_filled_np = y_filled_np.astype(y_np.dtype)

        # Convert everything to PyTorch Tensors
        x_tensor = torch.from_numpy(x_np).float() 
        y_tensor = torch.from_numpy(y_filled_np).float()
        positions_tensor = torch.from_numpy(positions_np).float()

        if self.return_positions:
            return x_tensor, y_tensor, positions_tensor
        else:
            return x_tensor, y_tensor


def get_timeseries_dataloaders(
    config: Config,
    return_positions: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Instantiates the offline time-series datasets and wraps them in PyTorch DataLoaders.

        return data['x'], data['y']

    """
    tc = config.training

    base_data_path = settings.TIMESERIES_SAMPLE_FOLDER

    # Define the paths to your precomputed splits
    train_dir = os.path.join(base_data_path, "train")
    val_dir = os.path.join(base_data_path, "val")
    test_dir = os.path.join(base_data_path, "test")

    # Instantiate the datasets
    train_dataset = TimeSeriesFireDataset(train_dir, return_positions=return_positions)
    val_dataset = TimeSeriesFireDataset(val_dir, return_positions=return_positions)
    test_dataset = TimeSeriesFireDataset(test_dir, return_positions=return_positions)

    print(f'Number of offline TS training samples   :: {len(train_dataset)}')
    print(f'Number of offline TS validation samples :: {len(val_dataset)}')
    print(f'Number of offline TS test samples       :: {len(test_dataset)}')

    # Wrap them in DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=tc.batch_size, 
        shuffle=True, 
        num_workers=tc.num_workers,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=tc.batch_size, 
        shuffle=False, 
        num_workers=tc.num_workers,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=tc.batch_size, 
        shuffle=False,
        num_workers=tc.num_workers,
        persistent_workers=False   
    )

    return train_loader, val_loader, test_loader
