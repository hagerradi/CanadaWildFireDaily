import os
import torch
from torch.utils.data import Dataset, DataLoader
from configs import settings
from src.config import Config
from skimage import morphology

class TimeSeriesFireDataset(Dataset):
    def __init__(self, data_dir: str):
        """
            data_dir: Path to the specific split folder (e.g., settings.TIMESERIES_SAMPLE_FOLDER + "/train")
        """
        self.data_dir = data_dir
        
        # Sort files to ensure consistent, reproducible ordering across runs
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        
        # Load the pre-computed tensor dictionary from the hard drive
        data = torch.load(file_path, weights_only=True)

        # APPLYING MAX_SIZE=1 HOLE FILLING TO THE MASK
        y_tensor = data['y']
        # Convert to boolean numpy array to find where ANY fire exists
        y_np = y_tensor.numpy() > 0 
        # Fill the 1-pixel projection gaps safely (projection noise)
        y_filled_np = morphology.remove_small_holes(y_np, area_threshold=1)
        # Find ONLY the pixels that were newly filled (False in original, True in filled)
        newly_filled_mask = y_filled_np & ~y_np
        # Safely add the new pixels as class 1.0 without destroying existing 2.0s
        y_tensor[newly_filled_mask] = 1.0
        data['y'] = y_tensor
        
        return data['x'], data['y']


def get_timeseries_dataloaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Instantiates the offline time-series datasets and wraps them in PyTorch DataLoaders.

    Args:
      config: Config: config parameters.

    Returns: train, validation, and test loaders

    """
    tc = config.training

    base_data_path = settings.TIMESERIES_SAMPLE_FOLDER

    # Define the paths to your precomputed splits
    train_dir = os.path.join(base_data_path, "train")
    val_dir = os.path.join(base_data_path, "val")
    test_dir = os.path.join(base_data_path, "test")

    # Instantiate the datasets
    train_dataset = TimeSeriesFireDataset(train_dir)
    val_dataset = TimeSeriesFireDataset(val_dir)
    test_dataset = TimeSeriesFireDataset(test_dir)

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