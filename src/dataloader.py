import os
import torch
from torch.utils.data import Dataset, DataLoader
from configs import settings
from src.config import Config
from skimage import morphology

class DailyFireDataset(Dataset):
    def __init__(self, data_dir: str, return_sat_age: bool = False):
        """
        data_dir: Path to the specific split folder (e.g., settings.SAMPLE_FOLDER + "/train")
        return_sat_age: Toggle to return the delta_t float alongside x and y.
        """
        self.data_dir = data_dir
        self.return_sat_age = return_sat_age
        
        # Sort files to ensure consistent, reproducible ordering across runs
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        
        # Load the pre-computed tensor dictionary from the hard drive
        data = torch.load(file_path, weights_only=True)

        # =========================================================
        # APPYING MAX_SIZE=1 HOLE FILLING TO THE MASK (remove noise pixels from projection)
        # =========================================================
        y_tensor = data['y']
        
        # Convert to boolean numpy array
        y_np = y_tensor.numpy() > 0 
        
        # Fill the 1-pixel projection gaps safely
        y_filled_np = morphology.remove_small_holes(y_np, area_threshold=1)
        
        # Convert back to PyTorch tensor and match the original data type
        data['y'] = torch.from_numpy(y_filled_np).to(y_tensor.dtype)
        # =========================================================
        
        if self.return_sat_age:
            return data['x'], data['y'], data['delta_t']
        else:
            return data['x'], data['y']


def get_dataloaders(config: Config, is_sat_age: bool) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Instantiates the offline datasets and wraps them in PyTorch DataLoaders.
    """
    tc = config.training
    
    base_data_path = settings.SAMPLE_FOLDER

    # Define the paths to your precomputed splits
    train_dir = os.path.join(base_data_path, "train")
    val_dir = os.path.join(base_data_path, "val")
    test_dir = os.path.join(base_data_path, "test")

    # Instantiate the datasets, passing the sat_age flag
    train_dataset = DailyFireDataset(train_dir, return_sat_age=is_sat_age)
    val_dataset = DailyFireDataset(val_dir, return_sat_age=is_sat_age)
    test_dataset = DailyFireDataset(test_dir, return_sat_age=is_sat_age)

    print(f'Number of offline training samples   :: {len(train_dataset)}')
    print(f'Number of offline validation samples :: {len(val_dataset)}')
    print(f'Number of offline test samples       :: {len(test_dataset)}')

    # 3. Wrap them in DataLoaders
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