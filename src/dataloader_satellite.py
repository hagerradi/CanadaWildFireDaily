from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
import os
import sys
import glob
import h5py
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
import gc
from datetime import datetime
import random

from src.config import Config
from src.dataloader_helper import create_stratified_splits, calculate_h5_statistics_filtered

# Add project root to sys.path to allow 'from configs import settings'
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from configs import settings

class H5FireSimpleDataset(Dataset):
    
    def __init__(self, h5_dir, id_list, dynamic_features, static_features=None, 
                 forecast_features=None, use_forecast=False,
                 fire_feature='firearea', patch_size=256, 
                 use_cumuarea_prev=False, use_cumuarea=False, transform=None,
                 normalize=False, stats_dict=None, sat_nodata=0, 
                 cloud_threshold=35.0,
                 max_sat_lookback_days=1,
                 is_train=False):
        
        self.h5_dir = h5_dir
        self.id_list = [str(fid) for fid in id_list] 
        self.dynamic_features = dynamic_features
        self.static_features = static_features if static_features else []
        self.fire_feature = fire_feature
        self.patch_size = patch_size
        
        # --- CCUMULATION FLAGS ---
        self.use_cumuarea_prev = use_cumuarea_prev  
        self.use_cumuarea = use_cumuarea            
        self.transform = transform

        # --- FORECAST PARAMS ---
        self.use_forecast = use_forecast
        self.forecast_features = forecast_features if forecast_features else ['tmax', 'prec', 'rh', 'ws']
        
        # --- NORMALIZATION PARAMS ---
        self.normalize = normalize
        self.stats_dict = stats_dict
        self.sat_nodata = sat_nodata
        
        # --- STORE CLOUD THRESHOLD ---
        self.cloud_threshold = cloud_threshold
        self.max_sat_lookback_days = max_sat_lookback_days

        self.is_train = is_train
        
        # Safety check
        if self.normalize and self.stats_dict is None:
            raise ValueError("If normalize=True, you must provide the stats_dict!")
        
        self.num_channels = len(self.dynamic_features) + len(self.static_features) + 1

        if self.use_forecast:
            self.num_channels += len(self.forecast_features)
        
        self.samples = [] 
        self.open_h5_handles = {}
        
        self._index_files()

    def __del__(self):
        """Clean up all open file handles."""
        if hasattr(self, 'open_h5_handles'):
            for f in self.open_h5_handles.values():
                try:
                    f.close()
                except:
                    pass

    def _index_files(self):
        """Scans H5 files, filtering by id_list and enforcing patch_size."""
        h5_files = glob.glob(os.path.join(self.h5_dir, "*.h5"))
        
        for file_path in tqdm(h5_files, desc="Indexing Fire Days"):
            
            # --- DYNAMIC ID FILTERING ---
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if base_name.replace('fire_', '') not in self.id_list:
                continue
            
            try:
                with h5py.File(file_path, 'r') as f:
                    fire_id = f.attrs.get('fire_id', 'unknown')
                    if 'days' not in f:
                        continue
                        
                    days_group = f['days']
                    day_keys = sorted(list(days_group.keys()))
                    if not day_keys:
                        continue
                    
                    # --- SIZE CHECK ---
                    first_day_grp = days_group[day_keys[0]]
                    sample_feat = list(first_day_grp['features'].keys())[0]
                    h, w = first_day_grp[f'features/{sample_feat}'].shape
                    
                    if h != self.patch_size or w != self.patch_size:
                        continue # Skip entirely if it doesn't match grid size
                        
                    # Pre-resolve paths
                    feature_paths, is_sat_flags = [], []
                    for feat in self.dynamic_features:
                        if f"features/{feat}" in first_day_grp:
                            feature_paths.append(f"features/{feat}")
                            is_sat_flags.append(False)
                        elif f"satellite/{feat}" in first_day_grp:
                            feature_paths.append(f"satellite/{feat}")
                            is_sat_flags.append(True)
                        else:
                            raise KeyError(f"Feature '{feat}' not found in {file_path}")
                    
                    # Use [:-1] to skip the very last day (since we predict T+1)
                    for i, day_key in enumerate(day_keys[:-1]):
                        
                        # ========================================================
                        # CLOUD COVER AND DATE FILTERING LOGIC
                        # ========================================================
                        day_group = days_group[day_key]
                        day_is_valid = True
                        
                        # Only run the check if we are strictly filtering (< 100 threshold)
                        if self.cloud_threshold < 100.0:
                            # Find the expected date
                            fire_datetime = day_group.attrs.get("noon_utc") or \
                                            day_group.attrs.get("start_utc") or \
                                            day_group.attrs.get("end_utc")
                            
                            expected_date = None
                            if fire_datetime:
                                if isinstance(fire_datetime, bytes):
                                    fire_datetime = fire_datetime.decode('utf-8')
                                expected_date = fire_datetime.split("T")[0]

                            # Check the stats
                            if "satellite" in day_group and "cloud_cover_stats" in day_group["satellite"].attrs:
                                stats_attr = day_group["satellite"].attrs["cloud_cover_stats"]
                                if isinstance(stats_attr, bytes):
                                    stats_attr = stats_attr.decode('utf-8')
                                
                                try:
                                    stats_list = json.loads(stats_attr)
                                    has_valid_sat = False
                                    
                                    for stat in stats_list:
                                        acq_date_str = stat.get("acquisition_date")
                                        cloud_pct = stat.get("cloud_cover_pct", float('inf'))
                                        
                                        if expected_date and acq_date_str:
                                            try:
                                                # Convert strings to datetime objects
                                                exp_dt = datetime.strptime(expected_date, "%Y-%m-%d")
                                                acq_dt = datetime.strptime(acq_date_str, "%Y-%m-%d")
                                                
                                                # Calculate difference in days
                                                day_diff = (exp_dt - acq_dt).days
                                                
                                                # Check if it's within the lookback window AND under the cloud threshold
                                                if 0 <= day_diff <= self.max_sat_lookback_days and cloud_pct <= self.cloud_threshold:
                                                    has_valid_sat = True
                                                    break
                                            except ValueError:
                                                pass
                                                
                                    # If no acquisitions met our criteria, the day is invalid
                                    if not has_valid_sat:
                                        day_is_valid = False
                                        
                                except json.JSONDecodeError:
                                    day_is_valid = False # Fail safe if unreadable
                            else:
                                day_is_valid = False # Fail safe if no stats exist
                        
                        # If it failed the cloud/date check, skip to the next day
                        if not day_is_valid:
                            continue
                        # ========================================================

                        fireday = days_group[day_key].attrs.get('fireday', i + 1)
                        next_day_key = day_keys[i + 1] # Get the target day
                        history_keys = day_keys[:i+1]  # Include current day in history
                        
                        self.samples.append({
                            'file_path': file_path,
                            'fire_id': fire_id,
                            'fireday': fireday,
                            'curr_day_key': day_key,
                            'next_day_key': next_day_key, # Store it
                            'history_keys': history_keys,
                            'feature_paths': feature_paths, 
                            'is_sat_flags': is_sat_flags
                        })
                        
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Lazy Loading Persistent File Handles
        file_path = s['file_path']
        if file_path not in self.open_h5_handles:
            self.open_h5_handles[file_path] = h5py.File(file_path, 'r', swmr=True)
            
        f = self.open_h5_handles[file_path]
        
        # Pre-allocate input tensor
        x_tensor = torch.empty((self.num_channels, self.patch_size, self.patch_size), dtype=torch.float32)
        
        curr_day_group = f[f"days/{s['curr_day_key']}"]
        channel_idx = 0

        # Load Dynamics & Satellite
        for path, is_sat in zip(s['feature_paths'], s['is_sat_flags']):
            
            feat_name = path.split('/')[-1]
            
            # Convert to float32 immediately
            arr_patch = curr_day_group[path][:].astype(np.float32)
            
            if is_sat:
                arr_patch = np.ascontiguousarray(np.flipud(arr_patch))
                
                # Compute valid mask
                if np.isnan(self.sat_nodata):
                    valid_mask = ~np.isnan(arr_patch)
                else:
                    valid_mask = (arr_patch != self.sat_nodata)
                
                # Apply sentinel-2 scaling factor (0 to 1 range)
                arr_patch = arr_patch * 0.0001
                
            else:
                valid_mask = ~np.isnan(arr_patch)
            
            # --- NORMALIZATION ---
            # Check "is_sat" so satellite data is not z-scored
            if self.normalize and feat_name in self.stats_dict and feat_name != self.fire_feature and not is_sat:
                mean = self.stats_dict[feat_name]['mean']
                std = self.stats_dict[feat_name]['std']
                
                # Apply Z-score only to the valid pixels of dynamic variables (weather, etc.)
                arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std
                
            # Safely zero out any NaNs/nodata values for BOTH satellite and dynamic features
            arr_patch[~valid_mask] = 0.0
            # ---------------------------
                
            x_tensor[channel_idx] = torch.from_numpy(arr_patch)
            channel_idx += 1
            
        # Load Statics
        if self.static_features and 'static_features' in f:
            static_group = f['static_features']
            for feature in self.static_features:
                arr_patch = static_group[feature][:].astype(np.float32)
                
                # --- NORMALIZATION LOGIC ---
                if self.normalize and feature in self.stats_dict and feature != self.fire_feature:
                    valid_mask = ~np.isnan(arr_patch)
                    mean = self.stats_dict[feature]['mean']
                    std = self.stats_dict[feature]['std']
                    
                    # Apply Z-score only to the valid pixels. NaNs remain NaNs!
                    arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std
                # ---------------------------
                
                x_tensor[channel_idx] = torch.from_numpy(arr_patch)
                channel_idx += 1
        
        # ---------------------------------------------------------
        # Process Forecasts (Tomorrow's Weather)
        # ---------------------------------------------------------
        if self.use_forecast:
            # Point to tomorrow's H5 group
            next_day_group = f[f"days/{s['next_day_key']}"]
            
            for feat in self.forecast_features:
                # Grab tomorrow's weather variables
                arr_patch = next_day_group[f"features/{feat}"][:].astype(np.float32)
                valid_mask = ~np.isnan(arr_patch)
                
                # Normalize using the SAME stats dictionary
                if self.normalize and feat in self.stats_dict:
                    mean = self.stats_dict[feat]['mean']
                    std = self.stats_dict[feat]['std']
                    arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std
                    
                x_tensor[channel_idx] = torch.from_numpy(arr_patch)
                channel_idx += 1

        # ---------------------------------------------------------
        # Process Fire Masks
        # ---------------------------------------------------------

        # --- Determine CURRENT mask (Input Channel) ---
        # The input is the fire up to Day T
        curr_fire_mask = torch.zeros((self.patch_size, self.patch_size), dtype=torch.bool)
        
        if self.use_cumuarea_prev:
            for h_key in s['history_keys']:
                h_arr = f[f"days/{h_key}/features/{self.fire_feature}"][:]
                curr_fire_mask |= (torch.from_numpy(h_arr) > 0)
        else:
            curr_arr = curr_day_group[f"features/{self.fire_feature}"][:]
            curr_fire_mask = (torch.from_numpy(curr_arr) > 0)
                
        x_tensor[-1] = curr_fire_mask.float()


        # --- Determine NEXT DAY mask (Output Label) ---
        # The target is Day T+1
        next_day_group = f[f"days/{s['next_day_key']}"]
        raw_next_arr = next_day_group[f"features/{self.fire_feature}"][:]
        raw_next_mask = (torch.from_numpy(raw_next_arr) > 0)
        
        y_tensor = torch.zeros((self.patch_size, self.patch_size), dtype=torch.float32)

        # Find the strictly new growth that happens during T+1
        new_growth_mask = raw_next_mask & ~curr_fire_mask

        # --- FLEXIBLE OUTPUT LABELING ---
        if self.use_cumuarea:
             # 3-Class Approach: 0 = Background, 1 = Existing Fire, 2 = New Fire
             y_tensor[curr_fire_mask] = 1.0 
             y_tensor[new_growth_mask] = 2.0
        else:
             # Binary Approach targeting ONLY new growth.
             # Leave the existing fire pixels as 0.0 (background)
             y_tensor[new_growth_mask] = 1.0
        
        # ==========================================
        # DATA AUGMENTATION
        # ==========================================
        if self.is_train:
            # Random Horizontal Flip (50% chance)
            if random.random() > 0.5:
                x_tensor = x_tensor.flip(-1)
                y_tensor = y_tensor.flip(-1)
                
            # Random Vertical Flip (50% chance)
            if random.random() > 0.5:
                x_tensor = x_tensor.flip(-2)
                y_tensor = y_tensor.flip(-2)

            # THE FIX: Force memory to be contiguous before returning
            x_tensor = x_tensor.contiguous()
            y_tensor = y_tensor.contiguous()
        # ==========================================

        if self.transform:
            x_tensor = self.transform(x_tensor)

        # meta = {
        #     "fire_id": s['fire_id'],
        #     "fireday": s['fireday'],
        #     "curr_day_key": s['curr_day_key']
        # }

        return x_tensor, y_tensor
    
def get_dataloaders(config: Config, cloud_threshold: float = 35.0, max_sat_lookback_days: int = 1) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    """Build train, validation, and test DataLoaders.

    Args:
        config: Global configuration object.
        cloud_threshold: Maximum allowed cloud cover percentage for satellite features.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    tc = config.training

    # 6 Weather/Indices + 6 Satellite Bands = 12 Dynamic Features
    dynamic_feats = [
        'tmax', 'rh', 'ws', 'prec', 'evi', 'ndvi', 
        's2_B02', 's2_B03', 's2_B04', 's2_B08', 's2_B11', 's2_B12'
    ]
    # 7 Static Features
    static_feats = [
        'dem_avg', 'slope', 'aspect', 'biomass', 'closure', 'prcc', 'prcb'
    ]

    # # Extract the actual IDs
    # valid_ids = np.load(f'{settings.METADATA_FOLDER}/fire_ids_256_5_years.npy').tolist()
    
    # # Parse the text file to identify problematic Fire IDs
    # report_path = f'{settings.METADATA_FOLDER}/diagnosis_report_full_features_256.txt'
    # to_remove = set()
    # current_fire_id = None

    # with open(report_path, 'r') as f:
    #     for line in f:
    #         line = line.strip()
            
    #         # Track the current Fire ID block
    #         if line.startswith("Fire ID:"):
    #             current_fire_id = line.replace("Fire ID:", "").strip()
                
    #         # If the line is an error flag under a Fire ID, mark it as problematic
    #         elif line.startswith("-") and current_fire_id:
    #             to_remove.add(current_fire_id)

    # valid_ids = [fid for fid in valid_ids if fid not in to_remove]
    # print(f'Valid ids after removing the problematic fires {len(valid_ids)}')

    # Dataframes
    path_2020 = f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_2020/Firegrowth_pts_v1_1_2020.csv'
    path_2021 = f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_2021/Firegrowth_pts_v1_1_2021.csv'
    path_2022 = f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_2022/Firegrowth_pts_v1_1_2022.csv'
    path_2023 = f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_2023/Firegrowth_pts_v1_1_2023.csv'
    path_2024 = f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_2024/Firegrowth_pts_v1_1_2024.csv'

    print('Reading dataframes.')
    columns_to_use = [col for col in settings.SUBSET_FEATURE_LIST if col not in ['easting', 'northing']]
    df_2020 = pd.read_csv(path_2020, usecols=columns_to_use)
    df_2021 = pd.read_csv(path_2021, usecols=columns_to_use)
    df_2022 = pd.read_csv(path_2022, usecols=columns_to_use)
    df_2023 = pd.read_csv(path_2023, usecols=columns_to_use)
    df_2024 = pd.read_csv(path_2024, usecols=columns_to_use)

    # Concatenate
    fire_growth_combined = pd.concat([df_2020, df_2021, df_2022, df_2023, df_2024], ignore_index=True)
    # Extract the IDS
    valid_ids = fire_growth_combined['ID'].unique()

    overall_dataset = H5FireSimpleDataset(
        h5_dir=settings.H5_OUTPUT_FOLDER, 
        id_list=valid_ids,
        dynamic_features=dynamic_feats,
        static_features=static_feats,
        fire_feature="cumuarea",
        patch_size=settings.GRID_SIZE,
        use_cumuarea=tc.use_cumuarea,
        use_cumuarea_prev=tc.use_cumuarea_prev,
        normalize=False, 
        stats_dict=None, 
        sat_nodata=np.nan,
        cloud_threshold=cloud_threshold,
        max_sat_lookback_days=max_sat_lookback_days
    )
    print(f'Number of clean samples : {len(overall_dataset)}')
    retained_ids = sorted(list({s['fire_id'] for s in overall_dataset.samples}))
    print(f'Number of clean fires {len(retained_ids)}')

    fire_growth_filtered = fire_growth_combined[fire_growth_combined['ID'].astype(str).isin(retained_ids)]
    print('Read dataframes.')

    train_ids, val_ids, test_ids = create_stratified_splits(fire_growth_filtered, 
                                                            train_split=tc.train_split,
                                                            random_state=42)
    print(len(train_ids), len(val_ids), len(test_ids))

    # ==========================================
    # RAM CLEANUP
    # ==========================================
    del df_2020
    del df_2021
    del df_2022
    del df_2023
    del df_2024
    del fire_growth_combined
    del fire_growth_filtered
    gc.collect() 
    print("DataFrames deleted to free up RAM.")
    # ==========================================

    stats_dict = calculate_h5_statistics_filtered(settings.H5_OUTPUT_FOLDER, 
                                                  train_ids, 
                                                  cloud_threshold=cloud_threshold,
                                                  sat_nodata=np.nan,
                                                  max_sat_lookback_days=max_sat_lookback_days)

    # Pass cloud_threshold to all three datasets
    train_dataset = H5FireSimpleDataset(
        h5_dir=settings.H5_OUTPUT_FOLDER, 
        id_list=train_ids,
        dynamic_features=dynamic_feats,
        static_features=static_feats,
        use_forecast=False,
        # forecast_features=['tmax', 'prec', 'rh', 'ws'],
        fire_feature="cumuarea",
        patch_size=settings.GRID_SIZE,
        use_cumuarea=tc.use_cumuarea,
        use_cumuarea_prev=tc.use_cumuarea_prev,
        normalize=True, 
        stats_dict=stats_dict, 
        sat_nodata=np.nan,
        cloud_threshold=cloud_threshold,
        max_sat_lookback_days=max_sat_lookback_days,
        is_train=True
    )

    val_dataset = H5FireSimpleDataset(
        h5_dir=settings.H5_OUTPUT_FOLDER, 
        id_list=val_ids,
        dynamic_features=dynamic_feats,
        static_features=static_feats,
        use_forecast=False,
        # forecast_features=['tmax', 'prec', 'rh', 'ws'],
        fire_feature="cumuarea",
        patch_size=settings.GRID_SIZE,
        use_cumuarea=tc.use_cumuarea,
        use_cumuarea_prev=tc.use_cumuarea_prev,
        normalize=True, 
        stats_dict=stats_dict, 
        sat_nodata=np.nan,
        cloud_threshold=cloud_threshold,
        max_sat_lookback_days=max_sat_lookback_days,
        is_train=False  
    )

    test_dataset = H5FireSimpleDataset(
        h5_dir=settings.H5_OUTPUT_FOLDER, 
        id_list=test_ids,
        dynamic_features=dynamic_feats,
        static_features=static_feats,
        use_forecast=False,
        # forecast_features=['tmax', 'prec', 'rh', 'ws'],
        fire_feature="cumuarea",
        patch_size=settings.GRID_SIZE,
        use_cumuarea=tc.use_cumuarea,
        use_cumuarea_prev=tc.use_cumuarea_prev,
        normalize=True, 
        stats_dict=stats_dict, 
        sat_nodata=np.nan,
        cloud_threshold=cloud_threshold,
        max_sat_lookback_days=max_sat_lookback_days,
        is_train=False  
    )
    
    print(f'Number of training samples :: {len(train_dataset)}')
    print(f'Number of test samples :: {len(test_dataset)}')
    print(f'Number of validation samples :: {len(val_dataset)}')

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