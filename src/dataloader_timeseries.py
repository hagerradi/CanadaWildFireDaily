from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
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
from src.dataloader_helper import create_stratified_splits, calculate_h5_statistics_filtered, smooth_fire_mask

# # Add project root to sys.path
# root = Path(__file__).resolve().parent.parent.parent
# if str(root) not in sys.path:
#     sys.path.append(str(root))

from configs import settings

class H5FireTimeSeriesDataset(Dataset):
    
    def __init__(self, h5_dir, id_list, dynamic_features, static_features=None, 
                 forecast_features=None, use_forecast=False,
                 fire_feature='firearea', patch_size=256, 
                 use_cumuarea_prev=False, use_cumuarea=False, transform=None,
                 normalize=False, stats_dict=None, sat_nodata=0, 
                 cloud_threshold=35.0,
                 max_sat_lookback_days=10, 
                 sequence_length=3,
                 downscale=False,
                 downscale_factor=2,
                 smooth_mask=False,
                 smooth_kernel=3,
                 use_cyclical_aspect=False):
        
        self.h5_dir = h5_dir
        self.id_list = [str(fid) for fid in id_list] 
        self.dynamic_features = dynamic_features
        self.static_features = static_features if static_features else []
        self.fire_feature = fire_feature
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        
        self.use_cumuarea_prev = use_cumuarea_prev  
        self.use_cumuarea = use_cumuarea            
        self.transform = transform

        self.use_forecast = use_forecast
        self.forecast_features = forecast_features if forecast_features else ['tmax', 'prec', 'rh', 'ws']
        
        self.normalize = normalize
        self.stats_dict = stats_dict
        self.sat_nodata = sat_nodata
        self.cloud_threshold = cloud_threshold
        self.max_sat_lookback_days = max_sat_lookback_days
        self.downscale = downscale
        self.downscale_factor = downscale_factor
        self.smooth_mask = smooth_mask
        self.smooth_kernel = smooth_kernel
        self.use_cyclical_aspect = use_cyclical_aspect
        
        if self.normalize and self.stats_dict is None:
            raise ValueError("If normalize=True, you must provide the stats_dict!")
        
        # Base channels: Dynamic + Static + 2 Fire Masks (curr_fire_mask & fireday_grid)
        base_channels = len(self.dynamic_features) + len(self.static_features) + 2
        
        # If we encode aspect, 1 channel becomes 2 (sin, cos), so we add 1 net channel
        if self.use_cyclical_aspect and 'aspect' in self.static_features:
            base_channels += 1
            
        self.num_channels = base_channels

        if self.use_forecast:
            self.num_channels += len(self.forecast_features)
        # -----------------------------------
        
        self.samples = [] 
        self.open_h5_handles = {}
        
        self._index_files()

    def __del__(self):
        if hasattr(self, 'open_h5_handles'):
            for f in self.open_h5_handles.values():
                try:
                    f.close()
                except:
                    pass

    def _get_expected_date(self, day_group) -> str | None:
        """Helper to extract the string date from a day group."""
        fire_datetime = day_group.attrs.get("noon_utc") or day_group.attrs.get("start_utc") or day_group.attrs.get("end_utc")
        if fire_datetime:
            if isinstance(fire_datetime, bytes):
                fire_datetime = fire_datetime.decode('utf-8')
            return fire_datetime.split("T")[0]
        return None

    def _is_day_cloud_free(self, day_group, expected_date_str) -> bool:
        """Checks if the satellite image stored in this day meets cloud and age criteria."""
        if self.cloud_threshold >= 100.0:
            return True
            
        if "satellite" not in day_group or "cloud_cover_stats" not in day_group["satellite"].attrs:
            return False
            
        stats_attr = day_group["satellite"].attrs["cloud_cover_stats"]
        if isinstance(stats_attr, bytes):
            stats_attr = stats_attr.decode('utf-8')
            
        try:
            stats_list = json.loads(stats_attr)
            for stat in stats_list:
                acq_date_str = stat.get("acquisition_date")
                cloud_pct = stat.get("cloud_cover_pct", float('inf'))
                
                if expected_date_str and acq_date_str:
                    try:
                        exp_dt = datetime.strptime(expected_date_str, "%Y-%m-%d")
                        acq_dt = datetime.strptime(acq_date_str, "%Y-%m-%d")
                        
                        # Calculate exactly how old the stored satellite image is
                        gap_days = (exp_dt - acq_dt).days
                        
                        # Accept it if it's recent enough AND under the cloud threshold
                        if 0 <= gap_days <= self.max_sat_lookback_days and cloud_pct <= self.cloud_threshold:
                            return True
                            
                    except ValueError:
                        pass
        except json.JSONDecodeError:
            pass
            
        return False

    def _index_files(self):
        """Scans H5 files. Anchors the satellite validity strictly on Day T."""
        h5_files = glob.glob(os.path.join(self.h5_dir, "*.h5"))
        
        for file_path in tqdm(h5_files, desc="Indexing Time-Series"):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            if base_name.replace('fire_', '') not in self.id_list:
                continue
            
            try:
                with h5py.File(file_path, 'r') as f:
                    fire_id = f.attrs.get('fire_id', 'unknown')
                    if isinstance(fire_id, bytes):
                        fire_id = fire_id.decode('utf-8')
                        
                    if 'days' not in f:
                        continue
                        
                    days_group = f['days']
                    day_keys = sorted(list(days_group.keys()))
                    
                    # Need enough days for the sequence + 1 target day
                    if len(day_keys) < self.sequence_length + 1:
                        continue

                    # Pre-resolve paths deterministically based on prefix
                    feature_paths, is_sat_flags = [], []
                    for feat in self.dynamic_features:
                        if feat.startswith('s2_'):
                            feature_paths.append(f"satellite/{feat}")
                            is_sat_flags.append(True)
                        else:
                            feature_paths.append(f"features/{feat}")
                            is_sat_flags.append(False)

                    # =========================================================
                    # CREATE THE SLIDING WINDOW SEQUENCES
                    # =========================================================
                    for idx in range(len(day_keys) - self.sequence_length):

                        # ========================================================
                        # TIME-SERIES QUALITY MASK CHECK
                        # ========================================================
                        skip_sequence = False
                        
                        # Check ALL days in the input sequence window
                        for step_offset in range(self.sequence_length):
                            step_group = days_group[day_keys[idx + step_offset]]
                            if "quality_mask" in step_group:
                                skip_sequence = True
                                break
                                
                        if skip_sequence:
                            continue
                            
                        # Check the target day if using forecasts
                        target_key = day_keys[idx + self.sequence_length]
                        target_group = days_group[target_key]
                        
                        if self.use_forecast and "quality_mask" in target_group:
                            continue
                        # ========================================================

                        # Day T is the first day in our sequence window
                        day_T_key = day_keys[idx]
                        day_T_group = days_group[day_T_key]
                        sat_date_T = self._get_expected_date(day_T_group)
                        
                        # The generation script already put the best recent image here.
                        # We just check if it meets our strict age and cloud criteria!
                        if self.cloud_threshold < 100.0:
                            
                            if not self._is_day_cloud_free(day_T_group, sat_date_T):
                                continue # Skip entire sequence if Day T has no valid base image
                        
                        valid_sat_key_T = day_T_key
                        
                        # --- 2. BUILD THE SEQUENCE ---
                        sequence_steps = []
                        last_valid_sat_key = valid_sat_key_T
                        last_valid_sat_date = sat_date_T
                        
                        for step_offset in range(self.sequence_length):
                            step_idx = idx + step_offset
                            step_key = day_keys[step_idx]
                            step_group = days_group[step_key]
                            step_date_str = self._get_expected_date(step_group)
                            
                            if step_offset == 0:
                                # Day T is already validated
                                current_sat_key = valid_sat_key_T
                                current_sat_date = sat_date_T
                            else:
                                # For Day T+1, T+2: Does it have its own valid image?
                                if self.cloud_threshold >= 100.0 or self._is_day_cloud_free(step_group, step_date_str):
                                    current_sat_key = step_key
                                    current_sat_date = step_date_str
                                else:
                                    # Fallback to the last valid image from the sequence
                                    current_sat_key = last_valid_sat_key
                                    current_sat_date = last_valid_sat_date
                            
                            # Keep tracking the last valid image used
                            last_valid_sat_key = current_sat_key
                            last_valid_sat_date = current_sat_date
                            
                            # Calculate exactly how many days old this satellite image is
                            gap_days = -1
                            if step_date_str and current_sat_date:
                                try:
                                    d1 = datetime.strptime(step_date_str, "%Y-%m-%d")
                                    d2 = datetime.strptime(current_sat_date, "%Y-%m-%d")
                                    gap_days = (d1 - d2).days
                                except ValueError:
                                    pass

                            sequence_steps.append({
                                'dyn_key': step_key,
                                'fireday': step_group.attrs.get('fireday', -1),
                                'dyn_date': step_date_str or "Unknown",
                                'sat_key': current_sat_key,
                                'sat_date': current_sat_date or "Unknown",
                                'gap_days': gap_days
                            })
                            
                        # The prediction target is T + sequence_length
                        target_key = day_keys[idx + self.sequence_length]
                        target_group = days_group[target_key]
                        
                        self.samples.append({
                            'file_path': file_path,
                            'fire_id': fire_id,
                            'target_key': target_key,
                            'target_fireday': target_group.attrs.get('fireday', -1),
                            'sequence_steps': sequence_steps,
                            'feature_paths': feature_paths,
                            'is_sat_flags': is_sat_flags
                        })
                            
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        file_path = s['file_path']
        if file_path not in self.open_h5_handles:
            self.open_h5_handles[file_path] = h5py.File(file_path, 'r', swmr=True)
            
        f = self.open_h5_handles[file_path]
        
        # 4D Tensor: (Time, Channels, Height, Width)
        x_tensor = torch.empty((self.sequence_length, self.num_channels, self.patch_size, self.patch_size), dtype=torch.float32)
        
        # ==========================================
        # BUILD METADATA DICT (Per your exact specs)
        # ==========================================
        meta = {
            "fire_id": str(s['fire_id']),
            "target_key": str(s['target_key']),
            "target_fireday": int(s['target_fireday']),
            
            # Sequence details stored as flat lists corresponding to [T, T+1, T+2]
            "seq_dyn_keys": [str(step['dyn_key']) for step in s['sequence_steps']],
            "seq_firedays": [int(step['fireday']) for step in s['sequence_steps']],
            "seq_dyn_dates": [str(step['dyn_date']) for step in s['sequence_steps']],
            "seq_sat_dates": [str(step['sat_date']) for step in s['sequence_steps']],
            "seq_gap_days": [int(step['gap_days']) for step in s['sequence_steps']]
        }
        
        # ==========================================
        # ASSEMBLE TIME STEPS
        # ==========================================
        for t, step_dict in enumerate(s['sequence_steps']):
            dyn_group = f[f"days/{step_dict['dyn_key']}"]
            sat_group = f[f"days/{step_dict['sat_key']}"]
            channel_idx = 0
            
            # 1. Load Dynamics & Satellite
            for path, is_sat in zip(s['feature_paths'], s['is_sat_flags']):
                feat_name = path.split('/')[-1]
                
                target_group = sat_group if is_sat else dyn_group
                arr_patch = target_group[path][:].astype(np.float32)
                
                if is_sat:
                    arr_patch = np.ascontiguousarray(np.flipud(arr_patch))
                    if np.isnan(self.sat_nodata):
                        valid_mask = ~np.isnan(arr_patch)
                    else:
                        valid_mask = (arr_patch != self.sat_nodata)
                    arr_patch = arr_patch * 0.0001
                else:
                    valid_mask = ~np.isnan(arr_patch)
                
                # if self.normalize and feat_name in self.stats_dict and feat_name != self.fire_feature and not is_sat:
                #     mean = self.stats_dict[feat_name]['mean']
                #     std = self.stats_dict[feat_name]['std']
                #     arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std
                # NEW: Added exclusion list for ndvi and evi
                if self.normalize and feat_name in self.stats_dict and feat_name != self.fire_feature and not is_sat:
                    if feat_name not in ['ndvi', 'evi']:
                        mean = self.stats_dict[feat_name]['mean']
                        std = self.stats_dict[feat_name]['std']
                        arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std
                    
                arr_patch[~valid_mask] = 0.0
                x_tensor[t, channel_idx] = torch.from_numpy(arr_patch)
                channel_idx += 1
                
            # 2. Load Statics
            if self.static_features and 'static_features' in f:
                static_group = f['static_features']
                for feature in self.static_features:
                    arr_patch = static_group[feature][:].astype(np.float32)
                    # ==========================================
                    # CYCLICAL ENCODING FOR ASPECT
                    # ==========================================
                    if feature == 'aspect' and self.use_cyclical_aspect:
                        # Convert degrees to radians
                        aspect_rad = arr_patch * (np.pi / 180.0)
                        # Calculate Sin and Cos
                        aspect_sin = np.sin(aspect_rad)
                        aspect_cos = np.cos(aspect_rad)
                        # Assign to tensor (Skipping Z-score normalization)
                        x_tensor[t, channel_idx] = torch.from_numpy(aspect_sin)
                        channel_idx += 1
                        x_tensor[t, channel_idx] = torch.from_numpy(aspect_cos)
                        channel_idx += 1
                        continue # Skip the normal processing and move to the next feature!
                    # ==========================================
                    if self.normalize and feature in self.stats_dict and feature != self.fire_feature:
                        valid_mask = ~np.isnan(arr_patch)
                        mean = self.stats_dict[feature]['mean']
                        std = self.stats_dict[feature]['std']
                        arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std
                    x_tensor[t, channel_idx] = torch.from_numpy(arr_patch)
                    channel_idx += 1
            
            # 3. Process Forecasts (Optional)
            if self.use_forecast:
                target_day_group = f[f"days/{s['target_key']}"]
                for feat in self.forecast_features:
                    arr_patch = target_day_group[f"features/{feat}"][:].astype(np.float32)
                    valid_mask = ~np.isnan(arr_patch)
                    if self.normalize and feat in self.stats_dict:
                        mean = self.stats_dict[feat]['mean']
                        std = self.stats_dict[feat]['std']
                        arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std
                    x_tensor[t, channel_idx] = torch.from_numpy(arr_patch)
                    channel_idx += 1
            
            # Process Historical Fire Masks & Pixel-Wise Fireday
            fireday_grid = torch.zeros((self.patch_size, self.patch_size), dtype=torch.float32)
            
            if self.use_cumuarea_prev:
                fire_mask = torch.zeros((self.patch_size, self.patch_size), dtype=torch.bool)
                all_day_keys = sorted(list(f['days'].keys()))
                current_day_idx = all_day_keys.index(step_dict['dyn_key'])
                
                # Loop from the absolute beginning of the fire up to today
                history_keys = all_day_keys[:current_day_idx + 1]
                
                for h_key in history_keys:
                    h_group = f[f"days/{h_key}"]
                    h_arr = h_group[f"features/{self.fire_feature}"][:]
                    day_mask = (torch.from_numpy(h_arr) > 0)

                    # Smooth the historical mask
                    if self.smooth_mask:
                        day_mask = smooth_fire_mask(day_mask, kernel_size=self.smooth_kernel)
                    
                    # Get the actual fireday for this specific historical day
                    h_fireday = h_group.attrs.get('fireday', -1)
                    
                    if h_fireday > 0:
                        # Find ONLY the pixels that caught fire on this specific day
                        new_burn_pixels = day_mask & ~fire_mask
                        # Stamp them with this day's scaled fireday
                        fireday_grid[new_burn_pixels] = float(h_fireday) / 100.0
                        
                    # Update the cumulative mask
                    fire_mask |= day_mask
            else:
                arr = dyn_group[f"features/{self.fire_feature}"][:]
                fire_mask = (torch.from_numpy(arr) > 0)

                # Smooth the base mask if no history
                if self.smooth_mask:
                    fire_mask = smooth_fire_mask(fire_mask, kernel_size=self.smooth_kernel)
                
                c_fireday = step_dict['fireday']
                if c_fireday > 0:
                    fireday_grid[fire_mask] = float(c_fireday) / 100.0
                    
            # Assign the binary cumulative mask to second-to-last channel
            x_tensor[t, -2] = fire_mask.float()
            
            # Assign your new pixel-wise fireday grid to the last channel
            x_tensor[t, -1] = fireday_grid
            
        # ==========================================
        # TARGET MASK (The Prediction Output)
        # ==========================================
        # Must pull from -2 because the binary fire mask shifted
        last_step_fire_mask = x_tensor[-1, -2] > 0
        
        target_day_group = f[f"days/{s['target_key']}"]
        raw_next_arr = target_day_group[f"features/{self.fire_feature}"][:]
        raw_next_mask = (torch.from_numpy(raw_next_arr) > 0)

        # Smooth the target mask
        if self.smooth_mask:
            raw_next_mask = smooth_fire_mask(raw_next_mask, kernel_size=self.smooth_kernel)
        
        y_tensor = torch.zeros((self.patch_size, self.patch_size), dtype=torch.float32)
        new_growth_mask = raw_next_mask & ~last_step_fire_mask

        if self.use_cumuarea:
             y_tensor[last_step_fire_mask] = 1.0 
             y_tensor[new_growth_mask] = 2.0
        else:
             y_tensor[new_growth_mask] = 1.0
        
        if self.downscale:
            
            # x_tensor is (Time, Channels, H, W). 
            # max_pool2d treats 'Time' as the batch dimension
            x_tensor = F.max_pool2d(x_tensor, 
                                    kernel_size=self.downscale_factor, 
                                    stride=self.downscale_factor)
            
            # y_tensor is (H, W). max_pool2d needs at least a channel dimension (C, H, W)
            # So we add a dummy channel, pool it, and remove the channel.
            y_tensor = y_tensor.unsqueeze(0)
            y_tensor = F.max_pool2d(y_tensor, 
                                    kernel_size=self.downscale_factor, 
                                    stride=self.downscale_factor)
            y_tensor = y_tensor.squeeze(0)

        return x_tensor, y_tensor


def get_timeseries_dataloaders(config: Config, cloud_threshold: float = 40.0, max_sat_lookback_days: int = 2, sequence_length: int = 3) -> tuple[DataLoader, DataLoader, DataLoader]:
    tc = config.training

    dynamic_feats = [
        'tmax', 'rh', 'ws', 'prec', 'u10', 'v10',
        'evi', 'ndvi', 
        's2_B02', 's2_B03', 's2_B04', 's2_B08', 's2_B11', 's2_B12'
    ]
    static_feats = ['dem_avg', 'slope', 'aspect']

    valid_ids = np.load(f'{settings.METADATA_FOLDER}/fire_ids_256_5_years.npy').tolist()

    # report_path = f'{settings.METADATA_FOLDER}/diagnosis_report_full_features_256.txt'
    # to_remove = set()
    # current_fire_id = None

    # with open(report_path, 'r') as f:
    #     for line in f:
    #         line = line.strip()
    #         if line.startswith("Fire ID:"):
    #             current_fire_id = line.replace("Fire ID:", "").strip()
    #         elif line.startswith("-") and current_fire_id:
    #             to_remove.add(current_fire_id)

    # valid_ids = [fid for fid in valid_ids if fid not in to_remove]
    
    print(f'Valid ids : {len(valid_ids)}')

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

    # Validating the specific Time-Series configuration
    overall_dataset = H5FireTimeSeriesDataset(
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
        max_sat_lookback_days=max_sat_lookback_days,
        sequence_length=sequence_length,
        use_cyclical_aspect=tc.use_cyclical_aspect
    )
    print(f'Number of clean time-series samples : {len(overall_dataset)}')
    retained_ids = sorted(list({s['fire_id'] for s in overall_dataset.samples}))
    
    fire_growth_combined = pd.concat([df_2020, df_2021, df_2022, df_2023, df_2024], ignore_index=True)
    fire_growth_filtered = fire_growth_combined[fire_growth_combined['ID'].astype(str).isin(retained_ids)]

    train_ids, val_ids, test_ids = create_stratified_splits(fire_growth_filtered, 
                                                            train_split=tc.train_split,
                                                            random_state=42)

    del df_2020, df_2021, df_2022, df_2023, df_2024, fire_growth_combined, fire_growth_filtered
    gc.collect() 

    stats_dict = calculate_h5_statistics_filtered(settings.H5_OUTPUT_FOLDER, 
                                                  train_ids, 
                                                  cloud_threshold=cloud_threshold,
                                                  sat_nodata=np.nan,
                                                  max_sat_lookback_days=max_sat_lookback_days)

    common_kwargs = {
        'h5_dir': settings.H5_OUTPUT_FOLDER,
        'dynamic_features': dynamic_feats,
        'static_features': static_feats,
        'use_forecast': False,
        'fire_feature': "cumuarea",
        'patch_size': settings.GRID_SIZE,
        'use_cumuarea': tc.use_cumuarea,
        'use_cumuarea_prev': tc.use_cumuarea_prev,
        'normalize': True,
        'stats_dict': stats_dict,
        'sat_nodata': np.nan,
        'cloud_threshold': cloud_threshold,
        'max_sat_lookback_days': max_sat_lookback_days,
        'sequence_length': sequence_length,
        'downscale': tc.downscale,
        'downscale_factor': tc.downscale_factor,
        'smooth_mask': tc.smooth_mask,
        'smooth_kernel': tc.smooth_kernel,
        'use_cyclical_aspect': tc.use_cyclical_aspect
    }

    train_dataset = H5FireTimeSeriesDataset(id_list=train_ids, **common_kwargs)
    val_dataset = H5FireTimeSeriesDataset(id_list=val_ids, **common_kwargs)
    test_dataset = H5FireTimeSeriesDataset(id_list=test_ids, **common_kwargs)
    
    print(f'Number of training samples :: {len(train_dataset)}')
    print(f'Number of validation samples :: {len(val_dataset)}')
    print(f'Number of test samples :: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, shuffle=True, num_workers=tc.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=tc.batch_size, shuffle=False, num_workers=tc.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=tc.batch_size, shuffle=False, num_workers=tc.num_workers)

    return train_loader, val_loader, test_loader