from __future__ import annotations

import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import pandas as pd 
from tqdm import tqdm
import gc
import json
from datetime import datetime
from collections import OrderedDict

from src.config import Config
from samples_generation.data_generator_helper import create_stratified_splits, calculate_h5_statistics

from configs import settings

class H5FireSimpleDataset(Dataset):
    """ """
    
    def __init__(self, h5_dir, id_list, mapper, dynamic_features, static_features=None, 
                 fire_feature='firearea', patch_size=256, 
                 use_cumuarea_prev=False, use_cumuarea=False, transform=None,
                 normalize=False, stats_dict=None, sat_nodata=0, 
                 use_cyclical_aspect=False, return_sat_age=False):
        
        self.h5_dir = h5_dir
        self.id_list = [str(fid) for fid in id_list] 
        self.mapper = mapper
        
        self.dynamic_features = dynamic_features
        self.static_features = static_features if static_features else []
        self.fire_feature = fire_feature
        self.patch_size = patch_size
        
        # Accumulation flags
        self.use_cumuarea_prev = use_cumuarea_prev  
        self.use_cumuarea = use_cumuarea            
        self.transform = transform

        # --- NORMALIZATION PARAMS ---
        self.normalize = normalize
        self.stats_dict = stats_dict
        self.sat_nodata = sat_nodata
        
        self.use_cyclical_aspect = use_cyclical_aspect
        self.return_sat_age = return_sat_age
        
        # Safety check
        if self.normalize and self.stats_dict is None:
            raise ValueError("If normalize=True, you must provide the stats_dict!")
        
        # +2 to account for BOTH the binary fire mask AND the fireday grid
        base_channels = len(self.dynamic_features) + len(self.static_features) + 2
        
        # If we encode aspect, 1 channel becomes 2 (sin, cos), so we add 1 net channel
        if self.use_cyclical_aspect and 'aspect' in self.static_features:
            base_channels += 1

        self.num_channels = base_channels
        
        self.samples = [] 
        # self.open_h5_handles = {}
        self.open_h5_handles = OrderedDict()
        self.max_open_files = 700
        
        self._index_files()

    # --- HELPER METHODS ---
    def _get_expected_date(self, day_group) -> str | None:
        """Extracts the fire day date.

        Args:
          day_group: the key of the day in the H5 file

        Returns:
            the fire's date
        """
        fire_datetime = day_group.attrs.get("noon_utc") or day_group.attrs.get("start_utc") or day_group.attrs.get("end_utc")
        if fire_datetime:
            if isinstance(fire_datetime, bytes):
                fire_datetime = fire_datetime.decode('utf-8')
            return fire_datetime.split("T")[0]
        return None

    def _get_sat_age_days(self, day_group, expected_date_str) -> float:
        """Calculates the gap between fire day and satellite acquisition.

        Args:
          day_group: the key of the day in the H5 file
          expected_date_str: the date of the fireday

        Returns:
            the difference in days between the fireday and the day of the satellite image
        """
        if not expected_date_str or "satellite" not in day_group:
            return 0.0
            
        sat_attrs = day_group["satellite"].attrs
        if "cloud_cover_stats" not in sat_attrs:
            return 0.0
            
        stats_attr = sat_attrs["cloud_cover_stats"]
        if isinstance(stats_attr, bytes):
            stats_attr = stats_attr.decode('utf-8')
            
        try:
            stats_list = json.loads(stats_attr)
            exp_dt = datetime.strptime(expected_date_str, "%Y-%m-%d")
            
            min_gap = float('inf')
            found_valid = False
            
            for stat in stats_list:
                acq_date_str = stat.get("acquisition_date")
                if acq_date_str:
                    acq_dt = datetime.strptime(acq_date_str, "%Y-%m-%d")
                    gap_days = (exp_dt - acq_dt).days
                    
                    if gap_days >= 0 and gap_days < min_gap:
                        min_gap = gap_days
                        found_valid = True
            
            return float(min_gap) if found_valid else 0.0
            
        except Exception:
            return 0.0

    def _get_h5_handle(self, fire_id):
        """Helper to manage persistent file handles with an LRU capacity limit.

        Args:
          fire_id: the fire's ID

        Returns: the fire's file handle

        """
        file_path = os.path.join(self.h5_dir, f"fire_{fire_id}.h5")
        
        # If it's already open, mark it as 'recently used' and return it
        if file_path in self.open_h5_handles:
            self.open_h5_handles.move_to_end(file_path)
            return self.open_h5_handles[file_path]
            
        # If the cache is full, close and remove the oldest file
        if len(self.open_h5_handles) >= self.max_open_files:
            oldest_path, oldest_handle = self.open_h5_handles.popitem(last=False)
            try:
                oldest_handle.close()
            except Exception:
                pass
                
        # Open the new file, add it to the cache, and return it
        new_handle = h5py.File(file_path, 'r', swmr=True)
        self.open_h5_handles[file_path] = new_handle
        return new_handle
        
    def __del__(self):
        """Clean up all open file handles."""
        if hasattr(self, 'open_h5_handles'):
            for f in self.open_h5_handles.values():
                try:
                    f.close()
                except:
                    pass

    def _index_files(self):
        """Builds dataset samples directly from the offline mapper JSON."""
        
        for global_key, curr_fires_raw in tqdm(self.mapper.items(), desc="Indexing Tile Days from Mapper"):
            
            # Extract identifiers from the global key: tile_X_Y_DOB_YYYY_DDD
            parts = global_key.split('_DOB_')
            tile_id = parts[0]
            year_dob = parts[1].split('_')
            year = int(year_dob[0])
            dob = int(year_dob[1])
            
            # Filter the fires to ensure they are in our active split (train/val/test)
            curr_fires = [f for f in curr_fires_raw if f['fire_id'] in self.id_list]
            if not curr_fires:
                continue
                
            # Automatically infer Tomorrow's global key
            next_global_key = f"{tile_id}_DOB_{year}_{dob + 1}"
            next_fires_raw = self.mapper.get(next_global_key, [])
            next_fires = [f for f in next_fires_raw if f['fire_id'] in self.id_list]
            
            if not next_fires:
                continue # We skip if there's no ground truth label available for tomorrow
                
            # We open the H5 file for the FIRST fire in the list strictly to run our size/quality checks
            primary_fire = curr_fires[0]
            file_path = os.path.join(self.h5_dir, f"fire_{primary_fire['fire_id']}.h5")
            
            try:
                with h5py.File(file_path, 'r') as f:
                    curr_day_group = f[f"{tile_id}/days/{primary_fire['day_key']}"]
                    
                    # --- SIZE CHECK ---
                    sample_feat = list(curr_day_group['features'].keys())[0]
                    h, w = curr_day_group[f'features/{sample_feat}'].shape
                    if h != self.patch_size or w != self.patch_size:
                        continue 
                        
                    # ========================================================
                    # QUALITY MASK CHECK
                    # ========================================================
                    if "quality_mask" in curr_day_group:
                        continue

                    # --- CALCULATE SAT AGE ---
                    delta_t_days = 0.0
                    if self.return_sat_age:
                        exp_date_str = self._get_expected_date(curr_day_group)
                        delta_t_days = self._get_sat_age_days(curr_day_group, exp_date_str)
                        
                    fireday = curr_day_group.attrs.get('fireday', -1)
                        
            except Exception as e:
                # File is missing or corrupted, skip
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
                    
            self.samples.append({
                'tile_id': tile_id,
                'dob': dob,
                'year': year,
                'curr_fires': curr_fires,      # List of all overlapping fires for TODAY
                'next_fires': next_fires,      # List of all overlapping fires for TOMORROW
                'primary_fire_id': primary_fire['fire_id'],
                'primary_curr_key': primary_fire['day_key'],
                'primary_next_key': next_fires[0]['day_key'],
                'fireday': fireday,
                'feature_paths': feature_paths, 
                'is_sat_flags': is_sat_flags,
                'delta_t': delta_t_days
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tile_id = s['tile_id']
        
        # Pre-allocate input tensor with ZEROS safely
        x_tensor = torch.zeros((self.num_channels, self.patch_size, self.patch_size), dtype=torch.float32)
        
        # Load Environment from PRIMARY Fire (Because Topo/Weather/Sat are identical)
        f_primary = self._get_h5_handle(s['primary_fire_id'])
        curr_day_group = f_primary[f"{tile_id}/days/{s['primary_curr_key']}"]
        channel_idx = 0
        
        # Load Dynamics & Satellite
        for path, is_sat in zip(s['feature_paths'], s['is_sat_flags']):
            feat_name = path.split('/')[-1]
            arr_patch = curr_day_group[path][:].astype(np.float32)
            
            if is_sat:
                # arr_patch = np.ascontiguousarray(np.flipud(arr_patch))
                
                if np.isnan(self.sat_nodata):
                    valid_mask = ~np.isnan(arr_patch)
                else:
                    valid_mask = (arr_patch != self.sat_nodata)
                
                if 'SCL' not in feat_name:
                    arr_patch = arr_patch * 0.0001
                else:
                    arr_patch = arr_patch * 0.1
            else:
                valid_mask = ~np.isnan(arr_patch)
            
            # --- NORMALIZATION LOGIC ---
            if self.normalize and feat_name in self.stats_dict and feat_name != self.fire_feature and not is_sat:
                if feat_name not in ['ndvi', 'evi']:
                    mean = self.stats_dict[feat_name]['mean']
                    std = self.stats_dict[feat_name]['std']
                    arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std
                
            arr_patch[~valid_mask] = 0.0
            x_tensor[channel_idx] = torch.from_numpy(arr_patch)
            channel_idx += 1
            
        # 3. Load Statics
        if self.static_features and f"{tile_id}/static_features" in f_primary:
            static_group = f_primary[f"{tile_id}/static_features"]
            for feature in self.static_features:
                arr_patch = static_group[feature][:].astype(np.float32)

                if feature == 'aspect' and self.use_cyclical_aspect:
                    aspect_rad = arr_patch * (np.pi / 180.0)
                    aspect_sin = np.sin(aspect_rad)
                    aspect_cos = np.cos(aspect_rad)
                    
                    x_tensor[channel_idx] = torch.from_numpy(aspect_sin)
                    channel_idx += 1
                    x_tensor[channel_idx] = torch.from_numpy(aspect_cos)
                    channel_idx += 1
                    continue
                
                if self.normalize and feature in self.stats_dict and feature != self.fire_feature:
                    valid_mask = ~np.isnan(arr_patch)
                    mean = self.stats_dict[feature]['mean']
                    std = self.stats_dict[feature]['std']
                    arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std
                
                x_tensor[channel_idx] = torch.from_numpy(arr_patch)
                channel_idx += 1

        # ---------------------------------------------------------
        # 4. Process Fire Masks (Dynamic Aggregation of All Overlaps)
        # ---------------------------------------------------------
        curr_fire_mask = torch.zeros((self.patch_size, self.patch_size), dtype=torch.bool)
        fireday_grid = torch.zeros((self.patch_size, self.patch_size), dtype=torch.float32)
        
        for fire_dict in s['curr_fires']:
            fire_id = fire_dict['fire_id']
            curr_key = fire_dict['day_key']
            f_fire = self._get_h5_handle(fire_id)
            
            if self.use_cumuarea_prev:
                # Dynamically retrieve history specifically for this fire
                day_keys = sorted(list(f_fire[f"{tile_id}/days"].keys()))
                curr_idx = day_keys.index(curr_key)
                h_keys = day_keys[:curr_idx + 1]
                
                for h_key in h_keys:
                    h_group = f_fire[f"{tile_id}/days/{h_key}"]
                    h_arr = h_group[f"features/{self.fire_feature}"][:]
                    day_mask = (torch.from_numpy(h_arr) > 0)
                    
                    h_fireday = h_group.attrs.get('fireday', -1)
                    if h_fireday > 0:
                        new_burn_pixels = day_mask & ~curr_fire_mask
                        fireday_grid[new_burn_pixels] = float(h_fireday) / 100.0
                        
                    curr_fire_mask |= day_mask
            else:
                curr_arr = f_fire[f"{tile_id}/days/{curr_key}/features/{self.fire_feature}"][:]
                day_mask = (torch.from_numpy(curr_arr) > 0)
                
                c_fireday = f_fire[f"{tile_id}/days/{curr_key}"].attrs.get('fireday', -1)
                if c_fireday > 0:
                    new_burn_pixels = day_mask & ~curr_fire_mask
                    fireday_grid[new_burn_pixels] = float(c_fireday) / 100.0
                    
                curr_fire_mask |= day_mask
                
        x_tensor[-2] = curr_fire_mask.float()
        x_tensor[-1] = fireday_grid

        # ---------------------------------------------------------
        # B. Determine NEXT DAY mask (Output Label)
        # ---------------------------------------------------------
        raw_next_mask = torch.zeros((self.patch_size, self.patch_size), dtype=torch.bool)
        
        for fire_dict in s['next_fires']:
            fire_id = fire_dict['fire_id']
            next_key = fire_dict['day_key']
            
            f_fire = self._get_h5_handle(fire_id)
            next_arr = f_fire[f"{tile_id}/days/{next_key}/features/{self.fire_feature}"][:]
            raw_next_mask |= (torch.from_numpy(next_arr) > 0)
        
        y_tensor = torch.zeros((self.patch_size, self.patch_size), dtype=torch.float32)
        new_growth_mask = raw_next_mask & ~curr_fire_mask

        if self.use_cumuarea:
             y_tensor[curr_fire_mask] = 1.0 
             y_tensor[new_growth_mask] = 2.0
        else:
             y_tensor[new_growth_mask] = 1.0

        if self.transform:
            x_tensor = self.transform(x_tensor)

        meta = {
            "tile_id": s['tile_id'],
            "dob": s['dob'],
            "primary_fire_id": s['primary_fire_id'],
            "fireday": s['fireday']
        }

        if self.return_sat_age:
            delta_t_tensor = torch.tensor(s['delta_t'], dtype=torch.float32)
            return x_tensor, y_tensor, delta_t_tensor
        else:
            return x_tensor, y_tensor

def save_dataset_to_disk(dataset, output_base_folder, split_name):
    """Iterates through the dataset and saves each sample as a .pt file.
    Creates the necessary subfolders automatically.

    Args:
      dataset: the samples dataset
      output_base_folder: the output folder of the samples
      split_name: train, val, or test
    """
    # Create the specific subfolder
    split_folder = os.path.join(output_base_folder, split_name)
    os.makedirs(split_folder, exist_ok=True)
    
    print(f"\nSaving {split_name.upper()} split to {split_folder}...")
    
    # Iterate directly through the dataset and save
    for idx in tqdm(range(len(dataset)), desc=f"Generating {split_name}"):
        
        x_tensor, y_tensor, delta_t = dataset[idx]

        x_np = x_tensor.numpy().astype(np.float16)
        y_np = y_tensor.numpy().astype(np.uint8)
        delta_t_np = delta_t.numpy().astype(np.uint8)

        file_path = os.path.join(split_folder, f"sample_{idx}.npz")
        np.savez(file_path, x=x_np, y=y_np, delta_t=delta_t_np)


def generate_simple_offline_data(config: Config) -> None:
    """

    Args:
      config: Config: the config parameters
    """
    
    tc = config.training

    is_sat_age = True

    dynamic_feats = ['tmax', 'rh', 'ws', 'prec', 'u10', 'v10', 'evi', 'ndvi', 
                     's2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b11', 's2_b12', 's2_scl']
    static_feats = ['dem', 'slope', 'aspect', 'biomass', 'closure', 'prcb', 'prcc']
    target_years = ["2020", "2021", "2022", "2023", "2024"] 
    
    # ---------------------------------------------------------
    # LOAD MAPPERS & DATAFRAMES
    # ---------------------------------------------------------
    print('Loading Offline Tile Mappers...')
    master_mapper = {}
    for year in target_years:
        mapper_path = os.path.join(settings.METADATA_FOLDER, f"tile_dob_mapper_{year}.json")
        if os.path.exists(mapper_path):
            with open(mapper_path, 'r') as f:
                master_mapper.update(json.load(f))

    print('Reading dataframes dynamically...')
    columns_to_use = [col for col in settings.SUBSET_FEATURE_LIST if col not in ['easting', 'northing']]
    dfs = [pd.read_csv(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv', usecols=columns_to_use) 
           for year in target_years if os.path.exists(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv')]
    
    fire_growth_combined = pd.concat(dfs, ignore_index=True)
    valid_ids = fire_growth_combined['ID'].unique()
    print(len(valid_ids))

    # ---------------------------------------------------------
    # FILTER VALID IDs
    # ---------------------------------------------------------
    overall_dataset = H5FireSimpleDataset(
        h5_dir=settings.H5_OUTPUT_FOLDER, id_list=valid_ids, mapper=master_mapper,
        dynamic_features=dynamic_feats, static_features=static_feats,
        fire_feature="cumuarea", patch_size=settings.GRID_SIZE,
        use_cumuarea=tc.use_cumuarea, use_cumuarea_prev=tc.use_cumuarea_prev,
        normalize=False, stats_dict=None, sat_nodata=np.nan
    )
    
    retained_ids = set()
    for s in overall_dataset.samples:
        for f_dict in s['curr_fires']:
            retained_ids.add(f_dict['fire_id'])
        for f_dict in s['next_fires']:
            retained_ids.add(f_dict['fire_id'])
    retained_ids = sorted(list(retained_ids))

    print(len(retained_ids))
    
    fire_growth_filtered = fire_growth_combined[fire_growth_combined['ID'].astype(str).isin(retained_ids)]

    train_ids, val_ids, test_ids = create_stratified_splits(
        fire_growth_filtered, train_split=tc.train_split, random_state=42, overlap_mapper=master_mapper
    )

    del dfs, fire_growth_combined, fire_growth_filtered
    gc.collect() 

    # ---------------------------------------------------------
    # CALCULATE STATS
    # ---------------------------------------------------------
    stats_dict = calculate_h5_statistics(
        settings.H5_OUTPUT_FOLDER, 
        master_mapper,
        train_ids,
        stats_filename="simple_dataset_stats.json"
    )


    common_kwargs = {
        'h5_dir': settings.H5_OUTPUT_FOLDER, 'mapper': master_mapper,
        'dynamic_features': dynamic_feats, 'static_features': static_feats,
        'fire_feature': "cumuarea", 'patch_size': settings.GRID_SIZE,
        'use_cumuarea': tc.use_cumuarea, 'use_cumuarea_prev': tc.use_cumuarea_prev,
        'normalize': True, 'stats_dict': stats_dict, 'sat_nodata': np.nan,
        'use_cyclical_aspect': tc.use_cyclical_aspect, 'return_sat_age': is_sat_age
    }

    # ---------------------------------------------------------
    # INSTANTIATE DATASETS AND SAVE TO DISK
    # ---------------------------------------------------------
    train_dataset = H5FireSimpleDataset(id_list=train_ids, **common_kwargs)
    val_dataset = H5FireSimpleDataset(id_list=val_ids, **common_kwargs)
    test_dataset = H5FireSimpleDataset(id_list=test_ids, **common_kwargs)

    # Save everything directly to the SAMPLE_FOLDER
    save_dataset_to_disk(train_dataset, settings.SAMPLE_FOLDER, "train")
    save_dataset_to_disk(val_dataset, settings.SAMPLE_FOLDER, "val")
    save_dataset_to_disk(test_dataset, settings.SAMPLE_FOLDER, "test")
    
    print("\nAll data successfully generated and saved to disk!")