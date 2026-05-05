from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
import os
import h5py
import numpy as np
import pandas as pd 
from tqdm import tqdm
import gc
import json
from collections import OrderedDict

from src.config import Config
from samples_generation.data_generator_helper import create_stratified_splits, calculate_h5_statistics

from configs import settings

class H5FireTimeSeriesDataset(Dataset):
    """
    Dataset to contruct the time series samples from the H5 files
    """
    
    def __init__(self, h5_dir, id_list, mapper, dynamic_features, static_features=None, 
                 fire_feature='firearea', patch_size=256, seq_length=3,
                 use_cumuarea_prev=False, use_cumuarea=False, transform=None,
                 normalize=False, stats_dict=None, sat_nodata=0, 
                 use_cyclical_aspect=False):
        
        self.h5_dir = h5_dir
        self.id_list = [str(fid) for fid in id_list] 
        self.mapper = mapper
        
        self.dynamic_features = dynamic_features
        self.static_features = static_features if static_features else []
        self.fire_feature = fire_feature
        self.patch_size = patch_size

        # Time series length
        self.seq_length = seq_length
        
        # Accumulation flags
        self.use_cumuarea_prev = use_cumuarea_prev  
        self.use_cumuarea = use_cumuarea            
        self.transform = transform

        # --- NORMALIZATION PARAMS ---
        self.normalize = normalize
        self.stats_dict = stats_dict
        self.sat_nodata = sat_nodata
        
        self.use_cyclical_aspect = use_cyclical_aspect
        
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
        self.open_h5_handles = OrderedDict()
        self.max_open_files = 100
        
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
            _, oldest_handle = self.open_h5_handles.popitem(last=False)
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
        """Builds time-series samples (T, T+1, T+2 -> Predict T+3) directly from the mapper."""
        
        for global_key, starting_fires_raw in tqdm(self.mapper.items(), desc="Indexing Tile Sequences"):
            
            # Extract identifiers for Day T
            parts = global_key.split('_DOB_')
            tile_id = parts[0]
            year = int(parts[1].split('_')[0])
            start_dob = int(parts[1].split('_')[1])
            
            sequence_valid = True
            sequence_fires_list = []
            
            # =======================================================
            # 1. BUILD INPUT SEQUENCE (T, T+1, T+2 ...)
            # =======================================================
            for step in range(self.seq_length):
                curr_dob = start_dob + step
                curr_key = f"{tile_id}_DOB_{year}_{curr_dob}"
                
                curr_fires_raw = self.mapper.get(curr_key, [])
                curr_fires = [f for f in curr_fires_raw if f['fire_id'] in self.id_list]
                
                if not curr_fires:
                    sequence_valid = False
                    break # Missing a day in the sequence
                
                primary_fire = curr_fires[0]
                file_path = os.path.join(self.h5_dir, f"fire_{primary_fire['fire_id']}.h5")
                
                try:
                    with h5py.File(file_path, 'r') as f:
                        day_group = f[f"{tile_id}/days/{primary_fire['day_key']}"]
                        
                        # Size check
                        sample_feat = list(day_group['features'].keys())[0]
                        h, w = day_group[f'features/{sample_feat}'].shape
                        if h != self.patch_size or w != self.patch_size:
                            sequence_valid = False
                            break
                            
                        # QUALITY MASK CHECK
                        if "quality_mask" in day_group:
                            sequence_valid = False
                            break
                            
                except Exception:
                    sequence_valid = False
                    break
                    
                sequence_fires_list.append(curr_fires)

            if not sequence_valid:
                continue # Discard sample if any input day is broken/missing

            # =======================================================
            # 2. CHECK TARGET DAY (T + seq_length)
            # =======================================================
            target_dob = start_dob + self.seq_length
            target_key = f"{tile_id}_DOB_{year}_{target_dob}"
            
            target_fires_raw = self.mapper.get(target_key, [])
            target_fires = [f for f in target_fires_raw if f['fire_id'] in self.id_list]
            
            if not target_fires:
                continue # Discard sample if we have no ground truth for tomorrow
                
            # Pre-resolve paths based on the starting day's features
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
                'year': year,
                'start_dob': start_dob,
                'sequence_fires': sequence_fires_list, # List of lists containing T, T+1, T+2
                'next_fires': target_fires,            # The target fires for T+3
                'feature_paths': feature_paths, 
                'is_sat_flags': is_sat_flags
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tile_id = s['tile_id']
        
        # [Seq_Length, Channels, H, W]
        x_tensor = torch.zeros((self.seq_length, self.num_channels, self.patch_size, self.patch_size), dtype=torch.float32)

        # Loop through T, T+1, T+2...
        for t_idx, step_fires in enumerate(s['sequence_fires']):
            
            primary_step_fire = step_fires[0]
            f_primary = self._get_h5_handle(primary_step_fire['fire_id'])
            step_day_group = f_primary[f"{tile_id}/days/{primary_step_fire['day_key']}"]
            
            channel_idx = 0

            # --- Load Dynamics & Satellite ---
            for path, is_sat in zip(s['feature_paths'], s['is_sat_flags']):
                feat_name = path.split('/')[-1]
                arr_patch = step_day_group[path][:].astype(np.float32)
            
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
                
                # FIXED: Assign to the specific time step (t_idx)
                x_tensor[t_idx, channel_idx] = torch.from_numpy(arr_patch)
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
                        
                        # FIXED: Assign to the specific time step (t_idx)
                        x_tensor[t_idx, channel_idx] = torch.from_numpy(aspect_sin)
                        channel_idx += 1
                        x_tensor[t_idx, channel_idx] = torch.from_numpy(aspect_cos)
                        channel_idx += 1
                        continue
                    
                    if self.normalize and feature in self.stats_dict and feature != self.fire_feature:
                        valid_mask = ~np.isnan(arr_patch)
                        mean = self.stats_dict[feature]['mean']
                        std = self.stats_dict[feature]['std']
                        arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std
                    
                    # FIXED: Assign to the specific time step (t_idx)
                    x_tensor[t_idx, channel_idx] = torch.from_numpy(arr_patch)
                    channel_idx += 1

            # ---------------------------------------------------------
            # 4. Process Fire Masks (Dynamic Aggregation of All Overlaps)
            # ---------------------------------------------------------
            curr_fire_mask = torch.zeros((self.patch_size, self.patch_size), dtype=torch.bool)
            fireday_grid = torch.zeros((self.patch_size, self.patch_size), dtype=torch.float32)
            
            for fire_dict in step_fires:
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
                    
            # FIXED: Assign to the specific time step (t_idx)
            x_tensor[t_idx, -2] = curr_fire_mask.float()
            x_tensor[t_idx, -1] = fireday_grid

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
        
        # Extract the fire mask from the LAST frame of the input sequence
        # (t_idx = -1, channel = -2)
        last_input_mask = x_tensor[-1, -2].bool()
        new_growth_mask = raw_next_mask & ~last_input_mask

        if self.use_cumuarea:
             y_tensor[last_input_mask] = 1.0 
             y_tensor[new_growth_mask] = 2.0
        else:
             y_tensor[new_growth_mask] = 1.0

        if self.transform:
            x_tensor = self.transform(x_tensor)

        positions = torch.arange(
            s['start_dob'],
            s['start_dob'] + self.seq_length,
            dtype=torch.float32,
        )

        return x_tensor, y_tensor, positions
    
def save_dataset_to_disk(dataset, output_base_folder, split_name):
    """Iterates through the dataset and saves each sample as a .pt file.
    Creates the necessary subfolders automatically.

    Args:
      dataset: the samples dataset
      output_base_folder: the saving folder's directory
      split_name: train, val, or test
    """
    # Create the specific subfolder
    split_folder = os.path.join(output_base_folder, split_name)
    os.makedirs(split_folder, exist_ok=True)
    
    print(f"\nSaving {split_name.upper()} split to {split_folder}...")
    
    # Iterate directly through the dataset and save
    for idx in tqdm(range(len(dataset)), desc=f"Generating {split_name}"):
        
        x_tensor, y_tensor, positions = dataset[idx]
        data_dict = {'x': x_tensor, 'y': y_tensor, 'positions': positions}
            
        # Save the tensor dictionary to disk
        file_path = os.path.join(split_folder, f"sample_{idx}.pt")
        torch.save(data_dict, file_path)

def generate_timeseries_offline_data(config: Config, seq_length: int) -> None:
    """
    End-to-end process to generate the samples

    Args:
      config: Config: the config params
      seq_length: int: the length of the time series
    """
    
    tc = config.training

    dynamic_feats = [
        'tmax', 'rh', 'ws', 'prec', 'u10', 'v10',
        'evi', 'ndvi', 
        's2_B02', 's2_B03', 's2_B04', 's2_B08', 's2_B11', 's2_B12',
        's2_SCL'
    ]
    dynamic_feats = [feat.lower() for feat in dynamic_feats]
    static_feats = [
        'dem', 'slope', 'aspect', 'biomass', 'closure', 'prcb', 'prcc'
    ]
    
    target_years = ["2020", "2021", "2022", "2023", "2024"] 
    
    # ---------------------------------------------------------
    # LOAD AND MERGE YEARLY MAPPERS
    # ---------------------------------------------------------
    print('Loading Offline Tile Mappers...')
    master_mapper = {}
    for year in target_years:
        mapper_path = os.path.join(settings.METADATA_FOLDER, f"tile_dob_mapper_{year}.json")
        if os.path.exists(mapper_path):
            with open(mapper_path, 'r') as f:
                master_mapper.update(json.load(f))
        else:
            print(f"[!] Warning: Could not find mapper for {year} at {mapper_path}")

    # ---------------------------------------------------------
    # DYNAMICALLY LOAD AND CONCAT DATAFRAMES
    # ---------------------------------------------------------
    print('Reading dataframes dynamically...')
    columns_to_use = [col for col in settings.SUBSET_FEATURE_LIST if col not in ['easting', 'northing']]
    
    dfs = []
    for year in target_years:
        csv_path = f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv'
        if os.path.exists(csv_path):
            print(f"  -> Loading {year} CSV...")
            dfs.append(pd.read_csv(csv_path, usecols=columns_to_use))
        else:
            print(f"[!] Warning: Could not find CSV for {year} at {csv_path}")
            
    # Combine all loaded years into one master dataframe
    fire_growth_combined = pd.concat(dfs, ignore_index=True)
    print('Read and combined dataframes.')

    valid_ids = fire_growth_combined['ID'].unique()
    print(f'Number of initial fires: {len(valid_ids)}')

    # Initialize the temporary "Overall Dataset" to retrieve retained IDs
    overall_dataset = H5FireTimeSeriesDataset(
        h5_dir=settings.H5_OUTPUT_FOLDER, 
        id_list=valid_ids,
        mapper=master_mapper,
        seq_length=seq_length,
        dynamic_features=dynamic_feats,
        static_features=static_feats,
        fire_feature="cumuarea",
        patch_size=settings.GRID_SIZE,
        use_cumuarea=tc.use_cumuarea,
        use_cumuarea_prev=tc.use_cumuarea_prev,
        normalize=False, 
        stats_dict=None, 
        sat_nodata=np.nan
    )

    # Since multiple fires can exist in one sample, we gather all unique fire_ids that made the cut
    retained_ids = set()
    for s in overall_dataset.samples:
        # Keep all fires in the input sequence
        for step_fires in s['sequence_fires']:
            for f_dict in step_fires:
                retained_ids.add(f_dict['fire_id'])
                
        # Keep all target/next day fires
        for f_dict in s['next_fires']:
            retained_ids.add(f_dict['fire_id'])
            
    retained_ids = sorted(list(retained_ids))
    
    print(f'Number of clean samples : {len(overall_dataset)}')
    print(f'Number of clean fires : {len(retained_ids)}')

    fire_growth_filtered = fire_growth_combined[fire_growth_combined['ID'].astype(str).isin(retained_ids)]
    print('Read dataframes.')

    train_ids, val_ids, test_ids = create_stratified_splits(fire_growth_filtered, 
                                                            train_split=tc.train_split,
                                                            random_state=42,
                                                            overlap_mapper=master_mapper)
    print(len(train_ids), len(val_ids), len(test_ids))

    del dfs  # Clean up the list of dataframes
    del fire_growth_combined
    del fire_growth_filtered
    gc.collect() 
    print("DataFrames deleted to free up RAM.")

    stats_dict = calculate_h5_statistics(settings.H5_OUTPUT_FOLDER, 
                                         master_mapper,
                                         train_ids,
                                         stats_filename="timeseries_dataset_stats.json")

    common_kwargs = {
        'h5_dir': settings.H5_OUTPUT_FOLDER,
        'mapper': master_mapper,
        'seq_length': seq_length,
        'dynamic_features': dynamic_feats,
        'static_features': static_feats,
        'fire_feature': "cumuarea",
        'patch_size': settings.GRID_SIZE,
        'use_cumuarea': tc.use_cumuarea,
        'use_cumuarea_prev': tc.use_cumuarea_prev,
        'normalize': True,
        'stats_dict': stats_dict,
        'sat_nodata': np.nan,
        'use_cyclical_aspect': tc.use_cyclical_aspect
    }

    train_dataset = H5FireTimeSeriesDataset( 
        id_list=train_ids,
        **common_kwargs
    )

    val_dataset = H5FireTimeSeriesDataset(
        id_list=val_ids,
        **common_kwargs
    )

    test_dataset = H5FireTimeSeriesDataset( 
        id_list=test_ids,
        **common_kwargs 
    )
    
    print(f'Number of training samples :: {len(train_dataset)}')
    print(f'Number of validation samples :: {len(val_dataset)}')
    print(f'Number of test samples :: {len(test_dataset)}')

    # Save everything directly to your new SAMPLE_FOLDER
    save_dataset_to_disk(train_dataset, settings.TIMESERIES_SAMPLE_FOLDER, "train")
    save_dataset_to_disk(val_dataset, settings.TIMESERIES_SAMPLE_FOLDER, "val")
    save_dataset_to_disk(test_dataset, settings.TIMESERIES_SAMPLE_FOLDER, "test")
    
    print("\nAll data successfully generated and saved to disk!")
