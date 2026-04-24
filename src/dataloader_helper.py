import pandas as pd
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import json
import os
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn.functional as F

def create_stratified_splits(df, train_split, random_state=42):
    """
    Takes a dataframe of fire coordinates, calculates bounding boxes and centroids, 
    stratifies by size and geography, and returns train, val, and test ID lists.
    """
    # Group by Fire ID to extract spatial boundaries and centroids
    fire_stats = df.groupby('ID').agg(
        min_lat=('lat', 'min'),
        max_lat=('lat', 'max'),
        min_lon=('lon', 'min'),
        max_lon=('lon', 'max'),
        center_lat=('lat', 'mean'),  
        center_lon=('lon', 'mean')   
    ).reset_index()

    # Prevent 0-area calculations
    fire_stats['fire_size'] = (fire_stats['max_lat'] - fire_stats['min_lat'] + 1e-5) * \
                              (fire_stats['max_lon'] - fire_stats['min_lon'] + 1e-5)

    # Add 'Mega' to the Size Bins
    fire_stats['size_bin'] = pd.qcut(
        fire_stats['fire_size'], 
        q=[0, 0.50, 0.80, 0.95, 1.0], 
        labels=['Small', 'Med', 'Large', 'Mega'], 
        duplicates='drop'
    )

    # Create Geographic Bins
    fire_stats['lat_bin'] = pd.qcut(fire_stats['center_lat'], q=2, labels=['South', 'North'])
    fire_stats['lon_bin'] = pd.qcut(fire_stats['center_lon'], q=2, labels=['West', 'East'])

    # Combine them into a single stratify label
    fire_stats['stratify_group'] = fire_stats['size_bin'].astype(str) + "_" + \
                                   fire_stats['lat_bin'].astype(str) + "_" + \
                                   fire_stats['lon_bin'].astype(str)

    # Step 1: A group needs at least 15 members to safely survive a 70/15/15 split.
    counts = fire_stats['stratify_group'].value_counts()
    rare_mask = fire_stats['stratify_group'].isin(counts[counts < 15].index)
    fire_stats.loc[rare_mask, 'stratify_group'] = fire_stats.loc[rare_mask, 'size_bin'].astype(str)

    # If the pure size group still has < 15 fires, force it to 'Large'
    final_counts = fire_stats['stratify_group'].value_counts()
    very_rare_mask = fire_stats['stratify_group'].isin(final_counts[final_counts < 15].index)
    fire_stats.loc[very_rare_mask, 'stratify_group'] = 'Large'

    # First Split: Train vs Temp
    train_df, temp_df = train_test_split(
        fire_stats, 
        test_size=round(1.0 - train_split, 4),
        random_state=random_state, 
        stratify=fire_stats['stratify_group']
    )

    # If temp_df ends up with only 1 member for a specific group
    # Drop stratification for the second split.
    temp_counts = temp_df['stratify_group'].value_counts()
    if (temp_counts < 2).any():
        print("Warning: Some strata have < 2 members in the validation/test pool. Falling back to unstratified random split for Val/Test.")
        stratify_col = None
    else:
        stratify_col = temp_df['stratify_group']

    # Second Split: Val vs Test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=random_state, 
        stratify=stratify_col
    )

    # Extract final lists of IDs
    train_ids = train_df['ID'].tolist()
    val_ids = val_df['ID'].tolist()
    test_ids = test_df['ID'].tolist()

    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    return train_ids, val_ids, test_ids

def calculate_h5_statistics(h5_dir, train_ids, sat_nodata=0):
    """
    Scans through the training H5 files, iterating over all days and static features,
    to compute global min, max, mean, and std, ignoring NaNs and missing data.
    """
    
    # We use a nested dictionary to store our running tallies for EVERY variable
    # Structure: stats['cumuarea'] = {'sum': 0, 'sum_sq': 0, 'count': 0, 'min': inf, 'max': -inf}
    stats = defaultdict(lambda: {
        'sum': 0.0, 
        'sum_sq': 0.0, 
        'count': 0, 
        'min': float('inf'), 
        'max': float('-inf')
    })

    print(f"Processing {len(train_ids)} training files...")

    for i, fire_id in tqdm(enumerate(train_ids), total=len(train_ids)):
        
        file_path = os.path.join(h5_dir, f"fire_{fire_id}.h5")
        
        if not os.path.exists(file_path):
            continue # Skip if file doesn't exist for some reason

        with h5py.File(file_path, 'r') as f:
            
            # ==========================================
            # 1. PROCESS STATIC FEATURES (Once per file)
            # ==========================================
            static_grp = f.get('static_features')
            if static_grp is not None:
                for feat_name in static_grp.keys():
                    data = static_grp[feat_name][:]
                    
                    # Create mask (ignoring NaNs)
                    valid_mask = ~np.isnan(data)
                    valid_pixels = data[valid_mask]
                    
                    if len(valid_pixels) > 0:
                        _update_tallies(stats, feat_name, valid_pixels)

            # ==========================================
            # 2. PROCESS DYNAMIC & SATELLITE (Loop over days)
            # ==========================================
            days_grp = f.get('days')
            
            if days_grp is not None:
                
                for day_key in days_grp.keys():
                    
                    day_grp = days_grp[day_key]

                    # ========================================================
                    # QUALITY MASK CHECK
                    # ========================================================
                    # Skip this day if it contains corrupted data
                    if "quality_mask" in day_grp:
                        continue
                    # ========================================================
                    
                    # A. Dynamic Features (Floats)
                    feat_grp = day_grp.get('features')
                    if feat_grp is not None:
                        for feat_name in feat_grp.keys():
                            data = feat_grp[feat_name][:]
                            valid_mask = ~np.isnan(data)
                            valid_pixels = data[valid_mask]
                            
                            if len(valid_pixels) > 0:
                                _update_tallies(stats, feat_name, valid_pixels)

                    # B. Satellite Features
                    sat_grp = day_grp.get('satellite')
                    if sat_grp is not None:
                        bands = [k for k in sat_grp.keys() if k.startswith('s2_')]
                        for band_name in bands:
                            data = sat_grp[band_name][:]
                            
                            # If sat_nodata is NaN, use np.isnan. Otherwise, use !=
                            if np.isnan(sat_nodata):
                                valid_mask = ~np.isnan(data)
                            else:
                                valid_mask = (data != sat_nodata)
                            # -----------------------
                            
                            # APPLY SENTINEL-2 SCALING FACTOR HERE
                            # Multiply by 0.0001 during the float conversion
                            valid_pixels = data[valid_mask].astype(np.float64) * 0.0001
                            
                            if len(valid_pixels) > 0:
                                _update_tallies(stats, band_name, valid_pixels)

        # Quick progress tracker
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(train_ids)}] files processed...")

    # ==========================================
    # 3. COMPUTE FINAL METRICS
    # ==========================================
    final_dict = {}
    for feat_name, tallies in stats.items():
        N = tallies['count']
        if N == 0:
            print(f"Warning: No valid data found for {feat_name}")
            continue
            
        mean = tallies['sum'] / N
        variance = (tallies['sum_sq'] / N) - (mean ** 2)
        std = np.sqrt(max(variance, 0.0))
        
        final_dict[feat_name] = {
            'mean': float(mean),
            'std': float(std + 1e-8),
            'min': float(tallies['min']),
            'max': float(tallies['max']),
            'count': int(N)
        }

    # Save to disk
    with open('training_normalization_stats.json', 'w') as out_file:
        json.dump(final_dict, out_file, indent=4)
        
    print("Done! Statistics saved to 'training_normalization_stats.json'.")
    return final_dict

def calculate_h5_statistics_filtered(h5_dir, train_ids, patch_size=256, cloud_threshold=35.0, sat_nodata=0, max_sat_lookback_days=1):
    """
    Scans through the training H5 files to compute global statistics,
    strictly matching the filtering logic (patch_size, cloud cover, excluded last day, and satellite lookback)
    used in the PyTorch Dataset.
    """
    
    stats = defaultdict(lambda: {
        'sum': 0.0, 
        'sum_sq': 0.0, 
        'count': 0, 
        'min': float('inf'), 
        'max': float('-inf')
    })

    print(f"Processing {len(train_ids)} training files...")

    for i, fire_id in tqdm(enumerate(train_ids), total=len(train_ids)):
        
        file_path = os.path.join(h5_dir, f"fire_{fire_id}.h5")
        
        if not os.path.exists(file_path):
            continue 

        with h5py.File(file_path, 'r') as f:
            
            days_grp = f.get('days')
            
            if days_grp is None:
                continue
                
            day_keys = sorted(list(days_grp.keys()))
            
            if not day_keys:
                continue

            # ========================================================
            # FILTER 1: SIZE CHECK
            # ========================================================
            first_day_grp = days_grp[day_keys[0]]
            sample_feat = list(first_day_grp['features'].keys())[0]
            h, w = first_day_grp[f'features/{sample_feat}'].shape
            
            if h != patch_size or w != patch_size:
                continue # Skip entirely if it doesn't match grid size

            # ========================================================
            # FILTER 2: CLOUD COVER AND DATE LOGIC (Excluding last day)
            # ========================================================
            valid_day_keys = []
            
            for day_key in day_keys[:-1]:
                
                day_group = days_grp[day_key]

                # ========================================================
                # QUALITY MASK CHECK
                # ========================================================
                # If the day has corrupted data (NaNs), skip it entirely
                if "quality_mask" in day_group:
                    continue
                # ========================================================

                day_is_valid = True
                
                if cloud_threshold < 100.0:
                    fire_datetime = day_group.attrs.get("noon_utc") or \
                                    day_group.attrs.get("start_utc") or \
                                    day_group.attrs.get("end_utc")
                    
                    expected_date = None
                    if fire_datetime:
                        if isinstance(fire_datetime, bytes):
                            fire_datetime = fire_datetime.decode('utf-8')
                        expected_date = fire_datetime.split("T")[0]

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
                                        if 0 <= day_diff <= max_sat_lookback_days and cloud_pct <= cloud_threshold:
                                            has_valid_sat = True
                                            break
                                    except ValueError:
                                        pass
                                        
                            # If no acquisitions met our criteria, the day is invalid
                            if not has_valid_sat:
                                day_is_valid = False
                                
                        except json.JSONDecodeError:
                            day_is_valid = False
                    else:
                        day_is_valid = False
                
                if day_is_valid:
                    valid_day_keys.append(day_key)

            # If no days passed the filters, this file won't contribute to the dataset at all. Skip it.
            if not valid_day_keys:
                continue

            # ==========================================
            # 1. PROCESS STATIC FEATURES 
            # ==========================================
            static_grp = f.get('static_features')
            if static_grp is not None:
                for feat_name in static_grp.keys():
                    data = static_grp[feat_name][:]
                    
                    valid_mask = ~np.isnan(data)
                    valid_pixels = data[valid_mask]
                    
                    if len(valid_pixels) > 0:
                        _update_tallies(stats, feat_name, valid_pixels)

            # ==========================================
            # 2. PROCESS DYNAMIC & SATELLITE (Filtered Days Only)
            # ==========================================
            for day_key in valid_day_keys:
                day_grp = days_grp[day_key]
                
                # A. Dynamic Features (Floats)
                feat_grp = day_grp.get('features')
                if feat_grp is not None:
                    for feat_name in feat_grp.keys():
                        data = feat_grp[feat_name][:]
                        valid_mask = ~np.isnan(data)
                        valid_pixels = data[valid_mask]
                        
                        if len(valid_pixels) > 0:
                            _update_tallies(stats, feat_name, valid_pixels)

                # B. Satellite Features
                sat_grp = day_grp.get('satellite')
                if sat_grp is not None:
                    bands = [k for k in sat_grp.keys() if k.startswith('s2_')]
                    for band_name in bands:
                        data = sat_grp[band_name][:]
                        
                        if np.isnan(sat_nodata):
                            valid_mask = ~np.isnan(data)
                        else:
                            valid_mask = (data != sat_nodata)
                        
                        # APPLY SENTINEL-2 SCALING FACTOR (Multiply by 0.0001)
                        valid_pixels = data[valid_mask].astype(np.float64) * 0.0001
                        
                        if len(valid_pixels) > 0:
                            _update_tallies(stats, band_name, valid_pixels)

    # ==========================================
    # 3. COMPUTE FINAL METRICS
    # ==========================================
    final_dict = {}
    for feat_name, tallies in stats.items():
        N = tallies['count']
        if N == 0:
            print(f"Warning: No valid data found for {feat_name}")
            continue
            
        mean = tallies['sum'] / N
        variance = (tallies['sum_sq'] / N) - (mean ** 2)
        std = np.sqrt(max(variance, 0.0))
        
        final_dict[feat_name] = {
            'mean': float(mean),
            'std': float(std + 1e-8), 
            'min': float(tallies['min']),
            'max': float(tallies['max']),
            'count': int(N)
        }

    with open('training_normalization_filtered_stats.json', 'w') as out_file:
        json.dump(final_dict, out_file, indent=4)
        
    print("Done! Statistics saved to 'training_normalization_filtered_stats.json'.")
    return final_dict

def calculate_h5_statistics_dual(h5_dir, train_ids, patch_size=256, cloud_threshold=35.0, 
                                     sat_nodata=np.nan, max_sat_lookback_days=30, 
                                     sequence_length=3, use_forecast=False):
    """
    Scans through the training H5 files to compute global statistics.
    EXCLUDES Satellite images and NDVI/EVI from the calculations.
    """
    
    stats = defaultdict(lambda: {
        'sum': 0.0, 
        'sum_sq': 0.0, 
        'count': 0, 
        'min': float('inf'), 
        'max': float('-inf')
    })

    print(f"Processing {len(train_ids)} training files for stats computation...")

    for fire_id in tqdm(train_ids, desc="Calculating Stats"):
        
        file_path = os.path.join(h5_dir, f"fire_{fire_id}.h5")
        
        if not os.path.exists(file_path):
            continue 

        with h5py.File(file_path, 'r') as f:
            
            days_grp = f.get('days')
            if days_grp is None:
                continue
                
            day_keys = sorted(list(days_grp.keys()))
            if len(day_keys) < sequence_length + 1:
                continue

            # ========================================================
            # FILTER 1: SIZE CHECK
            # ========================================================
            first_day_grp = days_grp[day_keys[0]]
            sample_feat = list(first_day_grp['features'].keys())[0]
            h, w = first_day_grp[f'features/{sample_feat}'].shape
            
            if h != patch_size or w != patch_size:
                continue 

            # ========================================================
            # MOCK DATALOADER: 1. Build the Satellite Catalog
            # ========================================================
            catalog = []
            seen_acq_dates = set()
            
            for key in day_keys:
                day_group = days_grp[key]
                if "satellite" not in day_group or "cloud_cover_stats" not in day_group["satellite"].attrs:
                    continue
                    
                stats_attr = day_group["satellite"].attrs["cloud_cover_stats"]
                if isinstance(stats_attr, bytes): stats_attr = stats_attr.decode('utf-8')
                    
                try:
                    stats_list = json.loads(stats_attr)
                    acq_date_str = stats_list[0].get("acquisition_date")
                    if not acq_date_str or acq_date_str in seen_acq_dates:
                        continue
                        
                    all_tile_ccs = [stat.get("cloud_cover_pct", 100.0) for stat in stats_list]
                    stitched_cc = max(all_tile_ccs) 
                    
                    if stitched_cc <= cloud_threshold:
                        catalog.append({
                            'day_key': key,
                            'acq_date_str': acq_date_str,
                            'acq_date_obj': datetime.strptime(acq_date_str, "%Y-%m-%d")
                        })
                        seen_acq_dates.add(acq_date_str)
                except (json.JSONDecodeError, TypeError, ValueError, IndexError):
                    continue
                    
            catalog.sort(key=lambda x: x['acq_date_obj'])

            # ========================================================
            # MOCK DATALOADER: 2. Find Valid Sequences
            # ========================================================
            used_dyn_keys = set()
            valid_fire = False
            
            for idx in range(len(day_keys) - sequence_length):
                skip_sequence = False
                for step_offset in range(sequence_length):
                    if "quality_mask" in days_grp[day_keys[idx + step_offset]]:
                        skip_sequence = True
                        break
                if skip_sequence: continue
                    
                target_key = day_keys[idx + sequence_length]
                if use_forecast and "quality_mask" in days_grp[target_key]:
                    continue

                last_seq_key = day_keys[idx + sequence_length - 1]
                last_seq_group = days_grp[last_seq_key]
                fire_datetime = last_seq_group.attrs.get("noon_utc") or last_seq_group.attrs.get("start_utc") or last_seq_group.attrs.get("end_utc")
                
                last_seq_date_str = None
                if fire_datetime:
                    if isinstance(fire_datetime, bytes): fire_datetime = fire_datetime.decode('utf-8')
                    last_seq_date_str = fire_datetime.split("T")[0]

                best_sat_key = None
                if last_seq_date_str:
                    try:
                        last_seq_date_obj = datetime.strptime(last_seq_date_str, "%Y-%m-%d")
                        for sat_item in reversed(catalog):
                            if sat_item['acq_date_obj'] <= last_seq_date_obj:
                                current_gap = (last_seq_date_obj - sat_item['acq_date_obj']).days
                                if current_gap <= max_sat_lookback_days:
                                    best_sat_key = sat_item['day_key']
                                    break
                    except ValueError:
                        pass

                # If the sequence survives the checks and finds a sat map, mark its days as USED
                if best_sat_key is not None:
                    valid_fire = True
                    for step_offset in range(sequence_length):
                        used_dyn_keys.add(day_keys[idx + step_offset])
                    if use_forecast:
                        used_dyn_keys.add(target_key) 

            if not valid_fire:
                continue 

            # ========================================================
            # 3. ACCUMULATE STATIC FEATURES
            # ========================================================
            static_grp = f.get('static_features')
            if static_grp is not None:
                for feat_name in static_grp.keys():
                    data = static_grp[feat_name][:]
                    valid_mask = ~np.isnan(data)
                    valid_pixels = data[valid_mask]
                    if len(valid_pixels) > 0:
                        _update_tallies(stats, feat_name, valid_pixels)

            # ========================================================
            # 4. ACCUMULATE DYNAMIC FEATURES (EXCLUDING NDVI/EVI)
            # ========================================================
            for day_key in used_dyn_keys:
                feat_grp = days_grp[day_key].get('features')
                if feat_grp is not None:
                    for feat_name in feat_grp.keys():
                        
                        # --- THE NEW EXCLUSION CHECK ---
                        # Skip ndvi, evi, and raw fire masks
                        if feat_name in ['ndvi', 'evi', 'firearea', 'cumuarea']:
                            continue
                            
                        data = feat_grp[feat_name][:]
                        valid_mask = ~np.isnan(data)
                        valid_pixels = data[valid_mask]
                        if len(valid_pixels) > 0:
                            _update_tallies(stats, feat_name, valid_pixels)

            # (Section 5: Satellite features completely removed)

    # ==========================================
    # 6. COMPUTE FINAL METRICS
    # ==========================================
    final_dict = {}
    for feat_name, tallies in stats.items():
        N = tallies['count']
        if N == 0:
            print(f"Warning: No valid data found for {feat_name}")
            continue
            
        mean = tallies['sum'] / N
        variance = (tallies['sum_sq'] / N) - (mean ** 2)
        std = np.sqrt(max(variance, 0.0))
        
        final_dict[feat_name] = {
            'mean': float(mean),
            'std': float(std + 1e-8), 
            'min': float(tallies['min']),
            'max': float(tallies['max']),
            'count': int(N)
        }

    with open('training_normalization_dual_stats.json', 'w') as out_file:
        json.dump(final_dict, out_file, indent=4)
        
    print("Done! Statistics saved to 'training_normalization_dual_stats.json'.")
    
    return final_dict

def _update_tallies(stats_dict, feat_name, valid_pixels):
    """Helper function to update the running sums and min/max."""
    stats_dict[feat_name]['sum'] += np.sum(valid_pixels)
    stats_dict[feat_name]['sum_sq'] += np.sum(valid_pixels ** 2)
    stats_dict[feat_name]['count'] += len(valid_pixels)
    
    current_min = np.min(valid_pixels)
    current_max = np.max(valid_pixels)
    
    if current_min < stats_dict[feat_name]['min']:
        stats_dict[feat_name]['min'] = current_min
    if current_max > stats_dict[feat_name]['max']:
        stats_dict[feat_name]['max'] = current_max

def smooth_fire_mask(mask_2d: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Applies Binary Closing (Dilation followed by Erosion) 
    to fill holes in a binary mask without inflating the perimeter.
    """
    # Add batch and channel dimensions for pooling: (H, W) -> (1, 1, H, W)
    m = mask_2d.unsqueeze(0).unsqueeze(0).float()
    padding = kernel_size // 2
    
    # DILATE: Inflate to fill the internal holes
    dilated = F.max_pool2d(m, kernel_size=kernel_size, stride=1, padding=padding)
    
    # ERODE: Shrink the outer boundary back to normal (-MaxPool of -X)
    eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=padding)
    
    # Return as a boolean 2D mask
    return (eroded.squeeze() > 0.5)