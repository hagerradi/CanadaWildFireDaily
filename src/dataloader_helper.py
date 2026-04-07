import pandas as pd
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import json
import os
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

def create_stratified_splits(df, train_split, random_state=42):
    """
    Takes a dataframe of fire coordinates, calculates bounding boxes and centroids, 
    stratifies by size and geography, and returns train, val, and test ID lists.
    
    Args:
        df (pd.DataFrame): Raw dataframe containing 'ID', 'lat', and 'lon' columns.
        random_state (int): Seed for reproducibility.
        
    Returns:
        tuple: (train_ids, val_ids, test_ids) as lists.
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

    fire_stats['size_bin'] = pd.qcut(
        fire_stats['fire_size'], 
        q=[0, 0.50, 0.80, 0.95, 1.0], 
        labels=['Small', 'Med', 'Large', 'Mega'], 
        duplicates='drop'
    )

    # Create Geographic Bins (Divide the map into 4 quadrants)
    fire_stats['lat_bin'] = pd.qcut(fire_stats['center_lat'], q=2, labels=['South', 'North'])
    fire_stats['lon_bin'] = pd.qcut(fire_stats['center_lon'], q=2, labels=['West', 'East'])

    # Combine them into a single stratify label (e.g., "Mega_North_West")
    fire_stats['stratify_group'] = fire_stats['size_bin'].astype(str) + "_" + \
                                   fire_stats['lat_bin'].astype(str) + "_" + \
                                   fire_stats['lon_bin'].astype(str)

    # A group needs at least 10 members to survive being split TWICE.
    # If it has < 10, strip the geography and just stratify it by its pure size.
    counts = fire_stats['stratify_group'].value_counts()
    rare_mask = fire_stats['stratify_group'].isin(counts[counts < 10].index)
    fire_stats.loc[rare_mask, 'stratify_group'] = fire_stats.loc[rare_mask, 'size_bin'].astype(str)

    # If the pure size group (e.g., 'Mega') STILL has < 10 fires in total, 
    # force it into the 'Large' bin so the train_test_split executes successfully.
    final_counts = fire_stats['stratify_group'].value_counts()
    very_rare_mask = fire_stats['stratify_group'].isin(final_counts[final_counts < 10].index)
    fire_stats.loc[very_rare_mask, 'stratify_group'] = 'Large'
    # -------------------------------------------------------

    # First Split: 70% Train, 30% Temp
    train_df, temp_df = train_test_split(
        fire_stats, 
        # test_size=0.30, 
        test_size=round(1.0 - train_split, 4),
        random_state=random_state, 
        stratify=fire_stats['stratify_group']
    )

    # Second Split: 15% Val, 15% Test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=random_state, 
        stratify=temp_df['stratify_group']
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
    
    # Use a nested dictionary to store our running tallies for EVERY variable
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
                    
                    # A. Dynamic Features (Floats)
                    feat_grp = day_grp.get('features')
                    if feat_grp is not None:
                        for feat_name in feat_grp.keys():
                            data = feat_grp[feat_name][:]
                            valid_mask = ~np.isnan(data)
                            valid_pixels = data[valid_mask]
                            
                            if len(valid_pixels) > 0:
                                _update_tallies(stats, feat_name, valid_pixels)

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
                            
                            # APPLY SENTINEL-2 SCALING FACTOR
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
        # Ensure variance isn't negative due to floating point inaccuracies
        std = np.sqrt(max(variance, 0.0))
        
        final_dict[feat_name] = {
            'mean': float(mean),
            'std': float(std + 1e-8), # Add tiny epsilon to prevent div by zero later
            'min': float(tallies['min']),
            'max': float(tallies['max']),
            'count': int(N)
        }

    # Save it to disk so we only ever run this once
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
                                            break # We found a good one, stop checking!
                                    except ValueError:
                                        pass # Ignore if date format is weird
                                        
                            # If no acquisitions met the criteria, the day is invalid
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