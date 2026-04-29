import pandas as pd
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import json
import os
from collections import defaultdict
from tqdm import tqdm

def create_stratified_splits(df, train_split, random_state=42, overlap_mapper=None):
    """
    Takes a dataframe of fire coordinates, strictly groups overlapping fires to prevent 
    data leakage, calculates combined bounding boxes/centroids, stratifies by size 
    and geography, and returns train, val, and test ID lists.
    """
    df_copy = df.copy()
    
    # ---------------------------------------------------------
    # UNION-FIND: Group overlapping fires into "Super-Fires"
    # ---------------------------------------------------------
    unique_fires = df_copy['ID'].astype(str).unique()
    
    # Initialize Union-Find dictionary (every fire is its own parent initially)
    parent = {f: f for f in unique_fires}
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Use the mapper to link fires that share a tile
    if overlap_mapper:
        for fires_list in overlap_mapper.values():
            fire_ids = [str(f['fire_id']) for f in fires_list if str(f['fire_id']) in parent]
            if len(fire_ids) > 1:
                first = fire_ids[0]
                for other in fire_ids[1:]:
                    union(first, other)

    # Assign every fire to its root "Super-Fire" ID
    df_copy['super_id'] = df_copy['ID'].astype(str).apply(lambda x: find(x))
    
    num_super_fires = df_copy['super_id'].nunique()
    print(f"Grouped {len(unique_fires)} individual fires into {num_super_fires} isolated Super-Fires to prevent data leaks.")

    # ---------------------------------------------------------
    # CALCULATE STRATIFICATION STATS ON SUPER-FIRES
    # ---------------------------------------------------------
    # Group by super_id so overlapping fires form one giant bounding box
    fire_stats = df_copy.groupby('super_id').agg(
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

    # A group needs at least 15 members to safely survive a 70/15/15 split.
    counts = fire_stats['stratify_group'].value_counts()
    rare_mask = fire_stats['stratify_group'].isin(counts[counts < 15].index)
    fire_stats.loc[rare_mask, 'stratify_group'] = fire_stats.loc[rare_mask, 'size_bin'].astype(str)

    # If the pure size group still has < 15 fires, force it to 'Large'
    final_counts = fire_stats['stratify_group'].value_counts()
    very_rare_mask = fire_stats['stratify_group'].isin(final_counts[final_counts < 15].index)
    fire_stats.loc[very_rare_mask, 'stratify_group'] = 'Large'

    # ---------------------------------------------------------
    # SPLIT THE SUPER-FIRES
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # UNPACK BACK TO INDIVIDUAL FIRE IDs
    # ---------------------------------------------------------
    train_super_ids = train_df['super_id'].tolist()
    val_super_ids = val_df['super_id'].tolist()
    test_super_ids = test_df['super_id'].tolist()

    train_ids = df_copy[df_copy['super_id'].isin(train_super_ids)]['ID'].unique().tolist()
    val_ids = df_copy[df_copy['super_id'].isin(val_super_ids)]['ID'].unique().tolist()
    test_ids = df_copy[df_copy['super_id'].isin(test_super_ids)]['ID'].unique().tolist()

    print(f"Final Individual Fire Split -> Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    return train_ids, val_ids, test_ids


def _update_tallies(stats, feat_name, valid_pixels):
    """Helper function to cleanly update running sums (using float64 to prevent overflow)."""
    # Force float64 math
    stats[feat_name]['sum'] += np.sum(valid_pixels, dtype=np.float64)
    stats[feat_name]['sum_sq'] += np.sum(valid_pixels ** 2, dtype=np.float64)
    stats[feat_name]['count'] += valid_pixels.size
    
    current_min = np.min(valid_pixels)
    current_max = np.max(valid_pixels)
    if current_min < stats[feat_name]['min']:
        stats[feat_name]['min'] = current_min
    if current_max > stats[feat_name]['max']:
        stats[feat_name]['max'] = current_max

def calculate_h5_statistics(h5_dir, mapper, train_ids, patch_size=256, stats_filename='dataset_stats.json'):
    """
    Scans through H5 files based on UNIQUE (Tile, DOB) pairs to compute global stats.
    - Prevents double-counting overlapping fires.
    - Explicitly skips Satellite, NDVI, and EVI for stats generation.
    """
    
    stats = defaultdict(lambda: {
        'sum': 0.0, 
        'sum_sq': 0.0, 
        'count': 0, 
        'min': float('inf'), 
        'max': float('-inf')
    })

    # Track which static tiles we have already processed to avoid double counting
    processed_static_tiles = set()
    
    # Filter the mapper to ONLY include (Tile, DOB) keys that belong to training fires
    train_ids_set = set([str(x) for x in train_ids])
    
    valid_global_keys = []
    for global_key, fires_list in mapper.items():
        # If at least one fire in this tile/day is in the train set, we process the environment
        if any(str(f['fire_id']) in train_ids_set for f in fires_list):
            valid_global_keys.append((global_key, fires_list))

    print(f"Processing {len(valid_global_keys)} unique Tile-Day environments...")

    for global_key, fires_list in tqdm(valid_global_keys, desc="Calculating Stats"):
        
        # Unpack the global key (e.g., tile_200_126_DOB_2024_195)
        parts = global_key.split('_DOB_')
        tile_id = parts[0]
        
        # We just need to open ONE fire's H5 file to read the environment for this tile/day
        primary_fire = fires_list[0]
        fire_id = primary_fire['fire_id']
        day_key = primary_fire['day_key']
        
        file_path = os.path.join(h5_dir, f"fire_{fire_id}.h5")
        if not os.path.exists(file_path):
            continue
            
        # ========================================================
        # OPEN AND AUTO-CLOSE THE FILE
        # ========================================================
        try:
            with h5py.File(file_path, 'r', swmr=True) as f:
                
                # --- SIZE CHECK & QUALITY CHECK ---
                if f"{tile_id}/days/{day_key}" not in f:
                    continue
                    
                day_group = f[f"{tile_id}/days/{day_key}"]
                
                sample_feat = list(day_group['features'].keys())[0]
                h, w = day_group[f'features/{sample_feat}'].shape
                if h != patch_size or w != patch_size:
                    continue 
                    
                if "quality_mask" in day_group:
                    continue

                # --- PROCESS STATIC FEATURES (Run once per Tile) ---
                if tile_id not in processed_static_tiles and f"{tile_id}/static_features" in f:
                    static_grp = f[f"{tile_id}/static_features"]
                    for feat_name in static_grp.keys():
                        data = static_grp[feat_name][:]
                        valid_mask = (~np.isnan(data)) & (data != -9999.0)
                        
                        # Cast to float64
                        valid_pixels = data[valid_mask].astype(np.float64)
                        
                        if len(valid_pixels) > 0:
                            _update_tallies(stats, feat_name, valid_pixels)
                            
                    processed_static_tiles.add(tile_id)

                # --- PROCESS DYNAMIC FEATURES ---
                feat_grp = day_group.get('features')
                if feat_grp is not None:
                    for feat_name in feat_grp.keys():
                        
                        # SKIP NDVI AND EVI
                        if feat_name.lower() in ['ndvi', 'evi']:
                            continue
                            
                        data = feat_grp[feat_name][:]
                        valid_mask = ~np.isnan(data)
                        valid_pixels = data[valid_mask].astype(np.float64)
                        
                        if len(valid_pixels) > 0:
                            _update_tallies(stats, feat_name, valid_pixels)
        
        except OSError:
            # Safely skips if a specific file happens to be corrupted
            continue

    # ==========================================
    # COMPUTE FINAL METRICS
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

    with open(stats_filename, 'w') as out_file:
        json.dump(final_dict, out_file, indent=4)
        
    print(f"Done! Statistics saved to {stats_filename}.")
    return final_dict