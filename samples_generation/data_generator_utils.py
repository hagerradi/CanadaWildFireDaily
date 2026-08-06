import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pyproj import Transformer, CRS
from data_preparation.data_configs.data_settings import DEG_NAD_CRS, METER_NAD_CRS

def lonlat_to_canada_lambert(df, lon_col='lon', lat_col='lat'):
    """Convert lon/lat in EPSG:4269 (NAD83) to EPSG:3347 (NAD83 / Canada Lambert) in meters.
    Returns df with 'easting' and 'northing' columns and the target CRS object.

    Args:
      df: the CFSD dataframe
      lon_col:  the name of the longitude column (Default value = 'lon')
      lat_col:  the name of the latitude column (Default value = 'lat')

    Returns: the df with 'easting' and 'northing' columns and the target CRS object

    """
    # Safety: require lon/lat columns
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns '{lon_col}' and '{lat_col}'")

    source_crs = CRS.from_epsg(int(DEG_NAD_CRS))   # NAD83 geographic (degrees)
    target_crs = CRS.from_epsg(int(METER_NAD_CRS))   # NAD83 / Canada Lambert (meters)

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    lons = df[lon_col].to_numpy(dtype=float)
    lats = df[lat_col].to_numpy(dtype=float)

    # transform (vectorized)
    eastings, northings = transformer.transform(lons, lats)

    df['easting'] = eastings
    df['northing'] = northings
    df.attrs['target_crs'] = target_crs.to_string()

    return df, target_crs

def append_tile_coordinates(df_mapped, grid_size=256, pixel_size=90, x_col='easting', y_col='northing'):
    """Calculates the global tile IDs and local 0-255 pixel coordinates
    for every row in the dataframe, appending them as new columns.

    Args:
      df: dataframe
      grid_size:  the grid size (Default value = 256)
      pixel_size:  the pixel size (Default value = 90)
      x_col:  the name of the X column (Default value = 'easting')
      y_col:  the name of the Y column (Default value = 'northing')

    Returns: the new dataframe with the columns related to the tiles
    """
    tile_span_m = grid_size * pixel_size
    
    # Calculate Tile Columns and Rows
    df_mapped['tile_col'] = np.floor(df_mapped[x_col] / tile_span_m).astype(np.int32)
    df_mapped['tile_row'] = np.floor(df_mapped[y_col] / tile_span_m).astype(np.int32)
    df_mapped['tile_id'] = 'tile_' + df_mapped['tile_col'].astype(str) + '_' + df_mapped['tile_row'].astype(str)
    
    # Calculate Bounding boxes for each pixel's respective tile
    df_mapped['tile_x_min'] = df_mapped['tile_col'] * tile_span_m
    df_mapped['tile_y_max'] = (df_mapped['tile_row'] + 1) * tile_span_m
    
    # Calculate Local Pixel Coordinates (0 to 255)
    df_mapped['pixel_x'] = np.floor((df_mapped[x_col] - df_mapped['tile_x_min']) / pixel_size).astype(int).clip(0, grid_size - 1)
    df_mapped['pixel_y'] = np.floor((df_mapped['tile_y_max'] - df_mapped[y_col]) / pixel_size).astype(int).clip(0, grid_size - 1)

    return df_mapped

def assign_spatial_regions_by_fire_id(df, fire_id_col='ID', n_regions=10):
    """
    Partitions the map into `n_regions` using Agglomerative Clustering (Ward linkage).
    Creates naturally compact regions that group nearby fires together,
    avoiding the harsh linear cutoffs of standard K-Means.
    """
    print(f"Step 1: Calculating geographic centroids for unique '{fire_id_col}'s.")
    fire_centroids = df.groupby(fire_id_col)[['tile_col', 'tile_row']].mean().reset_index()
    
    print(f"Step 2: Clustering {len(fire_centroids)} fires into exactly {n_regions} regions...")
    clustering = AgglomerativeClustering(
        n_clusters=n_regions, 
        linkage='ward'
    )
    
    fire_centroids['cluster_id'] = clustering.fit_predict(fire_centroids[['tile_col', 'tile_row']])
    
    print("Step 3: Mapping region assignments back to all rows.")
    df_clustered = df.merge(
        fire_centroids[[fire_id_col, 'cluster_id']], 
        on=fire_id_col, 
        how='left'
    )
    
    return df_clustered

def analyze_cluster_stats(df, holdout_year=2024):
    """
    Analyzes fire duration, size, and geography for each cluster.
    Optimized to prevent RAM spikes by aggregating before filtering.
    """
    # Group by fire ID
    fire_stats = df.groupby('ID').agg(
        year=('year', 'first'),              # Keep the year for filtering
        cluster_id=('cluster_id', 'first'),
        duration=('fireday', 'nunique'),     # Number of unique active days
        max_size_ha=('cumuarea', 'max'),     # Final cumulative footprint
        center_col=('tile_col', 'mean')      # East/West geography
    ).reset_index()
    
    # Filter out the holdout year on the aggregated dataframe
    fire_stats_train = fire_stats[fire_stats['year'] != holdout_year]
    
    # Aggregate up to the Cluster level
    cluster_summary = fire_stats_train.groupby('cluster_id').agg(
        total_fires=('ID', 'count'),
        avg_duration_days=('duration', 'mean'),
        avg_fire_size_ha=('max_size_ha', 'mean'),
        median_fire_size_ha=('max_size_ha', 'median'),
        geography_col=('center_col', 'mean')
    ).reset_index()
    
    # Sort geographically from West to East
    cluster_summary = cluster_summary.sort_values('geography_col').reset_index(drop=True)
    
    return cluster_summary

def apply_spatial_buffer(df_clustered, test_clusters, fire_id_col='ID', buffer_tiles=2):
    """
    Guarantees zero neighboring tiles between train and test sets by calculating 
    tile-to-tile Chebyshev/Euclidean distances directly on grid coordinates (tile_col, tile_row).
    
    Args:
        df_clustered: Dataframe with 'tile_col', 'tile_row', 'cluster_id', and fire_id_col
        test_clusters: List of holdout cluster IDs (e.g., [3, 7])
        buffer_tiles: Distance threshold in grid tiles (e.g., 2 means adjacent/neighboring tiles)
    """
    print(f"Calculating strict tile-level buffer zone (radius = {buffer_tiles} tiles)...")
    
    # Extract unique physical tiles for Test vs Train regions
    unique_tiles = df_clustered[['tile_col', 'tile_row', 'cluster_id', fire_id_col]].drop_duplicates(subset=['tile_col', 'tile_row'])
    
    test_tiles = unique_tiles[unique_tiles['cluster_id'].isin(test_clusters)]
    train_tiles = unique_tiles[~unique_tiles['cluster_id'].isin(test_clusters)]
    
    if test_tiles.empty or train_tiles.empty:
        df_clustered['is_buffer'] = False
        return df_clustered

    test_coords = test_tiles[['tile_col', 'tile_row']].values  # Shape: (N_test_tiles, 2)
    train_coords = train_tiles[['tile_col', 'tile_row']].values  # Shape: (N_train_tiles, 2)
    
    # Compute exact coordinate differences (Δcol, Δrow)
    # Using Chebyshev distance (max(|Δcol|, |Δrow|)) to capture grid adjacency (including diagonals)
    diffs = np.abs(train_coords[:, np.newaxis, :] - test_coords[np.newaxis, :, :])
    
    # Chebyshev distance (grid steps)
    grid_dists = diffs.max(axis=2)  # Shape: (N_train_tiles, N_test_tiles)
    
    # Find minimum tile distance from each train tile to any test tile
    min_dist_per_train_tile = grid_dists.min(axis=1)
    
    # Identify training tiles that fall within the buffer radius
    buffered_train_tiles = train_tiles[min_dist_per_train_tile <= buffer_tiles]
    
    # Get all fire IDs that touch these buffered tiles
    buffered_fire_ids = buffered_train_tiles[fire_id_col].unique()
    
    # Tag the main dataframe
    df_clustered['is_buffer'] = df_clustered[fire_id_col].isin(buffered_fire_ids)
    
    print(f" -> Marked {len(buffered_train_tiles)} tiles ({len(buffered_fire_ids)} fires) as 'Buffer'.")
    print(f" -> Guaranteed separation: No training tile is within {buffer_tiles} grid steps of any test tile.")
    
    return df_clustered

def select_stratified_test_clusters(cluster_summary, df_clustered, holdout_year, fire_id_col='ID', min_fires=10, min_holdout_fires=1):
    """
    Automated algorithm to select two highly diverse test clusters based on 
    geography, fire size, and fire duration.
    
    Args:
        cluster_summary: Dataframe output from analyze_cluster_stats
        df_clustered: The main clustered dataframe containing the 'year' column
        holdout_year: The year used for temporal testing (e.g., 2024)
        fire_id_col: Column name for unique fire IDs
        min_fires: Minimum number of total fires required in a cluster
        min_holdout_fires: Minimum unique fires required in the holdout_year
        
    Returns:
        tuple: (selected_clusters_list, comparison_dataframe)
    """
    # Calculate and merge holdout fire counts
    # Count unique fires in the holdout year for each cluster
    holdout_counts = df_clustered[df_clustered['year'] == holdout_year].groupby('cluster_id')[fire_id_col].nunique().reset_index(name='holdout_fires')
    
    # Merge these counts into the summary dataframe
    summary_with_counts = cluster_summary.merge(holdout_counts, on='cluster_id', how='left')
    summary_with_counts['holdout_fires'] = summary_with_counts['holdout_fires'].fillna(0)

    # Filter out sparse clusters AND clusters missing holdout data
    eligible = summary_with_counts[
        (summary_with_counts['total_fires'] >= min_fires) & 
        (summary_with_counts['holdout_fires'] >= min_holdout_fires)
    ].copy()
    
    if len(eligible) < 2:
        raise ValueError(f"Not enough clusters found. Need >= {min_fires} total fires AND >= {min_holdout_fires} fires in {holdout_year}.")
    
    # Normalize size and duration columns (0 to 1 scale) to compare them fairly
    eligible['norm_size'] = (eligible['median_fire_size_ha'] - eligible['median_fire_size_ha'].min()) / \
                            (eligible['median_fire_size_ha'].max() - eligible['median_fire_size_ha'].min() + 1e-6)
                            
    eligible['norm_duration'] = (eligible['avg_duration_days'] - eligible['avg_duration_days'].min()) / \
                                (eligible['avg_duration_days'].max() - eligible['avg_duration_days'].min() + 1e-6)

    # Split clusters into Western (West of median longitude) and Eastern halves
    geo_median = eligible['geography_col'].median()
    west_clusters = eligible[eligible['geography_col'] <= geo_median]
    east_clusters = eligible[eligible['geography_col'] > geo_median]
    
    # Fallback if all clusters fell on one side
    if len(west_clusters) == 0 or len(east_clusters) == 0:
        west_clusters = eligible.iloc[:len(eligible)//2]
        east_clusters = eligible.iloc[len(eligible)//2:]

    # Find the pair (one West, one East) that maximizes Behavioral Distance
    max_distance = -1
    best_pair = None

    for _, west_row in west_clusters.iterrows():
        for _, east_row in east_clusters.iterrows():
            # Calculate Euclidean distance in normalized (size, duration) space
            dist = np.sqrt(
                (west_row['norm_size'] - east_row['norm_size'])**2 + 
                (west_row['norm_duration'] - east_row['norm_duration'])**2
            )
            
            if dist > max_distance:
                max_distance = dist
                best_pair = (int(west_row['cluster_id']), int(east_row['cluster_id']))
    
    print(f"Selected Test Clusters: {best_pair[0]} (West) and {best_pair[1]} (East)")
    print(f"Diversity Distance Score: {max_distance:.3f}")
    
    return list(best_pair)

def generate_experiment_splits(df, holdout_year, holdout_clusters):
    """
    Generates lists of unique fire IDs for Train, Validation, and the 3 Test scenarios.
    Uses a ROTATING SPATIOTEMPORAL VALIDATION strategy (1 unique cluster per year).
    """
    print("Generating Spatiotemporal Experiment Splits...")
    
    # Create a lookup table of the unique fires
    fire_meta = df[['ID', 'year', 'cluster_id', 'is_buffer']].drop_duplicates()
    
    # Define the 3 Test Sets
    # Test Space: 2017-2023, Unseen Regions
    test_space_mask = (fire_meta['year'] != holdout_year) & (fire_meta['cluster_id'].isin(holdout_clusters))
    test_space_ids = fire_meta[test_space_mask]['ID'].tolist()
    
    # Test Time: 2024, Known Regions
    test_time_mask = (fire_meta['year'] == holdout_year) & (~fire_meta['cluster_id'].isin(holdout_clusters))
    test_time_ids = fire_meta[test_time_mask]['ID'].tolist()
    
    # Test Time & Space: 2024, Unseen Regions
    test_spacetime_mask = (fire_meta['year'] == holdout_year) & (fire_meta['cluster_id'].isin(holdout_clusters))
    test_spacetime_ids = fire_meta[test_spacetime_mask]['ID'].tolist()
    
    # Define the Train / Val Pool
    train_val_mask = (fire_meta['year'] != holdout_year) & (~fire_meta['cluster_id'].isin(holdout_clusters)) & (fire_meta['is_buffer'] == False)
    train_val_df = fire_meta[train_val_mask].copy()
    
    # Handle ROTATING Validation Assignment
    available_years = sorted(train_val_df['year'].unique())
    available_clusters = sorted(train_val_df['cluster_id'].unique())
    
    # Map 1 unique cluster to 1 year (zip naturally stops when years run out)
    val_mapping = dict(zip(available_years, available_clusters))
    
    print("\n--- Validation Assignments ---")
    for y, c in val_mapping.items():
        print(f"Year {y} -> Validation Cluster {c}")
    
    # Label each fire as Train or Validation based on the year-cluster map
    def get_train_val(row):
        if val_mapping.get(row['year']) == row['cluster_id']:
            return 'Validation'
        return 'Train'
        
    train_val_df['split'] = train_val_df.apply(get_train_val, axis=1)
    
    val_ids = train_val_df[train_val_df['split'] == 'Validation']['ID'].tolist()
    train_ids = train_val_df[train_val_df['split'] == 'Train']['ID'].tolist()
    
    # Package into a dictionary
    splits_dict = {
        "train_ids": [str(i) for i in train_ids],
        "val_ids": [str(i) for i in val_ids],
        "test_space_ids": [str(i) for i in test_space_ids],
        "test_time_ids": [str(i) for i in test_time_ids],
        "test_spacetime_ids": [str(i) for i in test_spacetime_ids]
    }
    
    # Print summary
    print("\n" + "-" * 45)
    print("FINAL EXPERIMENT SPLIT COUNTS (FIRES):")
    for k, v in splits_dict.items():
        print(f"{k:<22}: {len(v):>6} fires")
    print("-" * 45)
        
    return splits_dict