from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from configs import settings
from utils import helpers
import grid_generation, weather, fuel_scanfi, fuel_viirs, topography

TOPO_VARS = {
    'dem_avg': settings.ELEVATION_AVG_FOLDER,
    'slope': settings.SLOPE_FOLDER,
    'aspect': settings.ASPECT_FOLDER
}

# Configuration
CHUNK_SIZE = 10
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

def get_fire_ids(target_year, npy_path=None):
    """
    Retrieves fire IDs with a focus on speed.
    - Prioritizes loading from a .npy file to avoid heavy CSV parsing.
    - Filters IDs to ensure they contain the target_year string.
    """
    all_fire_ids = []
    
    # Load the target ids
    if npy_path and os.path.exists(npy_path):
        print(f"Loading from numpy: {npy_path}")
        all_fire_ids = np.load(npy_path).tolist()
    
    # Fallback to CSV only if needed
    else:
        print(f"Numpy path not found. Loading CSV for year {target_year}...")
        csv_path = f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{target_year}/Firegrowth_pts_v1_1_{target_year}.csv'
        
        df_ids = pd.read_csv(csv_path, usecols=['ID'])
        all_fire_ids = df_ids['ID'].unique().tolist()

    # Filter for IDs containing the target year
    filtered_ids = sorted([str(fid) for fid in all_fire_ids if str(fid).startswith(f"{target_year}_")])
    
    return filtered_ids

if __name__ == "__main__":

    if len(sys.argv) > 2:
        target_year = str(sys.argv[1])
        npy_path = str(sys.argv[2])
    # Check if only the year is provided
    elif len(sys.argv) > 1:
        target_year = str(sys.argv[1])
        npy_path = None
    # Default fallback
    else:
        target_year = '2020'
        npy_path = None

    print(settings.GRID_SIZE)

    # Load the full list of IDs
    fire_growth_pts = pd.read_csv(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{target_year}/Firegrowth_pts_v1_1_{target_year}.csv')
    all_fire_ids = get_fire_ids(target_year, npy_path)
    print(f'Target Year :: {target_year}')
    print(f'Path :: {npy_path}')
    print(f'Number of IDs :: {len(all_fire_ids)}')

    # Calculate which slice of the list this task handles
    start_idx = task_id * CHUNK_SIZE
    end_idx = start_idx + CHUNK_SIZE
    my_fires = all_fire_ids[start_idx:end_idx]

    print(f"Task {task_id} processing {len(my_fires)} fires: {my_fires}")

    # Loop through the 10 fires assigned to THIS task
    for fire_id in my_fires:
        print(f"--- Starting Fire: {fire_id} ---")
        
        # Filter DF for this specific fire
        fire_df = fire_growth_pts[fire_growth_pts['ID'] == fire_id].copy()
        fire_df, _ = helpers.lonlat_to_canada_lambert(fire_df)

        # Grid Construction
        grid_params = grid_generation.get_global_grid_params(fire_df, fire_id, settings.GRID_SIZE, settings.PIXEL_SIZE, settings.X_COL, settings.Y_COL)
        grid_generation.initialize_fire_h5(settings.H5_OUTPUT_FOLDER, fire_id, target_year, grid_params, pixel_size=settings.PIXEL_SIZE, fill_value=np.nan)
        
        for fireday in sorted(fire_df['fireday'].unique()):
            grid_generation.add_fire_day_to_h5(f"{settings.H5_OUTPUT_FOLDER}/fire_{fire_id}.h5", 
                                                fire_df, fire_id, fireday, grid_params, 
                                                pixel_size=settings.PIXEL_SIZE)

        h5_path = Path(settings.H5_OUTPUT_FOLDER) / f"fire_{fire_id}.h5"
        
        # Generate the environemental variables
        weather.run_single_h5_era5_pipeline(h5_path, settings.ERA5_FOLDER)
        topography.run_static_topography_pipeline(h5_path, TOPO_VARS)
        fuel_scanfi.run_single_fire_scanfi(h5_path, settings.SCANFI_FOLDER)
        fuel_viirs.run_single_fire_viirs(h5_path)

        print(f"--- Finished Fire: {fire_id} ---")

    print(f"Task {task_id} complete.")