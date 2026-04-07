from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import pystac_client

# Add project root to sys.path to allow 'from configs import settings'
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from configs import settings
import satellite_sentinel

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

def main():
    # Get the Task ID from Slurm
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    # Check if both year and path are provided
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

    all_fire_ids = get_fire_ids(target_year, npy_path)
    print(f'Target Year :: {target_year}')
    print(f'Path :: {npy_path}')
    print(f'Number of IDs :: {len(all_fire_ids)}')

    if task_id >= len(all_fire_ids):
        print(f"Task ID {task_id} is out of range. Total fires: {len(all_fire_ids)}")
        return

    fire_id = all_fire_ids[task_id]
    h5_path = Path(f"{settings.H5_OUTPUT_FOLDER}/fire_{fire_id}.h5")

    # Verify the H5 file exists before starting
    if not h5_path.exists():
        print(f"!!! Error: {h5_path} not found. Ensure Grid pipeline ran first.")
        sys.exit(1)

    print(f"--- [Task {task_id}] Starting Satellite Pipeline for Fire: {fire_id} ---")

    # Setup Planetary Computer Client
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )
    
    # All Sentinel-2 bands except B10
    selected_bands = [f"B{i:02}" for i in range(1, 13) if i != 10]

    success_marker = Path(settings.SATELLITE_STATUS_FOLDER) / f"success_{fire_id}.txt"
    fail_marker = Path(settings.SATELLITE_STATUS_FOLDER) / f"fail_{fire_id}.txt"

    if success_marker.exists():
        print(f"--- [Task {task_id}] Skipping Fire {fire_id}: Already exists in {settings.SATELLITE_STATUS_FOLDER} ---")
        return
    
    if fail_marker.exists():
        fail_marker.unlink()
    
    # Run the Pipeline
    try:
        satellite_sentinel.run_s2_h5_pipeline_v2(
            h5_path=h5_path,
            catalog=catalog,
            bands=selected_bands
        )
        success_marker.touch()
        print(f"--- [Task {task_id}] Successfully finished Fire: {fire_id} ---")
    except Exception as e:
        fail_marker.touch()
        print(f"!!! [Task {task_id}] Failed Fire {fire_id}: {str(e)}")
        sys.exit(1) 

if __name__ == "__main__":
    main()