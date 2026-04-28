import sys
import argparse
import pandas as pd

from configs import settings
from data_preparation.raw_data_preprocessing.topography_rasters import process_fire
from data_preparation.features_generation.helpers import lonlat_to_canada_lambert

def run_local(all_fire_ids, fire_growth_pts):
    """Processes all fires sequentially for local/generic execution."""
    total_fires = len(all_fire_ids)
    print(f"\n=== LOCAL MODE: Processing ALL {total_fires} fires sequentially ===")
    
    for i, target_fire_id in enumerate(all_fire_ids, start=1):
        print(f"\n[{i}/{total_fires}] Processing Fire ID: {target_fire_id}")
        process_fire(target_fire_id, fire_growth_pts)

def run_distributed(all_fire_ids, fire_growth_pts, task_id):
    """Processes a single fire based on the SLURM Array Task ID."""
    total_fires = len(all_fire_ids)
    
    if task_id < 0 or task_id >= total_fires:
        print(f"[!] Error: Task ID {task_id} is out of bounds (0 to {total_fires - 1}). Exiting.")
        sys.exit(1)
        
    target_fire_id = all_fire_ids[task_id]
    print(f"\n=== CLUSTER MODE: Task {task_id} / {total_fires - 1} ===")
    print(f"Processing Fire ID: {target_fire_id}")
    
    process_fire(target_fire_id, fire_growth_pts)

if __name__ == "__main__":
    # Setup standard argument parsing for GitHub users
    parser = argparse.ArgumentParser(description="Generate Terrain Tiles for Wildfire Dataset.")
    parser.add_argument("year", type=str, help="The target year to process (e.g., 2024)")
    parser.add_argument("--mode", type=str, choices=["local", "distributed"], default="local", 
                        help="Execution mode: 'local' (sequential) or 'distributed' (parallel array task)")
    parser.add_argument("--task-id", type=int, default=-1, 
                        help="SLURM Array Task ID (Required if mode is 'cluster')")
    
    args = parser.parse_args()

    # Load the dataset
    csv_path = f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{args.year}/Firegrowth_pts_v1_1_{args.year}.csv'
    try:
        fire_growth_pts = pd.read_csv(csv_path)
        fire_growth_pts, _ = lonlat_to_canada_lambert(fire_growth_pts)
    except FileNotFoundError:
        print(f"[!] Error: Could not find CSV at {csv_path}")
        sys.exit(1)
    
    all_fire_ids = sorted(fire_growth_pts['ID'].unique())

    # Route to the correct execution path
    if args.mode == "local":
        run_local(all_fire_ids, fire_growth_pts)
    elif args.mode == "distributed":
        if args.task_id == -1:
            print("[!] Error: You must provide a --task-id when using cluster mode.")
            sys.exit(1)
        run_distributed(all_fire_ids, fire_growth_pts, args.task_id)