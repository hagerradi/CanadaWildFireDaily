import odc.stac
import planetary_computer
import numpy as np
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import dask
from dask.diagnostics import ProgressBar
import h5py
import json
import planetary_computer
from odc.geo.geobox import GeoBox
from affine import Affine

def get_fire_master_cube(h5_path, catalog, bands):
    """
    Downloads the entire satellite history for a fire's duration 
    into a single 4D NumPy array (Time, Bands, grid_size, grid_size).
    Optimized with odc-stac for Just-In-Time token signing and stable downloads.
    """
    with h5py.File(h5_path, 'r') as f:
        # Get exact spatial bounds for EPSG:3347
        e = f['geometry/theoretical_easting'][:]
        n = f['geometry/theoretical_northing'][:]

        transform = Affine.translation(e.min(), n.max()) * Affine.scale(90, -90)
        height, width = e.shape
        geobox = GeoBox(shape=(height, width), affine=transform, crs="EPSG:3347")
        print(f"Confirmed: Shape is {geobox.shape}")
        print(f"Confirmed: Resolution is {geobox.resolution}")
        
        # Extract lon/lat for STAC search
        lons = f['geometry/theoretical_lon'][:]
        lats = f['geometry/theoretical_lat'][:]
        min_lon, max_lon = lons.min(), lons.max()
        min_lat, max_lat = lats.min(), lats.max()
        
        # Get temporal bounds
        day_keys = sorted(f['days'].keys())
        fire_year = int(f.attrs['year'])
        start_dob = int(f['days'][day_keys[0]].attrs['DOB'])
        end_dob = int(f['days'][day_keys[-1]].attrs['DOB'])

    # Search window: start 1 month (30 days) before fire, end 2 days after
    max_lookback_days = 30
    start_date = datetime(fire_year, 1, 1) + timedelta(days=start_dob - max_lookback_days)
    end_date = datetime(fire_year, 1, 1) + timedelta(days=end_dob + 1)

    print(f'Fire {h5_path} metadata acquired.')

    # ---------------------------------------------------------
    # STAC API Retry Loop
    # ---------------------------------------------------------
    max_api_retries = 5
    items = []
    
    for attempt in range(max_api_retries):
        try:
            print(f"Querying Planetary Computer (Attempt {attempt + 1})...")
            
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=[min_lon - 0.1, min_lat - 0.1, max_lon + 0.1, max_lat + 0.1],
                datetime=f"{start_date.date()}/{end_date.date()}",
                # query={"eo:cloud_cover": {"lte": threshold}}
            )
            items = list(search.items())
            
            print(f"Search done! Found {len(items)} items.")
            break  
            
        except Exception as e:
            print(f"\n[!] STAC API timeout/error on attempt {attempt + 1}")
            if attempt < max_api_retries - 1:
                wait_time = (attempt + 1) * 15  
                print(f"Azure Front Door is congested. Waiting {wait_time}s to retry...")
                time.sleep(wait_time)
            else:
                print("Max STAC API retries reached. Moving to next task.")
                raise e 

    if not items:
        return None, {}

    dataset = odc.stac.load(
        items,
        bands=bands,
        geobox=geobox,
        patch_url=planetary_computer.sign,
        groupby="id",
        chunks={'time': 1, 'x': width, 'y': height}
    )

    # Cast to float32 and convert to (Time, Band, Y, X) DataArray
    stack_da = dataset.astype(np.float32).to_array(dim="band").transpose("time", "band", "y", "x")

    print(f"Stacking done. Stack Shape: {stack_da.shape}")
    print(f"Estimated Size in RAM: {stack_da.nbytes / (1024**2):.2f} MB")

    # ---------------------------------------------------------
    # Dask Threading (Download Compute with Retries)
    # ---------------------------------------------------------
    print(f"Downloading Master Cube: {stack_da.shape} (Time, Bands, H, W)...")
    
    max_retries = 3
    cube = None
    download_success = False
    
    with dask.config.set(scheduler='threads', num_workers=8):
        
        for attempt in range(max_retries):
            try:
                print(f"Download Attempt {attempt + 1}...")
                with ProgressBar():
                    cube = stack_da.compute()            
                
                if cube is not None:
                    print("Download successful!")
                    download_success = True
                    break 
                    
            except Exception as e:
                print(f"\n[!] Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1) 
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("[!] Max optimistic retries exhausted. Preparing for fallback...")
                    # raise e
        
        # THE 404 OVERHEAD FALLBACK (Only runs if the optimistic attempts failed)
        if not download_success:
            print("\n[!] Standard download failed. Initiating 404 overhead scan to drop broken scenes...")
            
            valid_items = []
            for item in tqdm(items, desc="Checking Items"):
                try:
                    # 1x1 pixel slice to force odc-stac to check file existence
                    test_ds = odc.stac.load(
                        [item], 
                        bands=bands, 
                        geobox=geobox, 
                        patch_url=planetary_computer.sign,
                        chunks={} 
                    ).isel(x=slice(0, 1), y=slice(0, 1)).compute()
                    test_ds.compute()
                    valid_items.append(item)
                except Exception as e:
                    print(f"\n[!] SKIPPING BROKEN ITEM: {item.id}")
            
            if not valid_items:
                print("All items in this search returned 404 or errors.")
                return None, {}
            
            # Rebuild the dataset and stack with ONLY healthy items
            items = valid_items
            print(f"Rebuilding lazy cube with {len(items)} healthy items...")
            
            dataset_clean = odc.stac.load(
                items,
                bands=bands,
                geobox=geobox,
                patch_url=planetary_computer.sign,
                groupby="id",
                chunks={'time': 1, 'x': width, 'y': height}
            )
            stack_da_clean = dataset_clean.astype(np.float32).to_array(dim="band").transpose("time", "band", "y", "x")
            
            # One final compute
            print("Attempting to download cleaned Master Cube...")
            try:
                with ProgressBar():
                    cube = stack_da_clean.compute()
                print("Cleaned download successful!")
            except Exception as e:
                print("Cleaned download also failed. Max retries exhausted.")
                raise e
                    
    print(f'Download done.')
    
    # ---------------------------------------------------------
    # Map Metadata
    # ---------------------------------------------------------
    metadata = {
        "dates": [np.datetime64(i.datetime.date(), 'D') for i in items],
        "cloud_covers": np.array([i.properties.get("eo:cloud_cover", 100) for i in items]),
        "item_ids": np.array([i.id for i in items])
    }

    return cube.values if cube is not None else None, metadata

def process_h5_from_cube(h5_path, cube, metadata, bands):
    """
    Slices the pre-downloaded cube using a Single-Day Fallback strategy.
    """
    cube_dates = metadata["dates"]
    cube_ccs = metadata["cloud_covers"]
    max_lookback_days = 30

    with h5py.File(h5_path, 'a') as f:
        fire_year = int(f.attrs['year'])
        days_grp = f['days']
        
        for day_key in tqdm(sorted(days_grp.keys()), desc="Slicing Fire Days"):
            day_grp = days_grp[day_key]
            current_dob = int(day_grp.attrs['DOB'])
            
            target_dt = np.datetime64(datetime(fire_year, 1, 1) + timedelta(days=current_dob - 1), 'D')
            
            found_valid_mosaic = False
            final_mosaic = None
            used_day_indices = []

            # Define the cloud coverage tiers
            threshold_tiers = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
            
            # ---------------------------------------------------------
            # Tiered Single-Day Fallback Loop
            # ---------------------------------------------------------
            for current_threshold in threshold_tiers:

                for lookback in range(max_lookback_days):

                    check_dt = target_dt - np.timedelta64(lookback, 'D')
            
                    # Find indices for this exact day WHERE the metadata cloud cover 
                    # is less than or equal to the current threshold tier.
                    day_indices = [
                        i for i, d in enumerate(cube_dates) 
                        if d == check_dt and cube_ccs[i] <= current_threshold
                    ]
                    
                    if not day_indices:
                        continue  # No satellite pass on this day

                    # Sort the indices so the tile with the LOWEST metadata cloud cover is at index 0
                    day_indices = sorted(day_indices, key=lambda idx: cube_ccs[idx])

                    # Slice the cube for only this day
                    daily_stack = cube[day_indices]
                    
                    # Perform Mosaic (stitching tiles from the same day)
                    mask = ~np.isnan(daily_stack) & (daily_stack != 0)
                    idx_first_valid = np.argmax(mask, axis=0) 
                    
                    rows, cols = np.indices(daily_stack.shape[2:])
                    final_mosaic = np.zeros(daily_stack.shape[1:], dtype=daily_stack.dtype)
                    
                    for b in range(daily_stack.shape[1]):
                        final_mosaic[b] = daily_stack[idx_first_valid[b], b, rows, cols]

                    # ---------------------------------------------------------
                    # Evaluate the single-day mosaic
                    # ---------------------------------------------------------
                    # STRICT NaN CHECK (Excluding SCL)
                    check_indices = [i for i, b_name in enumerate(bands) if b_name != "SCL"]
                    check_mosaic = final_mosaic[check_indices]
                    
                    if np.isfinite(check_mosaic).all() and not np.all(check_mosaic == 0):   
                        found_valid_mosaic = True
                        used_day_indices = day_indices
                        break
                
                if found_valid_mosaic:
                    break # Break the threshold loop

            # ---------------------------------------------------------
            # Handle Results & Save to H5
            # ---------------------------------------------------------
            if not found_valid_mosaic:
                if 'satellite' in day_grp:
                    del day_grp['satellite']
                continue # Skip saving if no good day was found within the lookback days

            # Calculate stats for the successful day
            stats_list = []
            total_pix = cube.shape[2] * cube.shape[3]
            window_ccs = cube_ccs[used_day_indices]
            selected_ccs = window_ccs[idx_first_valid[0]] 
            
            u_ccs, counts = np.unique(selected_ccs, return_counts=True)
            for cc, count in zip(u_ccs, counts):
                local_idx = np.where(window_ccs == cc)[0][0]
                global_date_idx = used_day_indices[local_idx]
                
                stats_list.append({
                    "pixels_pct": round((count / total_pix) * 100, 2),
                    "cloud_cover_pct": round(float(cc), 2),
                    "acquisition_date": str(cube_dates[global_date_idx])
                })

            # Save to H5 (Remove the old group and create a fresh one)
            if 'satellite' in day_grp:
                del day_grp['satellite']
            sat_grp = day_grp.require_group('satellite')
            sat_grp.attrs["cloud_cover_stats"] = json.dumps(stats_list)
            for i, b_name in enumerate(bands):
                d_name = f"s2_{b_name}"
                if d_name in sat_grp: del sat_grp[d_name]
                # Save the categorical SCL mask as a tiny uint8
                if b_name == "SCL":
                    sat_grp.create_dataset(d_name, data=final_mosaic[i].astype(np.uint8), compression="lzf")
                else:
                    sat_grp.create_dataset(d_name, data=final_mosaic[i], compression="lzf")

def run_s2_h5_pipeline(h5_path, catalog, bands):

    pipeline_bands = list(bands)
    if "SCL" not in pipeline_bands:
        pipeline_bands.append("SCL")

    cube, metadata = get_fire_master_cube(h5_path, catalog, pipeline_bands)
    process_h5_from_cube(h5_path, cube, metadata, pipeline_bands)