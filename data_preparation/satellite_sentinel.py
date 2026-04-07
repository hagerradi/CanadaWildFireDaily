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
import xarray as xr
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
        print(f"✅ Confirmed: Shape is {geobox.shape}")
        print(f"✅ Confirmed: Resolution is {geobox.resolution}")
        
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

    # Search window: start 21 days before fire, end 1 day after
    start_date = datetime(fire_year, 1, 1) + timedelta(days=start_dob - 21)
    end_date = datetime(fire_year, 1, 1) + timedelta(days=end_dob + 1)

    print(f'Fire {h5_path} metadata acquired.')

    # ---------------------------------------------------------
    # STAC API Retry Loop (Protects against Azure 504s)
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
                # query={"eo:cloud_cover": {"lt": 80}} # Optimization: Drops garbage scenes
            )
            items = list(search.items())
            
            # Note: We DO NOT sign the items here anymore. odc-stac handles it.
            
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
        # chunks={'time': 1, 'x': 256, 'y': 256}
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
    
    with dask.config.set(scheduler='threads', num_workers=8):
        for attempt in range(max_retries):
            try:
                print(f"Download Attempt {attempt + 1}...")
                with ProgressBar():
                    cube = stack_da.compute()            
                
                if cube is not None:
                    print("Download successful!")
                    break 
                    
            except Exception as e:
                print(f"\n[!] Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1) 
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Failing task.")
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

    return cube.values, metadata

def process_h5_from_cube(h5_path, cube, metadata, bands, previous_days_buffer=4):
    """
    Slices the pre-downloaded cube for each day in the H5.
    """
    cube_dates = metadata["dates"]
    cube_ccs = metadata["cloud_covers"]

    with h5py.File(h5_path, 'a') as f:

        fire_year = int(f.attrs['year'])
        days_grp = f['days']
        
        for day_key in tqdm(sorted(days_grp.keys()), desc="Slicing Fire Days"):

            day_grp = days_grp[day_key]
            current_dob = int(day_grp.attrs['DOB'])
            
            # Define moving window for the specific day
            target_dt = np.datetime64(datetime(fire_year, 1, 1) + timedelta(days=current_dob - 1), 'D')
            
            current_buffer = previous_days_buffer
            max_lookback_days = 25  # Safety limit
            
            # Keep looping and expanding the window until we get a clean image
            while current_buffer <= max_lookback_days:
                lookback_dt = target_dt - np.timedelta64(current_buffer, 'D')
                
                indices = [i for i, d in enumerate(cube_dates) if lookback_dt <= d <= target_dt]
                
                if not indices:
                    current_buffer += 2
                    continue

                # Sort newest date first
                indices = sorted(indices, key=lambda idx: cube_dates[idx], reverse=True)
                threshold = 30.0  # 30% cloud cover            
                good_quality_indices = [i for i in indices if cube_ccs[i] <= threshold]           
                
                if good_quality_indices:
                    remaining = [i for i in indices if i not in good_quality_indices]
                    indices = good_quality_indices + remaining
                else:
                    indices = sorted(indices, key=lambda idx: cube_ccs[idx])

                # Slice the 4D cube
                daily_stack = cube[indices] 
                
                # Perform Mosaic
                mask = ~np.isnan(daily_stack) & (daily_stack != 0)
                idx_first_valid = np.argmax(mask, axis=0) 
                
                rows, cols = np.indices(daily_stack.shape[2:])
                final_mosaic = np.zeros(daily_stack.shape[1:], dtype=daily_stack.dtype)
                
                # Temporarily fill the mosaic to check if it's complete
                for b in range(daily_stack.shape[1]):
                    final_mosaic[b] = daily_stack[idx_first_valid[b], b, rows, cols]
                
                # --- EVALUATE ---
                if not np.isfinite(final_mosaic).all() or np.all(final_mosaic == 0):
                    current_buffer += 2  # Broken/Empty image! Look 2 days deeper.
                else:
                    break  # Clean image! Break out of the while loop.

            if not indices:
                continue # Skip this day entirely if no data exists even after 60 days

            stats_list = []
            total_pix = cube.shape[2] * cube.shape[3]

            for b in range(daily_stack.shape[1]):
                # Fill the band mosaic
                final_mosaic[b] = daily_stack[idx_first_valid[b], b, rows, cols]
                
                # Calculate stats only once (using Band 0 as proxy)
                if b == 0:
                    # Get the 1D list of cloud covers for this specific 3-day window
                    window_ccs = cube_ccs[indices]
                    
                    # Map the 1D cloud covers to the 2D grid of chosen pixels
                    selected_ccs = window_ccs[idx_first_valid[0]] 
                    
                    u_ccs, counts = np.unique(selected_ccs, return_counts=True)
                    for cc, count in zip(u_ccs, counts):
                        # Find which image in our window had this cloud cover
                        local_idx = np.where(window_ccs == cc)[0][0]
                        global_date_idx = indices[local_idx]
                        
                        stats_list.append({
                            "pixels_pct": round((count / total_pix) * 100, 2),
                            "cloud_cover_pct": round(float(cc), 2),
                            "acquisition_date": str(cube_dates[global_date_idx])
                        })

            # Save to H5
            sat_grp = day_grp.require_group('satellite')
            sat_grp.attrs["cloud_cover_stats"] = json.dumps(stats_list)
            for i, b_name in enumerate(bands):
                d_name = f"s2_{b_name}"
                if d_name in sat_grp: del sat_grp[d_name]
                # sat_grp.create_dataset(d_name, data=final_mosaic[i], compression="lzf", chunks=(256, 256))
                sat_grp.create_dataset(d_name, data=final_mosaic[i], compression="lzf")

def run_s2_h5_pipeline_v2(h5_path, catalog, bands):

    cube, metadata = get_fire_master_cube(h5_path, catalog, bands)
    process_h5_from_cube(h5_path, cube, metadata, bands, previous_days_buffer=4)