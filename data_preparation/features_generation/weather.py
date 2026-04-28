from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar
import xarray as xr
import h5py
import gc

def process_era5_grid(ds_fire, lon_grid, lat_grid, day_attrs):

    # Times
    t_end = pd.to_datetime(day_attrs['end_utc'])
    p_end = pd.to_datetime(day_attrs['noon_utc']) + pd.Timedelta(hours=1)
    noon  = pd.to_datetime(day_attrs['noon_utc']) + pd.Timedelta(hours=1)
    
    # Selection for Noon variables (Wind, RH)
    noon_data = ds_fire.interp(
        latitude=xr.DataArray(lat_grid, dims=("y", "x")),
        longitude=xr.DataArray(lon_grid, dims=("y", "x")),
        valid_time=noon,
        method="nearest"
    )

    # Selection for Tmax and Prec
    tmax_data = ds_fire['t2m_24h_max'].interp(
        latitude=xr.DataArray(lat_grid, dims=("y", "x")),
        longitude=xr.DataArray(lon_grid, dims=("y", "x")),
        valid_time=t_end,
        method="nearest"
    )
    
    prec_data = ds_fire['tp_24h_sum'].interp(
        latitude=xr.DataArray(lat_grid, dims=("y", "x")),
        longitude=xr.DataArray(lon_grid, dims=("y", "x")),
        valid_time=p_end,
        method="nearest"
    )

    # Calculations
    u, v = noon_data['u10'].values, noon_data['v10'].values
    ws = np.sqrt(u**2 + v**2) * 3.6
    u_kmh = u * 3.6
    v_kmh = v * 3.6
    
    t_c = noon_data['t2m'].values - 273.15
    d_c = noon_data['d2m'].values - 273.15
    rh = 100 * (((112 - 0.1 * t_c + d_c) / (112 + 0.9 * t_c))**8)
    
    return {
        "ws": ws,
        "rh": rh,
        "tmax": tmax_data.values - 273.15,
        "prec": prec_data.values * 1000,
        "u10": u_kmh,
        "v10": v_kmh
    }

def fast_deduplicate_by_data(ds):

    # The objective of this function is to remove duplicates in the data when merging multiple months from ERA5
    # Example: Month 5 and Month 6 both contain data for the day 31-05 
    # However, in Month 6, the data only contains NaNs
    # So we remove based on the day that contains NaN (that is what call later the qulity score)

    
    # Create a 'quality score' (number of non-NaN values per timestamp)
    # We sum across space to see which hour is 'full'
    quality = (~ds['t2m'].isnull()).sum(dim=['latitude', 'longitude'])

    # Add this quality as a temporary coordinate
    ds = ds.assign_coords(quality=quality)
    
    # Sort by valid_time AND quality
    # This puts the timestamps with NaNs at the 'bottom' of each duplicate group
    ds = ds.sortby(['valid_time', 'quality'])
    
    # Drop duplicates, keeping the 'last' (which is the one with highest quality)
    ds = ds.drop_duplicates("valid_time", keep='last')
    
    return ds.drop_vars('quality')

def run_single_h5_era5_pipeline(h5_path, era5_base_folder):
    """
    Processes ERA5 weather data for a single H5 fire file using the Tile-Based architecture.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"Error: {h5_path} does not exist.")
        return

    print(f"\n{'='*60}\nProcessing Single Fire H5: {h5_path.name}...")

    with h5py.File(h5_path, "a") as f:
        fire_id = f.attrs.get('fire_id', 'Unknown')
        y_int = int(f.attrs.get('year', 0))
        
        # Identify all tile groups
        tile_ids = [k for k in f.keys() if k.startswith('tile_')]
        if not tile_ids:
            print("No tiles found in this H5 file. Skipping.")
            return

        # ==========================================
        # STEP 1: First Pass - Find Global Bounds & Dates
        # ==========================================
        global_lon_min, global_lon_max = float('inf'), float('-inf')
        global_lat_min, global_lat_max = float('inf'), float('-inf')
        
        all_noon_utcs = []

        for tile_id in tile_ids:
            # Update Bounding Box across all tiles
            lon_grid = f[f"{tile_id}/coords/theoretical_lon"][:]
            lat_grid = f[f"{tile_id}/coords/theoretical_lat"][:]
            
            global_lon_min = min(global_lon_min, lon_grid.min())
            global_lon_max = max(global_lon_max, lon_grid.max())
            global_lat_min = min(global_lat_min, lat_grid.min())
            global_lat_max = max(global_lat_max, lat_grid.max())
            
            # Collect all active days to figure out which months we need
            days_grp = f[f"{tile_id}/days"]
            for d in days_grp.keys():
                noon_str = days_grp[d].attrs.get('noon_utc')
                if noon_str and noon_str != "N/A":
                    all_noon_utcs.append(pd.to_datetime(noon_str))

        if not all_noon_utcs:
            print("No valid days found across tiles. Skipping.")
            return

        # Calculate Required ERA5 Months based on all dates across the whole fire
        required_months = []
        unique_months = sorted(list(set(dt.month for dt in all_noon_utcs)))
        
        for m_int in unique_months:
            required_months.append((y_int, m_int))
            days_this_month = [dt for dt in all_noon_utcs if dt.month == m_int]
            
            # Rolling window logic: Add previous/next months if near boundaries
            if any(dt.day == 1 for dt in days_this_month):
                prev = datetime(y_int, m_int, 1) - relativedelta(months=1)
                required_months.append((prev.year, prev.month))
                
            last_day = calendar.monthrange(y_int, m_int)[1]
            if any(dt.day == last_day for dt in days_this_month):
                nxt = datetime(y_int, m_int, 1) + relativedelta(months=1)
                required_months.append((nxt.year, nxt.month))

        required_months = sorted(list(set(required_months)))

        # ==========================================
        # STEP 2: Load and Compute ERA5 Chunk
        # ==========================================
        required_ncs = []
        for y, m in required_months:
            nc_path = Path(era5_base_folder) / f"ERA5_LAND_{y}_{m}.nc"
            if nc_path.exists(): 
                required_ncs.append(nc_path)

        if not required_ncs:
            print(f"No NC files found for {h5_path.name}, skipping.")
            return

        print(f"Loading {len(required_ncs)} ERA5 files into RAM for global bounding box...")
        ds = xr.open_mfdataset(required_ncs, engine="h5netcdf", parallel=True)
        
        # Subset using the absolute min/max across ALL tiles
        buffer = 0.1
        ds = ds.sel(latitude=slice(global_lat_max + buffer, global_lat_min - buffer),
                    longitude=slice(global_lon_min - buffer, global_lon_max + buffer))

        ds_indexed = fast_deduplicate_by_data(ds)

        # Feature Engineering (Rolling)
        tp_prev = ds_indexed['tp'].shift(valid_time=1)
        ds_indexed['tp_hourly'] = xr.where(
            (ds_indexed.valid_time.dt.hour == 1) | ((ds_indexed['tp'] - tp_prev) < -1e-7),
            ds_indexed['tp'],
            ds_indexed['tp'] - tp_prev
        )
        ds_indexed['t2m_24h_max'] = ds_indexed['t2m'].rolling(valid_time=24).max()
        ds_indexed['tp_24h_sum'] = ds_indexed['tp_hourly'].rolling(valid_time=24).sum()

        start_bake = time.time()
        ds_indexed = ds_indexed.compute()
        print(f"ERA5 loaded to RAM in {time.time() - start_bake:.2f}s")

        # ==========================================
        # STEP 3: Write Weather Data into Each Tile
        # ==========================================
        for tile_id in tile_ids:
            # Grab this specific tile's theoretical coordinates
            tile_lon = f[f"{tile_id}/coords/theoretical_lon"][:]
            tile_lat = f[f"{tile_id}/coords/theoretical_lat"][:]
            
            days_grp = f[f"{tile_id}/days"]
            
            for day_key in sorted(days_grp.keys()):
                day_grp = days_grp[day_key]
                day_attrs = dict(day_grp.attrs)
                
                # Check for valid dates
                if day_attrs.get('noon_utc') == "N/A":
                    continue
                
                # Interpolate from RAM using the tile-specific grid
                weather_dict = process_era5_grid(ds_indexed, tile_lon, tile_lat, day_attrs)
                
                # Save into the new Features group
                env_grp = day_grp['features']
                for var_name, grid_data in weather_dict.items():
                    if var_name in env_grp:
                        env_grp[var_name][...] = grid_data
                    else:
                        env_grp.create_dataset(var_name, data=grid_data, compression="gzip")
        
        # Cleanup
        ds.close()
        ds_indexed.close()
        del ds, ds_indexed
        gc.collect()
        
        print(f"Successfully processed {h5_path.name}")