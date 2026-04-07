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
    
    t_c = noon_data['t2m'].values - 273.15
    d_c = noon_data['d2m'].values - 273.15
    rh = 100 * (((112 - 0.1 * t_c + d_c) / (112 + 0.9 * t_c))**8)
    
    return {
        "ws": ws,
        "rh": rh,
        "tmax": tmax_data.values - 273.15,
        "prec": prec_data.values * 1000
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

def run_h5_era5_pipeline(h5_folder, era5_base_folder):
    
    h5_files = list(Path(h5_folder).glob("*.h5"))
    
    for h5_path in tqdm(h5_files):
        
        print(f"\n{'='*60}\nProcessing Fire H5: {h5_path.name}...")

        # start_weather = time.time()
        
        with h5py.File(h5_path, "a") as f:
            
            # Get Geometry and Time Info
            lon_grid = f['geometry/theoretical_lon'][:]
            lat_grid = f['geometry/theoretical_lat'][:]

            fire_id = f.attrs['fire_id']
            
            # Collect all months present in this specific fire's days
            days_grp = f['days']
            fire_months = []
            fire_years = []
            
            for day_key in days_grp.keys():
                fire_months.append(int(days_grp[day_key].attrs['month']))
                fire_years.append(int(f.attrs['year'])) # Using global year attr

            
            required_months = []
            
            # Determine unique months from H5 attributes
            unique_months = sorted(set(int(f['days'][d].attrs['month']) for d in f['days'].keys()))
            y_int = int(f.attrs['year'])
            
            required_months = []
            for m_int in unique_months:
                
                required_months.append((y_int, m_int)) # Always add the current month
                
                # Filter days to ONLY this specific month
                days_this_month = [
                    pd.to_datetime(f['days'][d].attrs['noon_utc']) 
                    for d in f['days'].keys() 
                    if int(f['days'][d].attrs['month']) == m_int
                ]
                
                # If fire is on Day 1, add the PREVIOUS month for the 24h rolling window
                if any(d.day == 1 for d in days_this_month):
                    prev = datetime(y_int, m_int, 1) - relativedelta(months=1)
                    required_months.append((prev.year, prev.month))
                
                # If fire is on the Last Day, add the NEXT month for end_utc logic
                last_day_of_month = calendar.monthrange(y_int, m_int)[1]
                if any(d.day == last_day_of_month for d in days_this_month):
                    nxt = datetime(y_int, m_int, 1) + relativedelta(months=1)
                    required_months.append((nxt.year, nxt.month))
    
            # Deduplicate and sort
            required_months = sorted(list(set(required_months)))
    
            # fetch NCS
            required_ncs = []
            
            for y, m in required_months:
                nc_path = Path(era5_base_folder) / f"ERA5_LAND_{y}_{m}.nc"
                if nc_path.exists(): 
                    required_ncs.append(nc_path)
    
            print(f'Found {len(required_ncs)} nc files.')

            if not required_ncs:
                print(f"No NC file found for {h5_path.name}, skipping.")
                continue

            # Open and Process ERA5
            ds = xr.open_mfdataset(required_ncs, engine="h5netcdf", parallel=True)
            
            # Only load the box surrounding the fire with a buffer
            buffer = 0.1
            ds = ds.sel(latitude=slice(lat_grid.max() + buffer, lat_grid.min() - buffer),
                        longitude=slice(lon_grid.min() - buffer, lon_grid.max() + buffer))

            ds_indexed = fast_deduplicate_by_data(ds)
            print(f'Deduplicated.')

            # Rolling calculations
            tp_prev = ds_indexed['tp'].shift(valid_time=1)
            ds_indexed['tp_hourly'] = xr.where(
                (ds_indexed.valid_time.dt.hour == 1) | ((ds_indexed['tp'] - tp_prev) < -1e-7),
                ds_indexed['tp'],
                ds_indexed['tp'] - tp_prev
            )
            ds_indexed['t2m_24h_max'] = ds_indexed['t2m'].rolling(valid_time=24).max()
            ds_indexed['tp_24h_sum'] = ds_indexed['tp_hourly'].rolling(valid_time=24).sum()
            print(f'Temp and Prec calculated.')

            # Trigger the calculations
            print("Loading ERA5 slice into RAM...")
            start_bake = time.time()
            ds_indexed = ds_indexed.compute()
            print(f"Loading done in {time.time() - start_bake:.2f}s")

            # Save to H5
            for day_key in sorted(days_grp.keys()):
                
                day_grp = days_grp[day_key]
                day_attrs = dict(day_grp.attrs)
                
                # Process the theoretical Grid
                weather_v2 = process_era5_grid(ds_indexed, lon_grid, lat_grid, day_attrs)
                
                env_grp = day_grp['features']
                
                # Save
                for weather_dict in [weather_v2]:
                    for var_name, grid_data in weather_dict.items():
                        if var_name in env_grp:
                            env_grp[var_name][...] = grid_data
                        else:
                            env_grp.create_dataset(var_name, data=grid_data, compression="gzip")
            
            ds.close()
            ds_indexed.close()
            del ds, ds_indexed
            import gc
            gc.collect()
            
            print(f"Successfully processed {h5_path.name}")

def run_single_h5_era5_pipeline(h5_path, era5_base_folder):
    """
    Processes ERA5 weather data for a single H5 fire file.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"Error: {h5_path} does not exist.")
        return

    print(f"\n{'='*60}\nProcessing Single Fire H5: {h5_path.name}...")

    with h5py.File(h5_path, "a") as f:
        # Get Geometry and Time Info
        lon_grid = f['geometry/theoretical_lon'][:]
        lat_grid = f['geometry/theoretical_lat'][:]
        fire_id = f.attrs['fire_id']
        y_int = int(f.attrs['year'])
        days_grp = f['days']
        
        # Determine unique months and buffer months (prev/next)
        unique_months = sorted(set(int(days_grp[d].attrs['month']) for d in days_grp.keys()))
        
        required_months = []
        for m_int in unique_months:
            required_months.append((y_int, m_int))
            
            # Filter days to ONLY this specific month to check boundaries
            days_this_month = [
                pd.to_datetime(days_grp[d].attrs['noon_utc']) 
                for d in days_grp.keys() 
                if int(days_grp[d].attrs['month']) == m_int
            ]
            
            # Handle rolling window overlaps
            if any(d.day == 1 for d in days_this_month):
                prev = datetime(y_int, m_int, 1) - relativedelta(months=1)
                required_months.append((prev.year, prev.month))
            
            last_day_of_month = calendar.monthrange(y_int, m_int)[1]
            if any(d.day == last_day_of_month for d in days_this_month):
                nxt = datetime(y_int, m_int, 1) + relativedelta(months=1)
                required_months.append((nxt.year, nxt.month))

        required_months = sorted(list(set(required_months)))

        # Locate NC files
        required_ncs = []
        for y, m in required_months:
            nc_path = Path(era5_base_folder) / f"ERA5_LAND_{y}_{m}.nc"
            if nc_path.exists(): 
                required_ncs.append(nc_path)

        if not required_ncs:
            print(f"No NC files found for {h5_path.name}, skipping.")
            return

        # Open and Process ERA5 with xarray
        # parallel=True helps if you have multiple CPUs assigned in Slurm
        ds = xr.open_mfdataset(required_ncs, engine="h5netcdf", parallel=True)
        
        # Spatial Subset (Bounding Box + Buffer)
        buffer = 0.1
        ds = ds.sel(latitude=slice(lat_grid.max() + buffer, lat_grid.min() - buffer),
                    longitude=slice(lon_grid.min() - buffer, lon_grid.max() + buffer))

        # Helper function (assumed to be in your environment)
        ds_indexed = fast_deduplicate_by_data(ds)

        # Feature Engineering
        tp_prev = ds_indexed['tp'].shift(valid_time=1)
        ds_indexed['tp_hourly'] = xr.where(
            (ds_indexed.valid_time.dt.hour == 1) | ((ds_indexed['tp'] - tp_prev) < -1e-7),
            ds_indexed['tp'],
            ds_indexed['tp'] - tp_prev
        )
        ds_indexed['t2m_24h_max'] = ds_indexed['t2m'].rolling(valid_time=24).max()
        ds_indexed['tp_24h_sum'] = ds_indexed['tp_hourly'].rolling(valid_time=24).sum()

        # Compute and Save
        print("Loading ERA5 slice into RAM...")
        ds_indexed = ds_indexed.compute()

        for day_key in sorted(days_grp.keys()):
            day_grp = days_grp[day_key]
            day_attrs = dict(day_grp.attrs)
            
            # Interpolate ERA5 to your fire grid
            weather_v2 = process_era5_grid(ds_indexed, lon_grid, lat_grid, day_attrs)
            
            env_grp = day_grp['features']
            for var_name, grid_data in weather_v2.items():
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