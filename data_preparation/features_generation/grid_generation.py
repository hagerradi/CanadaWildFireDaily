import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import math
from pyproj import Transformer
from timezonefinder import TimezoneFinder
import pytz
from datetime import datetime, timedelta
import h5py

def local_to_utc(row, tf, lon_col, lat_col):
    
    if pd.isna(row[lat_col]) or pd.isna(row[lon_col]) or pd.isna(row['DOB']):
        return pd.Series([None, None, None, None])
    
    tz_name = tf.timezone_at(lat=row[lat_col], lng=row[lon_col])

    if tz_name is None:
        tz_name = "UTC"

    tz = pytz.timezone(tz_name)

    date = datetime(int(row['year']), 1, 1) + timedelta(days=int(row['DOB'])-1)
    start_local = tz.localize(datetime(date.year, date.month, date.day))
    end_local = start_local + timedelta(days=1)
    noon_local = tz.localize(datetime(date.year, date.month, date.day, 12))

    return pd.Series([
        int(date.month),
        start_local.astimezone(pytz.utc).isoformat()[:19],
        end_local.astimezone(pytz.utc).isoformat()[:19],
        noon_local.astimezone(pytz.utc).isoformat()[:19]
    ])

def get_global_grid_params(df, fire_id, grid_size=256, pixel_size=90, x_col='easting', y_col='northing', min_overlap=0.1):
    
    fire_df = df[df["ID"] == fire_id]
    
    # Get the absolute bounds (Just like in analyze_point_fit)
    x_min, x_max = fire_df[x_col].min(), fire_df[x_col].max()
    y_min, y_max = fire_df[y_col].min(), fire_df[y_col].max()
    
    # Calculate the exact width and height in meters
    width_m = x_max - x_min
    height_m = y_max - y_min
    
    # Calculate the true geometric center (Midpoint)
    x_center = x_min + (width_m / 2.0)
    y_center = y_min + (height_m / 2.0)

    # Calculate multipliers using the exact width/height
    # (No need to multiply by 2 anymore, because width_m is already the full span)
    mult_x = max(1, math.ceil(width_m / (grid_size * pixel_size)))
    mult_y = max(1, math.ceil(height_m / (grid_size * pixel_size)))

    # print(f"Grid Multipliers {fire_id}: {mult_x}, {mult_y}")
    
    base_grid_x = mult_x * grid_size
    base_grid_y = mult_y * grid_size

    # Calculate exact padding needed for the lowest overlap you plan to use
    stride = int(grid_size * (1.0 - min_overlap))
    
    def get_padding(base_dim):
        if base_dim <= grid_size:
            return 0  
            
        dist_to_edge = (base_dim - grid_size) / 2.0
        remainder = dist_to_edge % stride
        
        if remainder == 0:
            return 0
        else:
            return int(stride - remainder)

    # Get the padding required PER SIDE
    pad_x = get_padding(base_grid_x)
    pad_y = get_padding(base_grid_y)

    # Add the padding symmetrically to both sides (pad * 2)
    final_grid_x = base_grid_x + (pad_x * 2)
    final_grid_y = base_grid_y + (pad_y * 2)

    # Define the Master Top-Left Origin using the newly padded dimensions
    x_min_ref = x_center - (final_grid_x * pixel_size) / 2.0
    y_max_ref = y_center + (final_grid_y * pixel_size) / 2.0
    
    return {
        "grid_size_x": final_grid_x,
        "grid_size_y": final_grid_y,
        "x_min_ref": x_min_ref,
        "y_max_ref": y_max_ref
    }

def initialize_fire_h5(output_folder, fire_id, year, params, pixel_size=90, fill_value=np.nan):
    file_path = os.path.join(output_folder, f"fire_{fire_id}.h5")
    
    grid_x_size = params["grid_size_x"]
    grid_y_size = params["grid_size_y"]
    x_min_ref = params["x_min_ref"]
    y_max_ref = params["y_max_ref"]

    # Initialize the transformer
    # always_xy=True ensures we use (Easting, Northing) -> (Lon, Lat) order
    transformer = Transformer.from_crs("EPSG:3347", "EPSG:4269", always_xy=True)

    with h5py.File(file_path, "w") as f:
        # Global Attributes
        f.attrs["fire_id"] = fire_id
        f.attrs["year"] = year
        f.attrs["pixel_size"] = pixel_size
        f.attrs["meter_crs"] = "EPSG:3347"
        f.attrs["deg_crs"] = "EPSG:4269"
        

        # Create Geometry Group
        geo = f.create_group("geometry")
        
        # Logic for 2D Grid
        res_rows, res_cols = np.indices((grid_y_size, grid_x_size))
        y_min_ref = y_max_ref - (grid_y_size * pixel_size)
        
        theoretical_x = x_min_ref + (res_cols * pixel_size)
        theoretical_y = y_min_ref + (res_rows * pixel_size)

        # Transforming the full 2D arrays
        theoretical_lon, theoretical_lat = transformer.transform(theoretical_x, theoretical_y)

        # Save Easting/Northing
        geo.create_dataset("theoretical_easting", data=theoretical_x, compression="gzip")
        geo.create_dataset("theoretical_northing", data=theoretical_y, compression="gzip")
        
        # Save Lon/Lat
        geo.create_dataset("theoretical_lon", data=theoretical_lon, compression="gzip")
        geo.create_dataset("theoretical_lat", data=theoretical_lat, compression="gzip")

        # Create Daily Group
        f.create_group("days")

    print(f"Initialized H5 for {fire_id} with geometry and coordinate transformation.")
    
    return file_path

def add_fire_day_to_h5(file_path, df, fire_id, fireday, params, pixel_size, 
                       fill_value=-9999, x_col='easting', y_col='northing', 
                       lat_col='lat', lon_col='lon'):
    
    # Features Columns
    env_cols = ['firearea', 'cumuarea']
    # Metadata
    meta_keys = ["DOB"]

    # Filter data for the specific fire and day
    df_day = df[(df["ID"] == fire_id) & (df["fireday"] == fireday)].copy()
    
    grid_x_size = params["grid_size_x"]
    grid_y_size = params["grid_size_y"]
    x_min_ref = params["x_min_ref"]
    y_max_ref = params["y_max_ref"]
    y_min_ref = y_max_ref - (grid_y_size * pixel_size)

    # Initialize Grids for observations and features
    # Coordinates (Observed)
    grid_obs_e = np.full((grid_y_size, grid_x_size), fill_value, dtype=np.float32)
    grid_obs_n = np.full((grid_y_size, grid_x_size), fill_value, dtype=np.float32)
    grid_obs_lat = np.full((grid_y_size, grid_x_size), fill_value, dtype=np.float32)
    grid_obs_lon = np.full((grid_y_size, grid_x_size), fill_value, dtype=np.float32)
    
    # Features
    env_grids = {col: np.full((grid_y_size, grid_x_size), fill_value, dtype=np.float32) for col in env_cols}

    # Map the Real Points onto the Grid
    if not df_day.empty:
        # Calculate pixel indices
        day_cols = np.round((df_day[x_col] - x_min_ref) / pixel_size).astype(int)
        day_rows = np.round((df_day[y_col] - y_min_ref) / pixel_size).astype(int)

        valid = (day_rows >= 0) & (day_rows < grid_y_size) & (day_cols >= 0) & (day_cols < grid_x_size)
        
        t_rows = day_rows[valid].values
        t_cols = day_cols[valid].values

        # Fill Coordinate Grids
        grid_obs_e[t_rows, t_cols] = df_day[x_col].values[valid]
        grid_obs_n[t_rows, t_cols] = df_day[y_col].values[valid]
        if lat_col in df_day.columns: grid_obs_lat[t_rows, t_cols] = df_day[lat_col].values[valid]
        if lon_col in df_day.columns: grid_obs_lon[t_rows, t_cols] = df_day[lon_col].values[valid]
        
        
        # Fill feature Columns
        for col in env_cols:
            if col in df_day.columns:
                env_grids[col][t_rows, t_cols] = df_day[col].values[valid]

    # Write to HDF5
    with h5py.File(file_path, "a") as f:
        # Create group for the day (e.g., /days/day_005)
        day_grp_name = f"days/day_{int(fireday):03d}"
        if day_grp_name in f: del f[day_grp_name] # Overwrite if exists
        
        day_grp = f.create_group(day_grp_name)
        
        # Add Metadata as Attributes
        day_grp.attrs["fireday"] = fireday
        
        if not df_day.empty:

            first_row = df_day.iloc[0] # Grab one representative point for this fire day
            
            # Calculate on the 
            timezone_finder = TimezoneFinder()
            month, start_utc, end_utc, noon_utc = local_to_utc(first_row, timezone_finder, lon_col, lat_col)
            
            # Save
            day_grp.attrs["month"] = month
            
            # Use isoformat() to turn datetimes into safe strings for H5
            day_grp.attrs["start_utc"] = start_utc.isoformat() if hasattr(start_utc, 'isoformat') else str(start_utc)
            day_grp.attrs["end_utc"]   = end_utc.isoformat()   if hasattr(end_utc, 'isoformat')   else str(end_utc)
            day_grp.attrs["noon_utc"]  = noon_utc.isoformat()  if hasattr(noon_utc, 'isoformat')  else str(noon_utc)

            
            for k in meta_keys:
                if k in df_day.columns:
                    val = df_day[k].iloc[0]
                    # Handle None or NaN for H5 attributes
                    day_grp.attrs[k] = val if pd.notnull(val) else "N/A"

        # Save Grids
        obs_grp = day_grp.create_group("observed_coords")
        obs_grp.create_dataset("easting", data=grid_obs_e, compression="gzip")
        obs_grp.create_dataset("northing", data=grid_obs_n, compression="gzip")
        obs_grp.create_dataset("lat", data=grid_obs_lat, compression="gzip")
        obs_grp.create_dataset("lon", data=grid_obs_lon, compression="gzip")
        
        env_grp = day_grp.create_group("features")
        for col, grid in env_grids.items():
            env_grp.create_dataset(col, data=grid, compression="gzip")

    # print(f"Added day {fireday} to {file_path}")