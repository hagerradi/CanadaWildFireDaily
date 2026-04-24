from pathlib import Path
import rioxarray
from rioxarray.merge import merge_arrays
import xarray as xr
from pyproj import Transformer
import h5py

def get_tiles(fire_id, raster_folder):
    """
    Retrieves the specific terrain tile(s) for a given fire ID from a folder.
    """
    # Use glob to find any .tif file that starts with the fire_id
    # e.g., "2024_545_*.tif"
    overlapping = list(Path(raster_folder).glob(f"{fire_id}_*.tif"))
    
    return overlapping

def get_cropped_mosaic(tile_paths, target_x, target_y, is_latlon=True):
    
    """Opens, crops, masks NoData, and merges multiple raster tiles using pre-matched grids."""
    cropped_das = []
    
    buffer = 0
        
    # Get the bounding box directly from the grid you passed in
    min_x, max_x = target_x.min() - buffer, target_x.max() + buffer
    min_y, max_y = target_y.min() - buffer, target_y.max() + buffer

    with rioxarray.open_rasterio(tile_paths[0]) as da: 
        nodata_value = da.rio.nodata
    
    for path in tile_paths:
        
        with rioxarray.open_rasterio(path, masked=True) as da: 
            
            try:
                # Crop immediately using the pre-calculated min/max
                da_cropped = da.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
                
                # Mask out any remaining NoData
                da_cropped = da_cropped.where(da_cropped != nodata_value)
                
                cropped_das.append(da_cropped)
            
            except Exception:
                # Bbox misses this tile, safely ignore
                pass 
                
    if not cropped_das: 
        return None
    
    if len(cropped_das) == 1: 
        return cropped_das[0].fillna(nodata_value)
        
    # Merge and fill
    mosaic = merge_arrays(cropped_das)
    
    mosaic = mosaic.fillna(nodata_value)
    
    return mosaic

def run_static_topography_pipeline(h5_path, dem_folders_dict):
    
    print(f"\n{'='*60}\nRunning Static Topography for: {Path(h5_path).name}")
    
    with h5py.File(h5_path, "a") as f:
        
        # Get theoretical grid
        lon_grid = f['geometry/theoretical_lon'][:]
        lat_grid = f['geometry/theoretical_lat'][:]

        # Get theoretical grid
        easting_grid = f['geometry/theoretical_easting'][:]
        northing_grid = f['geometry/theoretical_northing'][:]

        fire_id = f.attrs['fire_id']
        
        # Ensure static group exists
        static_grp = f.require_group('static_features')
            
        for var_name, folder_path in dem_folders_dict.items():
            
            var_v2 = f"{var_name}"

            tiles = get_tiles(fire_id, folder_path)
            
            if not tiles:
                print(f"Skipping {var_v2}: No tiles found.")
                continue
            
            print(f"Extracting {var_v2}...")
            if var_name == 'dem_avg' or var_name == 'dem':
                da_mosaic = get_cropped_mosaic(tiles, lon_grid, lat_grid, is_latlon=True)
            else:
                da_mosaic = get_cropped_mosaic(tiles, easting_grid, northing_grid, is_latlon=False)
            
            if da_mosaic is None:
                continue
                
            transformer = Transformer.from_crs("EPSG:4269", da_mosaic.rio.crs, always_xy=True)
            target_x, target_y = transformer.transform(lon_grid, lat_grid)
            
            x_coords = xr.DataArray(target_x, dims=("y", "x"))
            y_coords = xr.DataArray(target_y, dims=("y", "x"))
            
            # Sample the continuous grid
            v2_sampled = da_mosaic.sel(x=x_coords, y=y_coords, method="nearest").compute()
            v2_data = v2_sampled.values[0] if v2_sampled.ndim == 3 else v2_sampled.values
            
            # Save to static_features
            if var_v2 in static_grp:
                static_grp[var_v2][...] = v2_data
            else:
                static_grp.create_dataset(var_v2, data=v2_data, compression="lzf")