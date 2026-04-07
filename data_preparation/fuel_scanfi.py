from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
import rioxarray
import xarray as xr
from pyproj import Transformer
import h5py
import json

SCANFI_VARS = {
    'Biomass': 'SCANFI_att_biomass_SW',
    'Closure': 'SCANFI_att_closure_SW',
    'prcC': 'SCANFI_sps_prcC_other_SW',
    'prcB': 'SCANFI_sps_prcB_SW'
}
SCANFI_KEYWORDS = ['biomass', 'closure', 'prcC', 'prcB']

def extract_scanfi_for_grid(tif_path, lon_grid, lat_grid):
    """
    Extracts 2D grid data from a large TIFF file efficiently by cropping first.
    """
    with rioxarray.open_rasterio(tif_path) as da:
        
        # Setup Transformer
        transformer = Transformer.from_crs("EPSG:4269", da.rio.crs, always_xy=True)
        
        # Transform the entire 2D grid to the Raster's CRS
        target_x, target_y = transformer.transform(lon_grid, lat_grid)
        
        # Crop the raster to the fire's extent
        # Add a 200m buffer
        buffer = 200
        min_x, max_x = target_x.min() - buffer, target_x.max() + buffer
        min_y, max_y = target_y.min() - buffer, target_y.max() + buffer
        
        da_cropped = da.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
        
        # Create 2D Xarray coordinates
        x_coords = xr.DataArray(target_x, dims=("y", "x"))
        y_coords = xr.DataArray(target_y, dims=("y", "x"))
        
        # Sample the cropped raster at the grid points
        sampled = da_cropped.sel(x=x_coords, y=y_coords, method="nearest").compute()
        
        # Extract the numpy array
        if sampled.ndim == 3 and sampled.shape[0] == 1:
            return sampled.values[0]
        
        return sampled.values
    
def run_static_scanfi_pipeline(h5_folder, scanfi_folder, current_year="2020"):
    
    h5_files = list(Path(h5_folder).glob("*.h5"))
    
    year_folder = Path(scanfi_folder) / current_year
    
    for h5_path in tqdm(h5_files):
            
        print(f"\n{'='*60}\nProcessing Static SCANFI for: {h5_path.name}")

        start_scanfi = time.time()
        
        with h5py.File(h5_path, "a") as f:
            
            # Get coordinates
            lon_grid = f['geometry/theoretical_lon'][:]
            lat_grid = f['geometry/theoretical_lat'][:]

            fire_id = f.attrs['fire_id']
            
            # Create static features bloc if not existing
            if 'static_features' not in f:
                static_grp = f.create_group('static_features')
            else:
                static_grp = f['static_features']
            
            # Loop on SCANFI variables
            for key, file_prefix in SCANFI_VARS.items():
                
                var_name = key.lower()  # e.g., 'biomass'
                
                # Fetch the SCANFI rasters
                matching_files = list(year_folder.glob(f"*{file_prefix}*_90m_v2.tif"))
                
                if not matching_files:
                    print(f"Skipping {var_name}: No TIF found matching '{file_prefix}'.")
                    continue
                
                tif_path = matching_files[0]
                    
                print(f"Extracting {var_name} from {tif_path.name}...")
                
                # Fetch the 2D array
                grid_data = extract_scanfi_for_grid(tif_path, lon_grid, lat_grid)

                #### TEMP
                grid_data = np.nan_to_num(grid_data, nan=0.0)
                
                # Save to HDF5
                static_grp.create_dataset(var_name, data=grid_data, compression="lzf")
                
                print(f"Saved {var_name}.")

def run_single_fire_scanfi(h5_path, scanfi_folder, current_year="2020"):
    """
    Processes SCANFI fuel data for a single H5 fire file.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"Error: {h5_path} does not exist.")
        return
    
    year_folder = Path(scanfi_folder) / current_year
    
    print(f"\n{'='*60}\nProcessing Static SCANFI for: {h5_path.name}")

    with h5py.File(h5_path, "a") as f:
        # Get coordinates from the fire-specific grid
        lon_grid = f['geometry/theoretical_lon'][:]
        lat_grid = f['geometry/theoretical_lat'][:]
        fire_id = f.attrs.get('fire_id', h5_path.stem)

        # Create static features group if not existing
        if 'static_features' not in f:
            static_grp = f.create_group('static_features')
        else:
            static_grp = f['static_features']
        
        # Loop on SCANFI variables (Biomass, etc.)
        for key, file_prefix in SCANFI_VARS.items():
            var_name = key.lower()
            
            # Find the specific TIF for this variable
            matching_files = list(year_folder.glob(f"*{file_prefix}*_90m_v2.tif"))
            
            if not matching_files:
                print(f"Skipping {var_name}: No TIF found matching '{file_prefix}' in {year_folder}.")
                continue
            
            tif_path = matching_files[0]
            
            # If the dataset already exists in this fire file, we overwrite/re-create it
            if var_name in static_grp:
                del static_grp[var_name]
                
            print(f"Extracting {var_name} from {tif_path.name}...")
            
            # Fetch the 2D array
            grid_data = extract_scanfi_for_grid(tif_path, lon_grid, lat_grid)

            #### TEMP
            grid_data = np.nan_to_num(grid_data, nan=0.0)
            
            # 5. Save to HDF5
            static_grp.create_dataset(var_name, data=grid_data, compression="lzf")
            print(f"Saved {var_name}.")

    print(f"Successfully processed SCANFI for {h5_path.name}")