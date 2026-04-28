from pathlib import Path
import numpy as np
import rioxarray
import xarray as xr
from pyproj import Transformer
import h5py

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
    
def run_single_fire_scanfi(h5_path, scanfi_folder, current_year="2020"):
    """
    Tile-Based Architecture.
    Processes static SCANFI fuel data and stores it inside each individual tile.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"Error: {h5_path} does not exist.")
        return
    
    year_folder = Path(scanfi_folder) / current_year
    
    print(f"\n{'='*60}\nRunning Tile-Based SCANFI Pipeline: {h5_path.name}")

    with h5py.File(h5_path, "a") as f:
        # Identify all tiles in the file
        tile_ids = [k for k in f.keys() if k.startswith('tile_')]
        
        if not tile_ids:
            print("No tiles found in this H5 file. Skipping.")
            return

        # Loop through the SCANFI variables first to minimize opening/closing the large TIFs
        for key, file_prefix in SCANFI_VARS.items():
            var_name = key.lower()
            
            # Find the specific TIF for this variable
            matching_files = list(year_folder.glob(f"*{file_prefix}*_90m_v2.tif"))
            if not matching_files:
                print(f"Skipping {var_name}: No TIF found for '{file_prefix}' in {year_folder}.")
                continue
            
            tif_path = matching_files[0]
            print(f"--- Extracting {var_name} from {tif_path.name} ---")

            # Process every tile for this specific variable
            for tid in tile_ids:
                # Navigate to (or create) the tile's static_features group
                tile_grp = f[tid]
                if 'static_features' not in tile_grp:
                    static_grp = tile_grp.create_group('static_features')
                else:
                    static_grp = tile_grp['static_features']

                # Get this specific tile's coordinates
                lon_grid = tile_grp['coords/theoretical_lon'][:]
                lat_grid = tile_grp['coords/theoretical_lat'][:]

                # Extract the 2D array for this tile's window
                grid_data = extract_scanfi_for_grid(tif_path, lon_grid, lat_grid)
                grid_data = np.nan_to_num(grid_data, nan=0.0)

                # Save to HDF5 (Overwrite if exists)
                if var_name in static_grp:
                    del static_grp[var_name]
                
                static_grp.create_dataset(var_name, data=grid_data, compression="lzf")

    print(f"Successfully processed all SCANFI tiles for {h5_path.name}")