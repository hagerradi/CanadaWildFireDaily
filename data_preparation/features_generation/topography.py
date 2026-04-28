from pathlib import Path
import rioxarray
import h5py
import time

def run_static_topography_pipeline(h5_path, dem_folders_dict, base_res=90):
    """
    Reads the perfectly pre-cropped tile TIFFs and saves them directly 
    into the tile's static_features group.
    """
    h5_path = Path(h5_path)
    print(f"\n{'='*60}\nRunning Tile-Based Topography for: {h5_path.name}")

    start_topography = time.time()
    
    with h5py.File(h5_path, "a") as f:
        # Identify all tiles in this file
        tile_ids = [k for k in f.keys() if k.startswith('tile_')]
        
        if not tile_ids:
            print("No tiles found. Skipping.")
            return
            
        # Iterate through each tile
        for tile_id in tile_ids:
            print(f"--- Processing {tile_id} ---")
            
            # Ensure static group exists for this specific tile
            static_grp = f[tile_id].require_group('static_features')
            
            # Load each variable from its corresponding folder
            for var_name, folder_path in dem_folders_dict.items():
                
                # Construct the exact filename the GDAL script generated
                # e.g., "tile_305_62_dem_90m.tif"
                tif_name = f"{tile_id}_{var_name}_{base_res}m.tif"
                tif_path = Path(folder_path) / tif_name
                
                if not tif_path.exists():
                    print(f"Skipping {var_name}: {tif_name} not found.")
                    continue
                
                # Open the TIFF and extract the raw array directly
                with rioxarray.open_rasterio(tif_path) as da:
                    # da.values gives the raw numpy array. 
                    # If it has a band dimension (1, 256, 256), we grab the first index [0]
                    grid_data = da.values[0] if da.ndim == 3 else da.values
                    
                # Save to HDF5 (using lowercase variable name)
                var_lower = var_name.lower()
                
                if var_lower in static_grp:
                    static_grp[var_lower][...] = grid_data
                else:
                    static_grp.create_dataset(var_lower, data=grid_data, compression="lzf")
                
                print(f"Saved {var_lower}.")

    print(f"Finished saving static topography in {time.time() - start_topography:.2f}s")