import subprocess
import sys
from pathlib import Path
import rasterio
from pyproj import Transformer
import pandas as pd

# # Add project root to sys.path to allow 'import config'
# root = Path(__file__).resolve().parent.parent.parent
# if str(root) not in sys.path:
#     sys.path.append(str(root))

from configs import settings

def generate_fire_terrain(
    fire_id, 
    fire_df, 
    vrt_path, 
    out_dem_dir, 
    out_slope_dir, 
    out_aspect_dir, 
    out_dem_avg_dir,
    buffer_deg=1.0,
    base_res=90,
    scale_factor=3
):
    """
    Crops a DEM from a VRT for a specific fire, and calculates Slope, Aspect, and a 3x3 Averaged DEM.
    """
    # Create directories
    for directory in [out_dem_dir, out_slope_dir, out_aspect_dir, out_dem_avg_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Buffered Lat/Lon bounding box
    min_lon = fire_df['lon'].min() - buffer_deg
    max_lon = fire_df['lon'].max() + buffer_deg
    min_lat = fire_df['lat'].min() - buffer_deg
    max_lat = fire_df['lat'].max() + buffer_deg

    # Convert the Lat/Lon box to EPSG:3347 Meters
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3347", always_xy=True)
    min_x_m, min_y_m = transformer.transform(min_lon, min_lat)
    max_x_m, max_y_m = transformer.transform(max_lon, max_lat)

    # Define all paths
    out_dem_path = str(Path(out_dem_dir) / f"{fire_id}_dem_{base_res}m.tif")
    out_slope_path = str(Path(out_slope_dir) / f"{fire_id}_slope_{base_res}m.tif")
    out_aspect_path = str(Path(out_aspect_dir) / f"{fire_id}_aspect_{base_res}m.tif")
    
    # Average
    out_dem_avg_path = str(Path(out_dem_avg_dir) / f"{fire_id}_dem_avg.tif")

    # Run GDAL 
    try:
        # STEP 1: Crop and project standard DEM from VRT
        print(f"[{fire_id}] 1/4: Cropping and projecting DEM to {base_res}m...")
        subprocess.run([
            "gdalwarp", 
            "-t_srs", "EPSG:3347",
            "-te", str(min_x_m), str(min_y_m), str(max_x_m), str(max_y_m),
            "-tr", str(base_res), str(base_res),
            "-tap", 
            "-r", "cubicspline",
            "-srcnodata", "-9999", 
            "-dstnodata", "-9999",
            "-overwrite",
            str(vrt_path), out_dem_path   
        ], check=True, capture_output=True, text=True)

        # STEP 2: Create the 3x3 Averaged DEM 
        print(f"[{fire_id}] 2/4: Averaging DEM 3x3...")
        
        # Determine the raw resolution of the VRT in degrees
        with rasterio.open(vrt_path) as src:
            orig_res_x, orig_res_y = src.res 
            # print(orig_res_x, orig_res_y)
            target_res_x = orig_res_x * scale_factor
            target_res_y = orig_res_y * scale_factor

        # Crop and Average the raw Lat/Lon data
        subprocess.run([
            "gdalwarp", 
            "-te", str(min_lon), str(min_lat), str(max_lon), str(max_lat), # Crop in degrees
            "-tr", str(target_res_x), str(target_res_y),                   # Scale in degrees
            "-r", "average", 
            "-ot", "Float32",
            "-tap",            # Snaps the grid globally
            "-overwrite", 
            str(vrt_path), out_dem_avg_path                                # Raw source to Raw output
        ], check=True, capture_output=True, text=True)

        # STEP 3: Calculate Slope
        print(f"[{fire_id}] 3/4: Calculating Slope...")
        subprocess.run([
            "gdaldem", "slope", 
            out_dem_path, out_slope_path,     
            # "-alg", "ZevenbergenThorne", 
            "-alg", "Horn",
            "-compute_edges"              
        ], check=True, capture_output=True, text=True)

        # STEP 4: Calculate Aspect
        print(f"[{fire_id}] 4/4: Calculating Aspect...")
        subprocess.run([
            "gdaldem", "aspect", 
            out_dem_path, out_aspect_path,
            # "-alg", "ZevenbergenThorne", 
            "-alg", "Horn",
            "-compute_edges",
            "-zero_for_flat"
        ], check=True, capture_output=True, text=True)
        
        # print(f"[{fire_id}] SUCCESS!.\n")

    except subprocess.CalledProcessError as e:
        
        print(f"[{fire_id}] ERROR: GDAL FAILED. Here is the exact error message:")
        print(e.stderr)

if __name__ == "__main__":
    
    # Grab the target year passed as an argument (Default to 2024 if none provided)
    if len(sys.argv) > 1:
        target_year = str(sys.argv[1])
    else:
        print("No year provided. Defaulting to '2024' for local testing.")
        target_year = '2024'

    # Load the dataset for the requested year
    csv_path = f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{target_year}/Firegrowth_pts_v1_1_{target_year}.csv'
    
    try:
        fire_growth_pts = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] Error: Could not find CSV for year {target_year} at {csv_path}")
        sys.exit(1)
    
    # Get the unique fires
    all_fire_ids = sorted(fire_growth_pts['ID'].unique())
    total_fires = len(all_fire_ids)
    print(target_year, total_fires)

    print(f"=== Starting Terrain Generation ===")
    print(f"Year: {target_year} | Total Fires to process: {total_fires}")
    print(f"===================================")

    # Loop through ALL fires for this year
    for i, target_fire_id in enumerate(all_fire_ids, start=1):
        
        print(f"[{i}/{total_fires}] Processing Fire ID: {target_fire_id}")
    
        # Memory Optimization: Filter the dataframe to ONLY this fire before passing it
        single_fire_df = fire_growth_pts[fire_growth_pts['ID'] == target_fire_id].copy()

        # Run the GDAL pipeline for this one fire
        generate_fire_terrain(
            fire_id=target_fire_id, 
            fire_df=single_fire_df,
            vrt_path=f"{settings.DEM_FOLDER}/dem_mosaic.vrt", 
            out_dem_dir=settings.ELEVATION_FOLDER, 
            out_slope_dir=settings.SLOPE_FOLDER, 
            out_aspect_dir=settings.ASPECT_FOLDER, 
            out_dem_avg_dir=settings.ELEVATION_AVG_FOLDER,
            buffer_deg=1.0,
            base_res=90,
            scale_factor=3
        )