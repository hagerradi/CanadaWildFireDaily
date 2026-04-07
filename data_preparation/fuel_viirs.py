from pathlib import Path
import ee
import time
import datetime
import requests
from rasterio.io import MemoryFile
import rioxarray
import xarray as xr
from tqdm import tqdm
import traceback
import h5py

try:
    ee.Initialize(project='widlfires-vegetation')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='widlfires-vegetation')

def get_gee_url_with_retry(ee_image, roi, max_retries=5):
    """
    Requests a download URL from GEE with exponential backoff.
    Injects the exact EPSG:4326 grid transform for perfect pixel alignment.
    """
    wait_time = 2
    
    for attempt in range(max_retries):
        try:
            url = ee_image.getDownloadURL({
                'crs': 'EPSG:4326', 
                'crs_transform': [0.004166666666666667, 0, -141.0, 
                                  0, -0.004166666666666667, 83.0],
                'region': roi,
                'format': 'GEO_TIFF'
            })
            return url
            
        except ee.EEException as e:
            error_msg = str(e).lower()
            if "too many" in error_msg or "capacity" in error_msg or "quota" in error_msg:
                print(f"GEE busy. Retrying in {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
                wait_time *= 2  
            else:
                raise e
                
    print("Failed to get GEE URL after max retries.")
    
    return None

def fetch_in_memory_raster(url):
    """
    Downloads GeoTIFF bytes and loads them into rioxarray in RAM.
    """
    response = requests.get(url)
    if response.status_code != 200:
        print(f"HTTP Error {response.status_code} during download.")
        return None, None, None
        
    memfile = MemoryFile(response.content)
    src = memfile.open()
    da = rioxarray.open_rasterio(src)
    return da, memfile, src

def run_daily_viirs_pipeline(h5_path):
    """
    Extracts dynamic NDVI and EVI for each day directly into RAM,
    fetching the composite from strictly before the current fire day.
    """
    fire_name = Path(h5_path).stem
    print(f"\n{'='*60}\nRunning Daily VIIRS Pipeline for: {fire_name}")
    
    with h5py.File(h5_path, "a") as f:
        # Geometry Setup
        lon_grid = f['geometry/theoretical_lon'][:]
        lat_grid = f['geometry/theoretical_lat'][:]
        
        buffer = 0.05
        min_lon, max_lon = lon_grid.min() - buffer, lon_grid.max() + buffer
        min_lat, max_lat = lat_grid.min() - buffer, lat_grid.max() + buffer
        roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
        
        x_coords = xr.DataArray(lon_grid, dims=("y", "x"))
        y_coords = xr.DataArray(lat_grid, dims=("y", "x"))
        
        days_grp = f['days']
        day_keys = sorted(days_grp.keys())
        if not day_keys: return

        # Time Window Calculation
        fire_year = int(f.attrs['year'])
        first_doy = int(days_grp[day_keys[0]].attrs['DOB'])
        last_doy  = int(days_grp[day_keys[-1]].attrs['DOB'])
        
        # Buffer the start date backward by 33 days to catch the previous 16-day composites
        start_date_ee = ee.Date.fromYMD(fire_year, 1, 1).advance(first_doy - 33, 'day')
        end_date_ee = ee.Date.fromYMD(fire_year, 1, 1).advance(last_doy, 'day')

        # Pre-fetch VIIRS composite timestamps (1 API Call)
        print("Fetching VIIRS composite schedule...")
        viirs_times_ms = ee.ImageCollection("NASA/VIIRS/002/VNP13A1") \
            .filterDate(start_date_ee, end_date_ee) \
            .aggregate_array('system:time_start') \
            .getInfo()
            
        viirs_times_ms = sorted(viirs_times_ms)
        
        # Cache locks
        cached_time_ms = None
        da_ndvi = None
        da_evi = None
        memfile = None
        src = None

        # Loop through Fire Days
        for day_key in day_keys:
            
            day_grp = days_grp[day_key]
            env_grp = day_grp.require_group('features')

            # Calculate the exact timestamp threshold using daily attribute
            current_doy = int(day_grp.attrs['DOB'])
            current_date_py = datetime.datetime(fire_year, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=current_doy - 1)
            target_limit_ms = int(current_date_py.timestamp() * 1000)
            
            # Find the most recent VIIRS image strictly before this day
            valid_times = [t for t in viirs_times_ms if t < target_limit_ms]
            if not valid_times:
                print(f"  -> No prior VIIRS image found for {day_key}, skipping...")
                continue
                
            target_time_ms = valid_times[-1]

            # The GEE Request Block (Only triggers if the 16-day window shifted)
            if target_time_ms != cached_time_ms:
                if memfile: 
                    src.close()
                    memfile.close() # Flush old TIFF from RAM
                
                # Ask GEE for the exact composite
                img = (ee.ImageCollection("NASA/VIIRS/002/VNP13A1")
                        .filter(ee.Filter.eq('system:time_start', target_time_ms))
                        .select(['NDVI', 'EVI'])
                        .first()
                        .clip(roi))
                
                url = get_gee_url_with_retry(img, roi)
                if not url: continue
                
                da, memfile, src = fetch_in_memory_raster(url)
                if da is None: continue
                
                # Select the bands (GEE keeps the order from `.select(['NDVI', 'EVI'])`)
                da_ndvi = da.sel(band=1)
                da_evi = da.sel(band=2)
                
                # Update the lock
                cached_time_ms = target_time_ms 
            
            # Calculate the grid values
            ndvi_v2 = da_ndvi.sel(x=x_coords, y=y_coords, method="nearest").compute().values
            evi_v2  = da_evi.sel(x=x_coords, y=y_coords, method="nearest").compute().values
            
            # Save into the daily features folder
            env_grp.require_dataset("ndvi", data=ndvi_v2, shape=ndvi_v2.shape, dtype=float, compression="lzf")
            env_grp.require_dataset("evi",  data=evi_v2,  shape=evi_v2.shape,  dtype=float, compression="lzf")
            
        # Final RAM Cleanup
        if memfile:
            src.close()
            memfile.close()

def process_all_fires_in_folder(folder_path):
    """
    Finds all .h5 files in the target folder and runs the daily VIIRS pipeline.
    """
    # Find all .h5 files
    h5_files = list(Path(folder_path).glob("*.h5"))
    print(f"Found {len(h5_files)} HDF5 files in {folder_path}\n" + "="*60)
    
    successful_fires = []
    failed_fires = []

    # Loop through the files
    for h5_file in tqdm(h5_files, desc="Batch Processing Fires"):
        try:
            # start_vegetation = time.time()
            
            run_daily_viirs_pipeline(h5_file)
            successful_fires.append(h5_file.name)

            fire_id = h5_file.stem.replace("fire_", "")
            
        except Exception as e:
            
            print(f"\n Error processing {h5_file.name}: {e}")
            traceback.print_exc() # Prints the exact line that caused the error for debugging
            failed_fires.append(h5_file.name)
            continue 

    # Print the final summary report
    print(f"\n{'='*60}\nBATCH PROCESSING COMPLETE")
    print(f"Successfully processed: {len(successful_fires)} fires")
    print(f"Failed to process: {len(failed_fires)} fires")
    
    if failed_fires:
        print("\nFailed files to review:")
        for failed in failed_fires:
            print(f" - {failed}")

def run_single_fire_viirs(h5_file):
    """
    Runs the daily VIIRS pipeline for a single .h5 file with error handling.
    """
    h5_file = Path(h5_file)
    
    if not h5_file.exists():
        print(f"Error: {h5_file} does not exist.")
        return False

    print(f"\n{'='*60}\nProcessing VIIRS for: {h5_file.name}")

    try:
        run_daily_viirs_pipeline(h5_file)
        print(f"Successfully processed VIIRS for: {h5_file.name}")
        return True

    except Exception as e:
        print(f"\n Error processing VIIRS for {h5_file.name}: {e}")
        traceback.print_exc() 
        return False