from pathlib import Path

GRID_SIZE = 256
PIXEL_SIZE = 90
X_COL = 'easting'
Y_COL = 'northing'
LON_COL = 'lon'
LAT_COL = 'lat'

SUBSET_FEATURE_LIST = ['ID', 'lon', 'lat', 'easting', 'northing', 'fireday', 'year', 'DOB', 'firearea', 'cumuarea', 'prec', 'tmax', 'ws', 'rh', 
                        'dem', 'slope', 'aspect', 'Biomass', 'Closure', 'prcB', 'prcC']

# To be replaced with the absolute paths
PROJECT_FOLDER = 'Wildfires_Spread'
DATA_FOLDER = 'Data'
OUTPUT_FOLDER = 'Output'

H5_OUTPUT_FOLDER = f'{OUTPUT_FOLDER}/Fires_H5'
METADATA_FOLDER = f'{PROJECT_FOLDER}/fires_metadata'
SATELLITE_STATUS_FOLDER = f'{PROJECT_FOLDER}/Satellite_Status'

BASE_FOLDER = f'{DATA_FOLDER}/Fire_growth_points'
ERA5_FOLDER = f'{DATA_FOLDER}/ERA5'
SCANFI_FOLDER = f'{DATA_FOLDER}/SCANFI'

# DEM/Topography Folders
DEM_FOLDER = f'{DATA_FOLDER}/DEM_API'
DEM_TILES_FOLDER = f'{DATA_FOLDER}/DEM_API/DEM_Tiles'
ELEVATION_FOLDER = f'{DATA_FOLDER}/DEM_API/Elevation'
ELEVATION_AVG_FOLDER = f'{DATA_FOLDER}/DEM_API/Elevation_avg'
SLOPE_FOLDER = f'{DATA_FOLDER}/DEM_API/Slope'
ASPECT_FOLDER = f'{DATA_FOLDER}/DEM_API/Aspect'


# ==========================================
# AUTOMATIC DIRECTORY CREATION
# ==========================================
DIRS_TO_CREATE = [
    OUTPUT_FOLDER,
    H5_OUTPUT_FOLDER,
    METADATA_FOLDER,
    SATELLITE_STATUS_FOLDER,
    DEM_FOLDER,
    ELEVATION_FOLDER,
    ELEVATION_AVG_FOLDER,
    SLOPE_FOLDER,
    ASPECT_FOLDER
]

for d in DIRS_TO_CREATE:
    Path(d).mkdir(parents=True, exist_ok=True)