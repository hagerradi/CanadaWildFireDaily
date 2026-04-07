GRID_SIZE = 256
PIXEL_SIZE = 90
X_COL = 'easting'
Y_COL = 'northing'
LON_COL = 'lon'
LAT_COL = 'lat'

SUBSET_FEATURE_LIST = ['ID', 'lon', 'lat', 'easting', 'northing', 'fireday', 'year', 'DOB', 'firearea', 'cumuarea', 'prec', 'tmax', 'ws', 'rh', 
                        'dem', 'slope', 'aspect', 'Biomass', 'Closure', 'prcB', 'prcC']

PROJECT_FOLDER = ''
DATA_FOLDER = ''
OUTPUT_FOLDER = ''


H5_OUTPUT_FOLDER = f'{OUTPUT_FOLDER}/Fires_H5'

METADATA_FOLDER = f'{PROJECT_FOLDER}/fires_metadata'

BASE_FOLDER = f'{DATA_FOLDER}/Fire_growth_points'
ERA5_FOLDER = f'{DATA_FOLDER}/ERA5'
SCANFI_FOLDER = f'{DATA_FOLDER}/SCANFI'
DEM_FOLDER = f'{DATA_FOLDER}/DEM_API'
ELEVATION_FOLDER = f'{DATA_FOLDER}/DEM_API/Elevation'
ELEVATION_AVG_FOLDER = f'{DATA_FOLDER}/DEM_API/Elevation_avg'
SLOPE_FOLDER = f'{DATA_FOLDER}/DEM_API/Slope'
ASPECT_FOLDER = f'{DATA_FOLDER}/DEM_API/Aspect'
DEM_TILES_FOLDER = f'{DATA_FOLDER}/DEM_API/DEM_Tiles'

SATELLITE_STATUS_FOLDER = f'{PROJECT_FOLDER}/Satellite_Status'