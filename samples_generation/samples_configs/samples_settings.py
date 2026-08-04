# samples_settings.py

# SEED
RANDOM_SEED = 42

# SIMPLE/TIMESERIES 
DYNAMIC_FEATURES = ['tmax', 'rh', 'ws', 'prec', 'u10', 'v10', 'evi', 'ndvi', 
                     's2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b11', 's2_b12', 's2_scl']
STATIC_FEATURES = ['dem', 'slope', 'aspect', 'biomass', 'closure', 'prcb', 'prcc']
TARGET_YEARS =  ['2020', '2021', '2022', '2023', '2024']

# TIMESERIES
SEQUENCE_LENGTH = 3

# TOGGLES
REMOVE_MISSING_DATA = True