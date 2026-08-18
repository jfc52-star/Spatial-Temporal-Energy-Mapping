"""
config.py
---------
Central path configuration. Edit PROJECT_ROOT here and the rest of the
scripts will pick up the new location automatically.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data locations (all relative to PROJECT_ROOT)
DATA_DIR           = os.path.join(PROJECT_ROOT, 'repo', 'building_data', 'processed_data')
WEATHER_DIR        = os.path.join(PROJECT_ROOT, 'repo', 'aux_data', 'MetOffice Weather Data', 'processed_data', 'bedford')
CENTROID_CSV_PATH  = os.path.join(PROJECT_ROOT, 'building_centroids.csv')
METADATA_XLSX_PATH = os.path.join(PROJECT_ROOT, 'Building_Mapping.xlsx')
SOLAR_DIR         = os.path.join(PROJECT_ROOT, 'repo', 'aux_data', 'RenewablesNinja Generation Data', 'processed_data')