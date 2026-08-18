"""
geocode_postcodes.py
--------------------
Reads Building_Mapping.xlsx, looks up each postcode using the free
postcodes.io API (no account or key needed), and writes a CSV of
building centroids ready for eui_to_gis.py.

Run once:
    pixi run python geocode_postcodes.py

Output:
    building_centroids.csv  (columns: building_id, postcode, latitude, longitude)
"""

import os
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION - edit if your file is somewhere else
# ---------------------------------------------------------------------------
MAPPING_XLSX  = os.path.join('Building_Mapping.xlsx')
SHEET_NAME    = 'Conversion'
ID_COL        = 'Building ID'   # e.g. b0, b1 ...
POSTCODE_COL  = 'Post Code'
OUTPUT_CSV    = 'building_centroids.csv'


# ---------------------------------------------------------------------------
# Geocode a single postcode via postcodes.io
# ---------------------------------------------------------------------------

def lookup_postcode(postcode: str) -> tuple:
    """Return (latitude, longitude) for a UK postcode, or (None, None) on failure."""
    url = f'https://api.postcodes.io/postcodes/{postcode.replace(" ", "")}'
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            result = r.json()['result']
            return result['latitude'], result['longitude']
        else:
            print(f'  [WARN] Postcode not found: {postcode} (status {r.status_code})')
            return None, None
    except Exception as e:
        print(f'  [WARN] Error looking up {postcode}: {e}')
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # Load the mapping file
    print(f'Reading {MAPPING_XLSX}...')
    df = pd.read_excel(MAPPING_XLSX, sheet_name=SHEET_NAME)

    # Keep only the columns we need and clean them up
    df = df[[ID_COL, POSTCODE_COL]].copy()
    df.columns = ['building_id', 'postcode']
    df['building_id'] = df['building_id'].astype(str).str.strip()
    df['postcode']    = df['postcode'].astype(str).str.strip()

    print(f'Found {len(df)} buildings. Geocoding postcodes...\n')

    latitudes  = []
    longitudes = []

    for _, row in df.iterrows():
        lat, lon = lookup_postcode(row['postcode'])
        latitudes.append(lat)
        longitudes.append(lon)
        print(f'  {row["building_id"]:6s}  {row["postcode"]:10s}  ->  {lat}, {lon}')
        time.sleep(0.1)  # be polite to the free API

    df['latitude']  = latitudes
    df['longitude'] = longitudes

    # Report any failures
    missing = df[df['latitude'].isna()]
    if len(missing) > 0:
        print(f'\n[WARNING] Could not geocode {len(missing)} buildings:')
        print(missing[['building_id', 'postcode']].to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f'\nSaved to {OUTPUT_CSV}')
    print(f'Successfully geocoded: {df["latitude"].notna().sum()} / {len(df)} buildings')
