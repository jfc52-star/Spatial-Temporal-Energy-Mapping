"""
non_spatial_metrics.py
----------------------
Non-spatial derived quantities for the Cambridge Estates dataset.

Where `metricsV3.py` produces one-scalar-per-building-per-year metrics
(EUI, BI, PLI, LF, FMR, Peak Timing), this module produces the
*shape-ful* outputs that don't fit on a map but are needed for the
non-spatial views:

    - Metadata table      : function type, age, tenure, construction era, ...
    - Daily load profiles : 24-value vectors per building/year/fuel/day-type
    - CLDC summaries      : scalar peakiness / concentration metrics
    - Distribution summaries : estate-wide percentiles per metric per year

STRUCTURE
---------
    1. Configuration
    2. Metadata loading       (_load_metadata_table)
    3. Daily load profiles    (compute_daily_profile, compute_all_profiles)
    4. CLDC summaries         (compute_cldc_summary, compute_all_cldc)
    5. Distribution summaries (compute_distribution_summary)
    6. Entry point

All heavy-lifting helpers (_load_csv_dir, _load_elec, _load_gas,
_load_gia_table, _yearly_mean) are imported from metricsV3 to avoid
duplication.
"""

import os
import numpy as np
import pandas as pd

from metricsV3 import (
    DATA_DIR,
    _load_elec,
    _load_gas,
    _load_gia_table,
    _yearly_mean,
    compute_all_buildings_temporal,
    _load_weather_year,
    _daily_mean_temperature,
    WEATHER_DIR,
)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
from config import METADATA_XLSX_PATH
# METADATA_XLSX_PATH = r'C:\Users\joshc\OneDrive\Documents\cambridge-energy\Building_Mapping.xlsx'
METADATA_SHEET     = 'Conversion'

ELEC_COL = 'equipment load [kWh]'
GAS_COL  = 'heating load [kWh]'

# Season definition (month ranges).  Summer = Jun-Aug, Winter = Dec-Feb,
# Shoulder = rest. This is a common split for UK building analysis; adjust
# if your dissertation uses a different convention.
SEASONS = {
    'summer':   (6, 7, 8),
    'winter':   (12, 1, 2),
    'shoulder': (3, 4, 5, 9, 10, 11),
    'all':      tuple(range(1, 13)),
}

# Construction era bins — used for age-based comparison in the similarity view.
# Edges are inclusive-lower, exclusive-upper.
ERA_BINS   = [0, 1900, 1945, 1975, 2000, 2015, 3000]
ERA_LABELS = ['pre-1900', '1900-1944', '1945-1974', '1975-1999', '2000-2014', '2015+']


# =============================================================================
# 2. METADATA LOADING
# =============================================================================

def _load_metadata_table(
    path: str = METADATA_XLSX_PATH,
    sheet: str = METADATA_SHEET,
) -> pd.DataFrame:
    """Load Building_Mapping.xlsx — one row per building.

    Returns DataFrame indexed by 'building_id' (e.g. 'b7') with columns:
        uprn, name, address, post_code, naa_m2, balance_m2, gia_m2,
        n_floors, date_built, tenure, condition, function,
        has_elec, has_gas, oldest_year, newest_year, era
    """
    df = pd.read_excel(path, sheet_name=sheet)

    # Normalise column names — the Excel uses spaces and parentheses
    rename = {
        'Building ID':         'building_id',
        'UPRN':                'uprn',
        'Contact ID':          'contact_id',
        'Point ID (elec)':     'point_id_elec',
        'Point ID (gas)':      'point_id_gas',
        'Building Code':       'building_code',
        'Name (as per CEBD)':  'name_cebd',
        'Name (as per summary)': 'name_summary',
        'Address':             'address',
        'Post Code':           'post_code',
        'NAA (m2)':            'naa_m2',
        'Balance (m2)':        'balance_m2',
        'GIA (m2)':            'gia_m2',
        'No. Floors':          'n_floors',
        'Date Built':          'date_built',
        'Tenure':              'tenure',
        'Condition':           'condition',
        'Function':            'function',
        'electricity':         'has_elec',
        'gas':                 'has_gas',
        'water':               'has_water',
        'other':               'has_other',
        'oldest':              'oldest_year',
        'newest':              'newest_year',
        'Notes':               'notes',
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Clean + coerce
    df['building_id'] = df['building_id'].astype(str).str.strip()
    for col in ['date_built', 'n_floors', 'oldest_year', 'newest_year',
                'naa_m2', 'balance_m2', 'gia_m2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalise string columns (strip + title-ish) — light touch, don't
    # lose the original casing entirely in case it matters downstream
    for col in ['function', 'tenure', 'condition', 'post_code']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(['nan', 'NaN', '']), col] = None

    # Construction era
    if 'date_built' in df.columns:
        df['era'] = pd.cut(
            df['date_built'],
            bins=ERA_BINS,
            labels=ERA_LABELS,
            right=False,
            include_lowest=True,
        ).astype(str)
        df.loc[df['date_built'].isna(), 'era'] = None

    return df.set_index('building_id')


# =============================================================================
# 3. DAILY LOAD PROFILES
# =============================================================================

def _season_of_month(m: int) -> str:
    for season, months in SEASONS.items():
        if season == 'all':
            continue
        if m in months:
            return season
    return 'shoulder'  # fallback


def _profile_from_df(
    df: pd.DataFrame,
    load_col: str,
    year: int,
    season: str,
    daytype: str,
) -> np.ndarray | None:
    """Return a 24-vector of mean hourly load, filtered to year/season/daytype.

    daytype in {'all', 'weekday', 'weekend'}.
    Returns None if no data survives the filter.
    """
    sub = df[df['year'] == year].copy()
    if season != 'all':
        sub = sub[sub['datetime'].dt.month.isin(SEASONS[season])]
    if daytype == 'weekday':
        sub = sub[sub['datetime'].dt.dayofweek < 5]
    elif daytype == 'weekend':
        sub = sub[sub['datetime'].dt.dayofweek >= 5]

    if len(sub) == 0:
        return None

    profile = sub.groupby('hour')[load_col].mean()
    # Ensure all 24 hours present (fill gaps with NaN, then interpolate)
    profile = profile.reindex(range(24))
    if profile.isna().all():
        return None
    profile = profile.interpolate(limit_direction='both')
    return profile.values.astype(float)


def _normalise_profile(profile: np.ndarray) -> dict:
    """Return three normalisations of a 24-value daily profile."""
    if profile is None or len(profile) == 0 or np.all(np.isnan(profile)):
        return {'profile_abs': None, 'profile_sum1': None, 'profile_unitvar': None}

    abs_profile = np.round(profile, 4).tolist()

    # Sum-to-1 (shape as probability distribution — standard for clustering)
    total = float(np.nansum(profile))
    sum1  = (profile / total).round(6).tolist() if total > 0 else None

    # Unit variance z-score (user's original suggestion)
    mu, sd = float(np.nanmean(profile)), float(np.nanstd(profile))
    unitvar = ((profile - mu) / sd).round(4).tolist() if sd > 0 else None

    return {
        'profile_abs':     abs_profile,
        'profile_sum1':    sum1,
        'profile_unitvar': unitvar,
    }


def compute_daily_profile(
    building_id: int,
    data_dir: str = DATA_DIR,
    seasons: tuple = ('all', 'summer', 'winter', 'shoulder'),
    daytypes: tuple = ('all', 'weekday', 'weekend'),
) -> list[dict]:
    """Daily 24-hour load profiles for one building, across fuels/years/seasons/daytypes.

    Returns a list of dicts — one per (fuel, year, season, daytype) combination
    that has data. Each dict contains all three normalisations of the profile.

    Intended to be stacked into a DataFrame by compute_all_profiles().
    """
    elec_df = _load_elec(building_id, data_dir)
    gas_df  = _load_gas(building_id, data_dir)

    records = []

    def _harvest(df, load_col, fuel):
        if df is None:
            return
        for yr in sorted(df['year'].unique()):
            for season in seasons:
                for daytype in daytypes:
                    prof = _profile_from_df(df, load_col, yr, season, daytype)
                    if prof is None:
                        continue
                    norm = _normalise_profile(prof)
                    records.append({
                        'building_id': building_id,
                        'fuel':        fuel,
                        'year':        int(yr),
                        'season':      season,
                        'daytype':     daytype,
                        **norm,
                    })

    _harvest(elec_df, ELEC_COL, 'elec')
    _harvest(gas_df,  GAS_COL,  'gas')
    return records


def compute_all_profiles(
    data_dir: str = DATA_DIR,
    seasons: tuple = ('all', 'summer', 'winter', 'shoulder'),
    daytypes: tuple = ('all', 'weekday', 'weekend'),
) -> pd.DataFrame:
    """Daily profiles for every building, every fuel, every year/season/daytype.

    Returns DataFrame with columns:
        building_id, fuel, year, season, daytype,
        profile_abs, profile_sum1, profile_unitvar

    Note: profile columns contain Python lists (length 24) rather than
    separate h0..h23 columns. This keeps the DataFrame compact and makes
    downstream clustering straightforward (stack to ndarray with
    np.vstack(df['profile_sum1'].values)).
    """
    import re
    pattern = re.compile(r'UCam_Building_b(\d+)')
    building_ids = sorted([
        int(pattern.findall(d)[0])
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and pattern.match(d)
    ])

    all_records = []
    for bid in building_ids:
        try:
            all_records.extend(compute_daily_profile(bid, data_dir, seasons, daytypes))
        except Exception as e:
            print(f'  [WARNING] profile b{bid} skipped: {e}')

    return pd.DataFrame(all_records)


# =============================================================================
# 4. CLDC SUMMARIES
# =============================================================================

def _cldc_summary_from_series(vals: pd.Series) -> dict:
    """Scalar summaries of a single year's hourly loads.

    Given ranked (desc) loads, the CLDC tells you how many hours the
    building operates above any given load level. We extract:

        - top_5pct_share   : fraction of annual kWh from top 5% of hours
        - top_20pct_share  : fraction of annual kWh from top 20% of hours
        - hrs_above_50pct  : hours per year above 50% of annual peak
        - hrs_above_80pct  : hours per year above 80% of annual peak
        - peak_to_median   : ratio of 99th-pctile load to median load
        - load_concentration_gini : Gini coefficient of the load distribution
                                    (0 = flat, 1 = all energy in one hour)

    Gini is a nice single number for "how spiky" the operation is —
    comparable across buildings regardless of absolute scale.
    """
    vals = vals.dropna()
    if len(vals) == 0:
        return {
            'top_5pct_share':          None,
            'top_20pct_share':         None,
            'hrs_above_50pct':         None,
            'hrs_above_80pct':         None,
            'peak_to_median':          None,
            'load_concentration_gini': None,
        }

    sorted_vals = np.sort(vals.values)[::-1]   # descending
    n           = len(sorted_vals)
    total       = float(sorted_vals.sum())

    if total <= 0:
        return {k: None for k in [
            'top_5pct_share', 'top_20pct_share',
            'hrs_above_50pct', 'hrs_above_80pct',
            'peak_to_median', 'load_concentration_gini',
        ]}

    n_top5   = max(1, int(round(n * 0.05)))
    n_top20  = max(1, int(round(n * 0.20)))

    peak_99  = float(vals.quantile(0.99))
    median   = float(vals.median())

    # Gini coefficient — efficient formula on the sorted (asc) array
    asc = np.sort(vals.values)
    cum = np.cumsum(asc)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n if cum[-1] > 0 else None

    return {
        'top_5pct_share':          round(float(sorted_vals[:n_top5].sum()  / total), 4),
        'top_20pct_share':         round(float(sorted_vals[:n_top20].sum() / total), 4),
        'hrs_above_50pct':         int(np.sum(vals >= 0.50 * peak_99)),
        'hrs_above_80pct':         int(np.sum(vals >= 0.80 * peak_99)),
        'peak_to_median':          round(peak_99 / median, 3) if median > 0 else None,
        'load_concentration_gini': round(float(gini), 4) if gini is not None else None,
    }


def compute_cldc_summary(
    building_id: int,
    data_dir: str = DATA_DIR,
) -> dict:
    """CLDC-derived scalar summaries for one building, per year, per fuel.

    Returns nested dict:
        {
          'cldc_elec': {year: {'top_5pct_share': ..., ...}, ...},
          'cldc_gas':  {year: {...}, ...},
        }
    """
    elec_df = _load_elec(building_id, data_dir)
    gas_df  = _load_gas(building_id, data_dir)

    def _per_year(df, load_col):
        if df is None:
            return None
        result = {}
        for yr, grp in df.groupby('year'):
            result[int(yr)] = _cldc_summary_from_series(grp[load_col])
        return result

    return {
        'building_id': building_id,
        'cldc_elec':   _per_year(elec_df, ELEC_COL),
        'cldc_gas':    _per_year(gas_df,  GAS_COL),
    }


def compute_all_cldc(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """CLDC summaries for every building — one row per (building, year, fuel).

    Returns DataFrame with columns:
        building_id, year, fuel,
        top_5pct_share, top_20pct_share,
        hrs_above_50pct, hrs_above_80pct,
        peak_to_median, load_concentration_gini
    """
    import re
    pattern = re.compile(r'UCam_Building_b(\d+)')
    building_ids = sorted([
        int(pattern.findall(d)[0])
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and pattern.match(d)
    ])

    records = []
    for bid in building_ids:
        try:
            out = compute_cldc_summary(bid, data_dir)
            for fuel_key, fuel_label in [('cldc_elec', 'elec'), ('cldc_gas', 'gas')]:
                per_year = out[fuel_key]
                if not per_year:
                    continue
                for yr, summary in per_year.items():
                    records.append({
                        'building_id': bid,
                        'year':        yr,
                        'fuel':        fuel_label,
                        **summary,
                    })
        except Exception as e:
            print(f'  [WARNING] CLDC b{bid} skipped: {e}')

    return pd.DataFrame(records)

# =============================================================================
# 5. ENERGY SIGNATURE DAILY DATA
# =============================================================================

def compute_daily_signatures(
    data_dir: str = DATA_DIR,
    weather_dir: str = WEATHER_DIR,
) -> pd.DataFrame:
    """Daily (building, year, date, t_ext, q_gas) rows.

    One row per (building, day). Source data for energy-signature scatter
    plots — y = q_gas (kWh/day), x = t_ext (°C daily mean).

    Buildings without gas supply are excluded. Buildings without temperature
    coverage for a given year are excluded for that year.

    Returns DataFrame with columns:
        building_id, year, date, t_ext, q_gas
    """
    import re
    pattern = re.compile(r'UCam_Building_b(\d+)')
    building_ids = sorted([
        int(pattern.findall(d)[0])
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and pattern.match(d)
    ])

    # Cache weather per year to avoid re-reading from disk
    weather_cache = {}
    def _get_weather(year):
        if year not in weather_cache:
            w = _load_weather_year(year, weather_dir)
            weather_cache[year] = _daily_mean_temperature(w) if w is not None else None
        return weather_cache[year]

    records = []
    for bid in building_ids:
        gas_df = _load_gas(bid, data_dir)
        if gas_df is None:
            continue

        gas_df['date'] = gas_df['datetime'].dt.date
        daily_gas = (gas_df
                     .groupby(['year', 'date'])['heating load [kWh]']
                     .sum()
                     .rename('q_gas')
                     .reset_index())

        for year, grp in daily_gas.groupby('year'):
            temp_daily = _get_weather(int(year))
            if temp_daily is None or len(temp_daily) == 0:
                continue
            merged = grp.set_index('date').join(temp_daily, how='inner').reset_index()
            if len(merged) == 0:
                continue
            for _, row in merged.iterrows():
                records.append({
                    'building_id': int(bid),
                    'year':        int(year),
                    'date':        str(row['date']),
                    't_ext':       round(float(row['t_ext']), 2),
                    'q_gas':       round(float(row['q_gas']), 2),
                })

    return pd.DataFrame(records)

# =============================================================================
# 6. DISTRIBUTION SUMMARIES
# =============================================================================

# Metrics from metricsV3 that we want estate-wide distributions for.
# Kept as a module-level list so view builders (distribution_dashboard.py)
# can import and use it directly.
DISTRIBUTION_METRICS = [
    'eui_combined', 'eui_elec', 'eui_gas',
    'bi_elec',  'bi_gas',
    'pli_elec', 'pli_gas',
    'lf_elec',  'lf_gas',
    'fmr',
    'peak_hr_elec', 'peak_hr_gas',
]


def compute_distribution_summary(
    temporal_df: pd.DataFrame = None,
    metrics: list = None,
    data_dir: str = DATA_DIR,
) -> pd.DataFrame:
    """Per-metric, per-year summary statistics across the estate.

    Useful for the distribution dashboard and for contextualising a single
    building against the estate-wide spread.

    Returns DataFrame with columns:
        year, metric,
        n, mean, std,
        p05, p25, p50, p75, p95,
        min, max

    If temporal_df is not supplied, it is computed fresh via
    compute_all_buildings_temporal(). Passing a pre-computed one is
    much faster if you're calling this repeatedly.
    """
    if temporal_df is None:
        temporal_df = compute_all_buildings_temporal(data_dir=data_dir)
    if metrics is None:
        metrics = [m for m in DISTRIBUTION_METRICS if m in temporal_df.columns]

    records = []
    for yr, grp in temporal_df.groupby('year'):
        for metric in metrics:
            if metric not in grp.columns:
                continue
            vals = grp[metric].dropna()
            if len(vals) == 0:
                continue
            records.append({
                'year':   int(yr),
                'metric': metric,
                'n':      int(len(vals)),
                'mean':   round(float(vals.mean()), 3),
                'std':    round(float(vals.std()),  3),
                'p05':    round(float(vals.quantile(0.05)), 3),
                'p25':    round(float(vals.quantile(0.25)), 3),
                'p50':    round(float(vals.quantile(0.50)), 3),
                'p75':    round(float(vals.quantile(0.75)), 3),
                'p95':    round(float(vals.quantile(0.95)), 3),
                'min':    round(float(vals.min()), 3),
                'max':    round(float(vals.max()), 3),
            })

    return pd.DataFrame(records)


# =============================================================================
# 7. ENTRY POINT
# =============================================================================

if __name__ == '__main__':

    print('Loading metadata table...')
    meta = _load_metadata_table()
    print(f'  {len(meta)} buildings in mapping')
    print(f'  Functions: {meta["function"].value_counts().head(10).to_dict()}')
    print(f'  Eras:      {meta["era"].value_counts().to_dict()}')
    meta.to_csv('metadata_table.csv')
    print('  Saved to metadata_table.csv')

    print('\nComputing distribution summaries...')
    dist = compute_distribution_summary()
    print(f'  {len(dist)} (year, metric) combinations')
    dist.to_csv('distribution_summary.csv', index=False)
    print('  Saved to distribution_summary.csv')

    print('\nComputing CLDC summaries (this may take a minute)...')
    cldc = compute_all_cldc()
    print(f'  {len(cldc)} (building, year, fuel) rows')
    cldc.to_csv('cldc_summary.csv', index=False)
    print('  Saved to cldc_summary.csv')

    print('\nComputing daily profiles (this may take several minutes)...')
    profiles = compute_all_profiles()
    print(f'  {len(profiles)} profile rows')
    # Profiles are lists — save as pickle to preserve structure
    profiles.to_pickle('daily_profiles.pkl')
    print('  Saved to daily_profiles.pkl')
    print('  (Use pd.read_pickle to reload — lists are preserved.)')
    print('\nComputing daily energy signatures (this may take a minute)...')
    if os.path.exists('daily_signatures.pkl'):
        print('  (cache exists — skipping)')
    else:
        sigs = compute_daily_signatures()
        print(f'  {len(sigs):,} (building, day) rows')
        sigs.to_pickle('daily_signatures.pkl')
        print('  Saved to daily_signatures.pkl')
    print('\nDone.')