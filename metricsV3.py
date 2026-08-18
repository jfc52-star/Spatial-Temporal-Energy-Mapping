"""
metricsV2.py
------------
Building energy performance metrics for the Cambridge Estates dataset.

STRUCTURE
---------
  1. Configuration
  2. Private helpers      (_load_csv_dir, _load_gia_table, _get_gia, etc.)
  3. Metric functions     (compute_eui, compute_bi, compute_pli)
  4. Estate-wide runners  (compute_all_buildings, compute_all_buildings_temporal)
  5. Entry point

METRIC DEFINITIONS
------------------
  EUI  – Energy Use Intensity       = annual kWh / GIA               [kWh/m²/yr]
  BI   – Baseload Intensity         =median of lowest 10% of unoccupied-hour readings / GIA.   [W/m²]
  PLI  – Peak Load Intensity        = 99th-percentile load / GIA     [W/m²]

DATA LAYOUT
-----------
  processed_data/
  ├── building_floor_roof_areas.csv
  └── UCam_Building_b{id}/
      ├── electricity/
      │   └── {year}.csv    columns: datetime, equipment load [kWh]
      └── gas/
          └── {year}.csv    columns: datetime, heating load [kWh]
"""

import os
import re
import numpy as np
import pandas as pd

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
from config import DATA_DIR, WEATHER_DIR
# DATA_DIR = r'C:\Users\joshc\OneDrive\Documents\cambridge-energy\repo\building_data\processed_data'
# WEATHER_DIR = r'C:\Users\joshc\OneDrive\Documents\cambridge-energy\repo\aux_data\MetOffice Weather Data\processed_data\bedford'
# Hours treated as unoccupied for baseload calculation (00:00–06:00 default)
UNOCC_HOURS = range(0, 6)


# =============================================================================
# 2. PRIVATE HELPERS
# =============================================================================

def _load_csv_dir(directory: str, datetime_col: str) -> pd.DataFrame:
    """Concatenate every .csv in *directory* into one DataFrame."""
    csvs = [
        f for f in os.listdir(directory)
        if f.endswith('.csv') and os.path.isfile(os.path.join(directory, f))
    ]
    if not csvs:
        raise FileNotFoundError(f'No CSV files found in: {directory}')

    frames = [pd.read_csv(os.path.join(directory, f)) for f in csvs]
    df = pd.concat(frames, ignore_index=True)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.sort_values(datetime_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _load_gia_table(data_dir: str) -> pd.DataFrame:
    """Load building_floor_roof_areas.csv indexed by Building ID (e.g. 'b7')."""
    path = os.path.join(data_dir, 'building_floor_roof_areas.csv')
    df = pd.read_csv(path)
    df['Building ID'] = df['Building ID'].astype(str).str.strip()
    return df.set_index('Building ID')


def _get_gia(building_id: int, gia_table: pd.DataFrame) -> float:
    """Look up GIA for a single building."""
    key = f'b{building_id}'
    if key not in gia_table.index:
        raise KeyError(f'Building {key} not found in building_floor_roof_areas.csv')
    return float(gia_table.loc[key, 'GIA (m2)'])


def _load_elec(building_id: int, data_dir: str) -> pd.DataFrame | None:
    """Load all electricity CSVs for a building. Returns None if folder missing."""
    elec_dir = os.path.join(data_dir, f'UCam_Building_b{building_id}', 'electricity')
    if not os.path.exists(elec_dir):
        return None
    df = _load_csv_dir(elec_dir, 'datetime')
    df['hour'] = df['datetime'].dt.hour
    df['year'] = df['datetime'].dt.year
    return df


GAS_CV_SCALE = 11.13542

GAS_OM_OUT = {
    0, 1, 4, 5, 8, 13, 15, 16, 17, 18, 19, 20, 22, 24, 25, 27, 28, 30, 31,
    32, 33, 35, 36, 40, 44, 45, 46, 47, 52, 53, 55, 57, 58, 59, 60, 61,
    63, 64, 65, 68, 91, 96, 97, 99, 101, 102, 103, 104, 105, 108, 111, 112, 118,
}

GAS_CORRECT = {
    43, 50, 56, 62, 69, 70, 71, 72, 74, 75, 79, 85, 110, 113, 114,
    119, 120,
}

GAS_INCORRECT = {
    2, 7, 76, 77, 78, 80, 81, 82, 83, 84, 86, 87, 107, 109, 117,
}

UNCLASSIFIED_POLICY = 'exclude'   # 'exclude' or 'include_raw'

# Global default cutoff
GAS_START_DATE = pd.Timestamp('2018-01-01')

# Per-building overrides — applied instead of GAS_START_DATE for these
# buildings. Used where building-specific data quality issues require a
# later cutoff than the estate-wide default (see methodology).
GAS_START_OVERRIDES = {
    59: pd.Timestamp('2019-01-01'),
    40: pd.Timestamp('2019-01-01'),
    53: pd.Timestamp('2020-01-01'),
}

GAS_YEAR_EXCLUDE = {
    27: {2020},
}

def _load_gas(building_id: int, data_dir: str) -> pd.DataFrame | None:
    """Load all gas CSVs for a building, applying validation corrections
    and the start-date cutoff.

    Returns None if:
      - the gas folder is missing
      - the building is flagged GAS_INCORRECT
      - the building is unclassified and UNCLASSIFIED_POLICY == 'exclude'
      - no rows remain after the date cutoff

    The cutoff applied is GAS_START_OVERRIDES[building_id] if present,
    otherwise GAS_START_DATE.

    If the building is flagged GAS_OM_OUT, the 'heating load [kWh]'
    column is multiplied by GAS_CV_SCALE.
    """
    # ---- validation gate ----
    if building_id in GAS_INCORRECT:
        return None

    is_om = building_id in GAS_OM_OUT
    is_ok = building_id in GAS_CORRECT
    classified = is_om or is_ok

    if not classified and UNCLASSIFIED_POLICY == 'exclude':
        return None

    # ---- original loading logic ----
    gas_dir = os.path.join(data_dir, f'UCam_Building_b{building_id}', 'gas')
    if not os.path.exists(gas_dir):
        return None
    df = _load_csv_dir(gas_dir, 'datetime')
    df['hour'] = df['datetime'].dt.hour
    df['year'] = df['datetime'].dt.year

    # ---- date window cutoff (per-building override falls back to global) ----
    cutoff = GAS_START_OVERRIDES.get(building_id, GAS_START_DATE)
    df = df[df['datetime'] >= cutoff]
    if len(df) == 0:
        return None
    # ---- date window cutoff (per-building override falls back to global) ----
    cutoff = GAS_START_OVERRIDES.get(building_id, GAS_START_DATE)
    df = df[df['datetime'] >= cutoff]
    if len(df) == 0:
        return None

    # ---- per-building year exclusions ----
    excluded_years = GAS_YEAR_EXCLUDE.get(building_id)
    if excluded_years:
        df = df[~df['datetime'].dt.year.isin(excluded_years)]
        if len(df) == 0:
            return None

    # ---- apply OM-out scaling ----
    if is_om:
        df['heating load [kWh]'] = df['heating load [kWh]'] * GAS_CV_SCALE

    return df

def _yearly_mean(values_by_year: dict) -> float | None:
    """Mean across years, or None if dict is empty or None."""
    if not values_by_year:
        return None
    return round(float(np.mean(list(values_by_year.values()))), 3)

def _load_weather_year(year: int, weather_dir: str = WEATHER_DIR) -> pd.DataFrame | None:
    """Load hourly weather data for a single year. Returns None if missing.

    Returns a DataFrame with columns: datetime, air_temperature [degC]
    """
    path = os.path.join(weather_dir, f'{year}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=['datetime', 'air_temperature [degC]'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def _daily_mean_temperature(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather to daily mean temperature.

    Returns DataFrame indexed by date with a single column 't_ext'.
    """
    if weather_df is None or len(weather_df) == 0:
        return pd.DataFrame(columns=['t_ext'])
    daily = (weather_df
             .set_index('datetime')['air_temperature [degC]']
             .resample('D').mean()
             .rename('t_ext')
             .to_frame())
    daily.index = daily.index.date
    return daily

# =============================================================================
# 3. METRIC FUNCTIONS
# =============================================================================

def compute_eui(
    building_id: int,
    data_dir: str = DATA_DIR,
    gia_table: pd.DataFrame = None,
) -> dict:
    """Energy Use Intensity — annual kWh normalised by GIA.

    Formula : EUI = sum(hourly kWh over year) / GIA
    Units   : kWh / m² / yr
    """
    if gia_table is None:
        gia_table = _load_gia_table(data_dir)
    gia = _get_gia(building_id, gia_table)

    elec_df = _load_elec(building_id, data_dir)
    gas_df  = _load_gas(building_id, data_dir)

    eui_elec = None
    if elec_df is not None:
        annual   = elec_df.groupby('year')['equipment load [kWh]'].sum()
        eui_elec = (annual / gia).round(3).to_dict()

    eui_gas = None
    if gas_df is not None:
        annual  = gas_df.groupby('year')['heating load [kWh]'].sum()
        eui_gas = (annual / gia).round(3).to_dict()

    eui_combined       = None
    eui_combined_flag  = None   # 'both', 'elec_only', 'gas_only', or 'mixed'

    if eui_elec and eui_gas:
        all_years = set(eui_elec) | set(eui_gas)
        eui_combined = {
            yr: round(eui_elec.get(yr, 0) + eui_gas.get(yr, 0), 3)
            for yr in sorted(all_years)
        }
        # 'both' where both fuels present that year, else 'mixed'
        eui_combined_flag = {
            yr: 'both' if (yr in eui_elec and yr in eui_gas) else 'mixed'
            for yr in sorted(all_years)
        }
    elif eui_elec:
        eui_combined      = dict(eui_elec)
        eui_combined_flag = {yr: 'elec_only' for yr in eui_elec}
    elif eui_gas:
        eui_combined      = dict(eui_gas)
        eui_combined_flag = {yr: 'gas_only' for yr in eui_gas}
    
    return {
        'building_id':       building_id,
        'gia_m2':            gia,
        'eui_elec':          eui_elec,
        'eui_gas':           eui_gas,
        'eui_combined':      eui_combined,
        'mean_eui_elec':     _yearly_mean(eui_elec),
        'mean_eui_gas':      _yearly_mean(eui_gas),
        'mean_eui_combined': _yearly_mean(eui_combined),
    }


def compute_bi(
    building_id: int,
    data_dir: str = DATA_DIR,
    gia_table: pd.DataFrame = None,
    unocc_hours: range = UNOCC_HOURS,
) -> dict:
    """Baseload Intensity — median unoccupied-hour load normalised by GIA.

    Formula : BI = median(kWh during unoccupied hours) / GIA * 1000
    Units   : W / m²
    """
    if gia_table is None:
        gia_table = _load_gia_table(data_dir)
    gia = _get_gia(building_id, gia_table)

    elec_df = _load_elec(building_id, data_dir)
    gas_df  = _load_gas(building_id, data_dir)

    def _bi_by_year(df: pd.DataFrame, load_col: str) -> dict:
        unocc  = df[df['hour'].isin(unocc_hours)]
        result = {}
        for yr, grp in unocc.groupby('year'):
            vals = grp[load_col].dropna()
            if len(vals) == 0:
                continue
            threshold = vals.quantile(0.10)
            lowest_10pct = vals[vals <= threshold]
            result[yr] = round((lowest_10pct.median() / gia) * 1000, 3)
        return result

    bi_elec = _bi_by_year(elec_df, 'equipment load [kWh]') if elec_df is not None else None
    bi_gas  = _bi_by_year(gas_df,  'heating load [kWh]')   if gas_df  is not None else None

    return {
        'building_id':  building_id,
        'gia_m2':       gia,
        'bi_elec':      bi_elec,
        'bi_gas':       bi_gas,
        'mean_bi_elec': _yearly_mean(bi_elec),
        'mean_bi_gas':  _yearly_mean(bi_gas),
    }


def compute_pli(
    building_id: int,
    data_dir: str = DATA_DIR,
    gia_table: pd.DataFrame = None,
) -> dict:
    """Peak Load Intensity — 99th-percentile hourly load normalised by GIA.

    Formula : PLI = percentile(kWh, 99) / GIA * 1000
    Units   : W / m²
    """
    if gia_table is None:
        gia_table = _load_gia_table(data_dir)
    gia = _get_gia(building_id, gia_table)

    elec_df = _load_elec(building_id, data_dir)
    gas_df  = _load_gas(building_id, data_dir)

    def _pli_by_year(df: pd.DataFrame, load_col: str) -> dict:
        result = {}
        for yr, grp in df.groupby('year'):
            result[yr] = round((grp[load_col].quantile(0.99) / gia) * 1000, 3)
        return result

    pli_elec = _pli_by_year(elec_df, 'equipment load [kWh]') if elec_df is not None else None
    pli_gas  = _pli_by_year(gas_df,  'heating load [kWh]')   if gas_df  is not None else None

    return {
        'building_id':   building_id,
        'gia_m2':        gia,
        'pli_elec':      pli_elec,
        'pli_gas':       pli_gas,
        'mean_pli_elec': _yearly_mean(pli_elec),
        'mean_pli_gas':  _yearly_mean(pli_gas),
    }

def compute_hs(
    building_id: int,
    data_dir: str = DATA_DIR,
    weather_dir: str = WEATHER_DIR,
    gia_table: pd.DataFrame = None,
    balance_point_grid: tuple = tuple(np.arange(8.0, 20.01, 0.5)),
) -> dict:
    """Heating Sensitivity — piecewise linear fit of daily gas vs degree-days.

    Model: Q_gas(T) = Q0 + HS * max(0, T_bal - T_ext)

    Fit procedure:
      1. Aggregate hourly gas to daily kWh.
      2. Join with daily mean outdoor temperature.
      3. For each candidate balance point in the grid:
           transform T_ext → DD = max(0, T_bal - T_ext)
           fit ordinary least squares: Q_gas = Q0 + HS * DD
           record R² of the fit
      4. Return parameters from the balance point with highest R².

    Returns per-year dicts of slope (kWh/degree-day), balance point (°C),
    baseload Q₀ (kWh/day), and R² of the fit.

    Returns None entries for buildings with no gas supply.
    """
    if gia_table is None:
        gia_table = _load_gia_table(data_dir)
    gia = _get_gia(building_id, gia_table)

    gas_df = _load_gas(building_id, data_dir)
    if gas_df is None:
        return {
            'building_id':   building_id,
            'gia_m2':        gia,
            'hs_slope':      None,
            'hs_balance':    None,
            'hs_q0':         None,
            'hs_r2':         None,
            'mean_hs_slope': None,
            'mean_hs_balance': None,
        }

    # Daily gas per year
    gas_df['date'] = gas_df['datetime'].dt.date
    daily_gas = (gas_df
                 .groupby(['year', 'date'])['heating load [kWh]']
                 .sum()
                 .rename('q_gas')
                 .reset_index())

    hs_slope_by_year   = {}
    hs_balance_by_year = {}
    hs_q0_by_year      = {}
    hs_r2_by_year      = {}

    for year, grp in daily_gas.groupby('year'):
        weather_df = _load_weather_year(int(year), weather_dir)
        if weather_df is None:
            continue
        temp_daily = _daily_mean_temperature(weather_df)
        if len(temp_daily) == 0:
            continue

        # Join gas with temperature on date
        merged = grp.set_index('date').join(temp_daily, how='inner')
        merged = merged.dropna(subset=['q_gas', 't_ext'])
        if len(merged) < 30:       # need enough days for a stable fit
            continue

        q    = merged['q_gas'].values
        tvec = merged['t_ext'].values

        best_r2      = -np.inf
        best_params  = None
        for t_bal in balance_point_grid:
            dd = np.maximum(0.0, t_bal - tvec)
            # Skip candidate if DD is constant (all-zero) — no heating signal at this balance point
            if dd.std() == 0:
                continue
            # OLS: q = a + b * dd
            b, a = np.polyfit(dd, q, 1)      # slope, intercept
            q_pred = a + b * dd
            ss_res = np.sum((q - q_pred) ** 2)
            ss_tot = np.sum((q - q.mean()) ** 2)
            if ss_tot <= 0:
                continue
            r2 = 1 - ss_res / ss_tot
            # Only accept physically-sensible slopes (heating should increase gas use)
            if b < 0:
                continue
            if r2 > best_r2:
                best_r2 = r2
                best_params = (float(t_bal), float(b), float(a))

        if best_params is not None:
            t_bal, slope, q0 = best_params
            hs_slope_by_year[int(year)]   = round(slope, 3)
            hs_balance_by_year[int(year)] = round(t_bal, 2)
            hs_q0_by_year[int(year)]      = round(q0, 2)
            hs_r2_by_year[int(year)]      = round(best_r2, 3)

    # Cast to None if no years were fit
    def _or_none(d):
        return d if d else None

    return {
        'building_id':       building_id,
        'gia_m2':            gia,
        'hs_slope':          _or_none(hs_slope_by_year),
        'hs_balance':        _or_none(hs_balance_by_year),
        'hs_q0':             _or_none(hs_q0_by_year),
        'hs_r2':             _or_none(hs_r2_by_year),
        'mean_hs_slope':     _yearly_mean(hs_slope_by_year),
        'mean_hs_balance':   _yearly_mean(hs_balance_by_year),
    }

def compute_lf(
    building_id: int,
    data_dir: str = DATA_DIR,
    gia_table: pd.DataFrame = None,
) -> dict:
    """Load Factor — ratio of average load to peak load.

    Formula : LF = total_kWh / (peak_kW * hours_in_period)
    Units   : dimensionless, bounded 0 < LF <= 1

    High LF  = steady, flat operation (labs, 24/7 facilities)
    Low  LF  = spiky operation (teaching, infrequent high peaks)

    Peak uses the 99th-percentile hourly load (consistent with PLI) rather
    than the single max, to avoid sensor spikes distorting the ratio.
    """
    if gia_table is None:
        gia_table = _load_gia_table(data_dir)
    gia = _get_gia(building_id, gia_table)  # unused in LF itself but kept for API consistency

    elec_df = _load_elec(building_id, data_dir)
    gas_df  = _load_gas(building_id, data_dir)

    def _lf_by_year(df: pd.DataFrame, load_col: str) -> dict:
        result = {}
        for yr, grp in df.groupby('year'):
            vals = grp[load_col].dropna()
            if len(vals) == 0:
                continue
            total_kwh = vals.sum()
            peak_kw   = vals.quantile(0.99)   # assumes hourly data: kWh/h = kW
            hours     = len(vals)
            if peak_kw <= 0 or hours == 0:
                continue
            result[yr] = round(total_kwh / (peak_kw * hours), 3)
        return result

    lf_elec = _lf_by_year(elec_df, 'equipment load [kWh]') if elec_df is not None else None
    lf_gas  = _lf_by_year(gas_df,  'heating load [kWh]')   if gas_df  is not None else None

    return {
        'building_id':  building_id,
        'gia_m2':       gia,
        'lf_elec':      lf_elec,
        'lf_gas':       lf_gas,
        'mean_lf_elec': _yearly_mean(lf_elec),
        'mean_lf_gas':  _yearly_mean(lf_gas),
    }

def compute_fmr(
    building_id: int,
    data_dir: str = DATA_DIR,
    gia_table: pd.DataFrame = None,
) -> dict:
    """Fuel Mix Ratio — annual gas kWh / annual electricity kWh.

    Formula : FMR = gas_annual_kWh / elec_annual_kWh
    Units   : dimensionless

    FMR > 1  = gas-dominant (heating-heavy, decarbonisation exposure)
    FMR = 1  = parity
    FMR < 1  = electric-dominant (potentially already partly decarbonised)
    FMR = 0  = no gas supply (all-electric building)

    Only computed for years where BOTH fuels are present. Years with only
    one fuel return None for that year.
    """
    if gia_table is None:
        gia_table = _load_gia_table(data_dir)
    gia = _get_gia(building_id, gia_table)

    elec_df = _load_elec(building_id, data_dir)
    gas_df  = _load_gas(building_id, data_dir)

    fmr = None
    if elec_df is not None and gas_df is not None:
        elec_annual = elec_df.groupby('year')['equipment load [kWh]'].sum()
        gas_annual  = gas_df.groupby('year')['heating load [kWh]'].sum()
        common      = set(elec_annual.index) & set(gas_annual.index)
        fmr = {}
        for yr in sorted(common):
            e = elec_annual[yr]
            g = gas_annual[yr]
            if e <= 0:
                continue   # avoid div-by-zero; building has no elec that year
            fmr[yr] = round(g / e, 3)
    elif elec_df is not None and gas_df is None:
        # All-electric: FMR = 0 for every year with electricity data
        elec_annual = elec_df.groupby('year')['equipment load [kWh]'].sum()
        fmr = {int(yr): 0.0 for yr in elec_annual.index}

    return {
        'building_id': building_id,
        'gia_m2':      gia,
        'fmr':         fmr,
        'mean_fmr':    _yearly_mean(fmr),
    }

def compute_peak_timing(
    building_id: int,
    data_dir: str = DATA_DIR,
    gia_table: pd.DataFrame = None,
    top_pct: float = 0.05,
) -> dict:
    """Peak Timing — modal hour-of-day at which peak demand occurs.

    Formula : mode( hour | P > P_{1-top_pct} )
    Units   : hour of day (0-23), cyclic

    Takes the top 5% of hourly readings for each building-year and returns
    the most common hour-of-day among them. This captures *when* the
    building tends to peak, smoothing over one-off spikes.

    Returned value is a float in [0, 24) — typically integer, but could be
    non-integer if a tie is broken by averaging (not done here; mode picks
    the first mode).
    """
    if gia_table is None:
        gia_table = _load_gia_table(data_dir)
    gia = _get_gia(building_id, gia_table)

    elec_df = _load_elec(building_id, data_dir)
    gas_df  = _load_gas(building_id, data_dir)

    def _peak_hour_by_year(df: pd.DataFrame, load_col: str) -> dict:
        result = {}
        for yr, grp in df.groupby('year'):
            vals = grp[[load_col, 'hour']].dropna(subset=[load_col])
            if len(vals) == 0:
                continue
            threshold = vals[load_col].quantile(1.0 - top_pct)
            top_hours = vals.loc[vals[load_col] >= threshold, 'hour']
            if len(top_hours) == 0:
                continue
            # mode() can return multiple rows if tied; take first
            result[yr] = int(top_hours.mode().iloc[0])
        return result

    peak_hr_elec = _peak_hour_by_year(elec_df, 'equipment load [kWh]') if elec_df is not None else None
    peak_hr_gas  = _peak_hour_by_year(gas_df,  'heating load [kWh]')   if gas_df  is not None else None

    return {
        'building_id':        building_id,
        'gia_m2':             gia,
        'peak_hr_elec':       peak_hr_elec,
        'peak_hr_gas':        peak_hr_gas,
        'mean_peak_hr_elec':  _yearly_mean(peak_hr_elec),
        'mean_peak_hr_gas':   _yearly_mean(peak_hr_gas),
    }
# =============================================================================
# 4. ESTATE-WIDE RUNNERS
# =============================================================================

def compute_all_buildings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Compute mean metrics for every building — one row per building.

    Returns
    -------
    DataFrame with columns:
        building_id, gia_m2,
        mean_eui_elec, mean_eui_gas, mean_eui_combined,
        mean_bi_elec,  mean_bi_gas,
        mean_pli_elec, mean_pli_gas
    """
    pattern      = re.compile(r'UCam_Building_b(\d+)')
    building_ids = sorted([
        int(pattern.findall(d)[0])
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and pattern.match(d)
    ])
    gia_table = _load_gia_table(data_dir)
    records   = []

    for bid in building_ids:
        try:
            eui = compute_eui(bid, data_dir, gia_table)
            bi  = compute_bi( bid, data_dir, gia_table)
            pli = compute_pli(bid, data_dir, gia_table)
            lf   = compute_lf(         bid, data_dir, gia_table)
            fmr  = compute_fmr(        bid, data_dir, gia_table)
            pk   = compute_peak_timing(bid, data_dir, gia_table)

            records.append({
                'building_id':        bid,
                'gia_m2':             eui['gia_m2'],
                'mean_eui_elec':      eui['mean_eui_elec'],
                'mean_eui_gas':       eui['mean_eui_gas'],
                'mean_eui_combined':  eui['mean_eui_combined'],
                'mean_bi_elec':       bi['mean_bi_elec'],
                'mean_bi_gas':        bi['mean_bi_gas'],
                'mean_pli_elec':      pli['mean_pli_elec'],
                'mean_pli_gas':       pli['mean_pli_gas'],
                'mean_lf_elec':       lf['mean_lf_elec'],
                'mean_lf_gas':        lf['mean_lf_gas'],
                'mean_fmr':           fmr['mean_fmr'],
                'mean_peak_hr_elec':  pk['mean_peak_hr_elec'],
                'mean_peak_hr_gas':   pk['mean_peak_hr_gas'],
            })
        except Exception as e:
            print(f'  [WARNING] b{bid} skipped: {e}')

    return pd.DataFrame(records)


def compute_all_buildings_temporal(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Compute per-year metrics for every building — one row per building per year.

    This is the temporal version of compute_all_buildings(). Instead of
    returning mean values, it returns a row for every (building, year)
    combination that has data, enabling year-by-year comparison.

    Returns
    -------
    DataFrame with columns:
        building_id, year, gia_m2,
        eui_elec, eui_gas, eui_combined,
        bi_elec,  bi_gas,
        pli_elec, pli_gas
    """
    pattern      = re.compile(r'UCam_Building_b(\d+)')
    building_ids = sorted([
        int(pattern.findall(d)[0])
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and pattern.match(d)
    ])
    gia_table = _load_gia_table(data_dir)
    records   = []

    for bid in building_ids:
        try:
            eui = compute_eui(bid, data_dir, gia_table)
            bi  = compute_bi( bid, data_dir, gia_table)
            pli = compute_pli(bid, data_dir, gia_table)
            lf   = compute_lf(         bid, data_dir, gia_table)
            fmr  = compute_fmr(        bid, data_dir, gia_table)
            pk   = compute_peak_timing(bid, data_dir, gia_table)
            hs = compute_hs(bid, data_dir, gia_table=gia_table)

            # Collect all years present across any metric for this building
            all_years = set()
            for d in [eui['eui_elec'], eui['eui_gas'], eui['eui_combined'],
                      bi['bi_elec'], bi['bi_gas'],
                      pli['pli_elec'], pli['pli_gas'],
                      lf['lf_elec'], lf['lf_gas'],
                      fmr['fmr'],
                      pk['peak_hr_elec'], pk['peak_hr_gas'],
                      hs['hs_slope'], hs['hs_balance']]:
                if d:
                    all_years.update(d.keys())

            for yr in sorted(all_years):
                records.append({
                    'building_id':  bid,
                    'year':         yr,
                    'gia_m2':       eui['gia_m2'],
                    # EUI
                    'eui_elec':     eui['eui_elec'].get(yr)     if eui['eui_elec']     else None,
                    'eui_gas':      eui['eui_gas'].get(yr)      if eui['eui_gas']      else None,
                    'eui_combined': eui['eui_combined'].get(yr) if eui['eui_combined'] else None,
                    # BI
                    'bi_elec':      bi['bi_elec'].get(yr)       if bi['bi_elec']       else None,
                    'bi_gas':       bi['bi_gas'].get(yr)        if bi['bi_gas']        else None,
                    # PLI
                    'pli_elec':     pli['pli_elec'].get(yr)     if pli['pli_elec']     else None,
                    'pli_gas':      pli['pli_gas'].get(yr)      if pli['pli_gas']      else None,
                    # LF
                    'lf_elec':      lf['lf_elec'].get(yr)       if lf['lf_elec']       else None,
                    'lf_gas':       lf['lf_gas'].get(yr)        if lf['lf_gas']        else None,
                    # FMR
                    'fmr':          fmr['fmr'].get(yr)          if fmr['fmr']          else None,
                    # Peak timing
                    'peak_hr_elec': pk['peak_hr_elec'].get(yr)  if pk['peak_hr_elec']  else None,
                    'peak_hr_gas':  pk['peak_hr_gas'].get(yr)   if pk['peak_hr_gas']   else None,
                    # HS
                    'hs_slope':     hs['hs_slope'].get(yr)    if hs['hs_slope']    else None,
                    'hs_q0':        hs['hs_q0'].get(yr)       if hs['hs_q0']       else None,
                    'hs_balance':   hs['hs_balance'].get(yr)  if hs['hs_balance']  else None,
                    'hs_r2':        hs['hs_r2'].get(yr)       if hs['hs_r2']       else None,
                })
        except Exception as e:
            print(f'  [WARNING] b{bid} skipped: {e}')

    return pd.DataFrame(records)


# Backwards-compatible alias
def compute_eui_all_buildings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Alias for compute_all_buildings() — kept for backwards compatibility."""
    return compute_all_buildings(data_dir)


# =============================================================================
# 5. ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print('Computing mean metrics...\n')
    df = compute_all_buildings()
    df.sort_values('mean_eui_combined', ascending=False, inplace=True)
    print(df.to_string(index=False))
    df.to_csv('metrics_results.csv', index=False)
    print('\nSaved to metrics_results.csv')

    print('\nComputing temporal metrics...\n')
    df_t = compute_all_buildings_temporal()
    print(f'Rows: {len(df_t)}  (buildings × years)')
    df_t.to_csv('metrics_results_temporal.csv', index=False)
    print('Saved to metrics_results_temporal.csv')