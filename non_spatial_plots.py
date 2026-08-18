"""
non_spatial_plots.py
--------------------
Single-page HTML with four views selectable from the top:

    - Distributions      : estate-wide distribution of each metric per year
    - CLDC               : cumulative-load-duration scalars per building per year
    - Daily Load Profile : 24-hour average profiles per building
    - Energy Signatures  : daily gas vs outdoor temperature

Features:
    - Cross-view building filter (type a list; all views respond)
    - Click any histogram bar → side panel lists buildings in that bin
    - Top-N button on Distributions and CLDC → ranked tables
    - Signatures filter-mode with optional estate cloud context
    - PNG/SVG export at dissertation-ready resolution
    - Short description block at top of each view
    - Clear axis titles with units on every plot
    - Legend for the profiles view

RUN
---
    pixi run python non_spatial_plots.py

Depends on:
    metrics_results_temporal.csv   (metricsV3.py)
    distribution_summary.csv       (non_spatial_metrics.py)
    cldc_summary.csv               (non_spatial_metrics.py)
    daily_profiles.pkl             (non_spatial_metrics.py)
    daily_signatures.pkl           (non_spatial_metrics.py)
    metadata_table.csv             (non_spatial_metrics.py)
"""

import os
import json
import numpy as np
import pandas as pd

from plotly.offline import get_plotlyjs_version

from metricsV3           import compute_all_buildings_temporal, DATA_DIR
from non_spatial_metrics import (
    compute_distribution_summary,
    compute_all_cldc,
    compute_all_profiles,
    compute_daily_signatures,
    _load_metadata_table,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_HTML = 'non_spatial.html'

CACHES = {
    'temporal':       'metrics_results_temporal.csv',
    'distributions':  'distribution_summary.csv',
    'cldc':           'cldc_summary.csv',
    'profiles':       'daily_profiles.pkl',
    'metadata':       'metadata_table.csv',
    'signatures':     'daily_signatures.pkl',
}

METRIC_GROUPS = {
    'Intensity': [
        ('eui_combined', 'EUI – Combined',    'kWh/m²/yr'),
        ('eui_elec',     'EUI – Electricity', 'kWh/m²/yr'),
        ('eui_gas',      'EUI – Gas',         'kWh/m²/yr'),
        ('bi_elec',      'BI – Electricity',  'W/m²'),
        ('bi_gas',       'BI – Gas',          'W/m²'),
        ('pli_elec',     'PLI – Electricity', 'W/m²'),
        ('pli_gas',      'PLI – Gas',         'W/m²'),
    ],
    'Load shape': [
        ('lf_elec',      'Load Factor – Elec', '—'),
        ('lf_gas',       'Load Factor – Gas',  '—'),
    ],
    'Fuel & carbon': [
        ('fmr',          'Fuel Mix Ratio',     'gas/elec'),
    ],
    'Timing': [
        ('peak_hr_elec', 'Peak Hour – Elec',   'hour'),
        ('peak_hr_gas',  'Peak Hour – Gas',    'hour'),
    ],
}

CLDC_METRICS = [
    ('top_5pct_share',          'Top 5% load share',       'fraction'),
    ('top_20pct_share',         'Top 20% load share',      'fraction'),
    ('hrs_above_50pct',         'Hours above 50% peak',    'h/yr'),
    ('hrs_above_80pct',         'Hours above 80% peak',    'h/yr'),
    ('peak_to_median',          'Peak-to-median ratio',    '—'),
    ('load_concentration_gini', 'Load-concentration Gini', '—'),
]

N_BINS = 25


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_or_compute_temporal() -> pd.DataFrame:
    if os.path.exists(CACHES['temporal']):
        print(f'  loading {CACHES["temporal"]}')
        return pd.read_csv(CACHES['temporal'])
    print('  computing temporal metrics (no cache)')
    df = compute_all_buildings_temporal(data_dir=DATA_DIR)
    df.to_csv(CACHES['temporal'], index=False)
    return df


def _load_or_compute_distributions(temporal_df: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(CACHES['distributions']):
        print(f'  loading {CACHES["distributions"]}')
        return pd.read_csv(CACHES['distributions'])
    print('  computing distribution summary (no cache)')
    df = compute_distribution_summary(temporal_df=temporal_df)
    df.to_csv(CACHES['distributions'], index=False)
    return df


def _load_or_compute_cldc() -> pd.DataFrame:
    if os.path.exists(CACHES['cldc']):
        print(f'  loading {CACHES["cldc"]}')
        return pd.read_csv(CACHES['cldc'])
    print('  computing CLDC summaries (no cache, may take a minute)')
    df = compute_all_cldc(data_dir=DATA_DIR)
    df.to_csv(CACHES['cldc'], index=False)
    return df


def _load_or_compute_profiles() -> pd.DataFrame:
    if os.path.exists(CACHES['profiles']):
        print(f'  loading {CACHES["profiles"]}')
        return pd.read_pickle(CACHES['profiles'])
    print('  computing daily profiles (no cache, may take several minutes)')
    df = compute_all_profiles(data_dir=DATA_DIR)
    df.to_pickle(CACHES['profiles'])
    return df


def _load_or_compute_metadata() -> pd.DataFrame:
    if os.path.exists(CACHES['metadata']):
        print(f'  loading {CACHES["metadata"]}')
        return pd.read_csv(CACHES['metadata'], index_col=0)
    print('  loading metadata (no cache)')
    meta = _load_metadata_table()
    meta.to_csv(CACHES['metadata'])
    return meta


def _load_or_compute_signatures() -> pd.DataFrame:
    if os.path.exists(CACHES['signatures']):
        print(f'  loading {CACHES["signatures"]}')
        return pd.read_pickle(CACHES['signatures'])
    print('  computing daily signatures (no cache, may take a minute)')
    df = compute_daily_signatures()
    df.to_pickle(CACHES['signatures'])
    return df


# =============================================================================
# PAYLOAD HELPERS
# =============================================================================

def _build_meta_lookup(meta: pd.DataFrame) -> dict:
    """Build bid_str → {label, function, era, gia_m2} lookup."""
    lookup = {}
    for bid_str, row in meta.iterrows():
        lookup[str(bid_str)] = {
            'label':    str(row.get('name_summary') or row.get('name_cebd') or bid_str),
            'function': row.get('function'),
            'era':      row.get('era'),
            'gia_m2':   float(row.get('gia_m2')) if pd.notna(row.get('gia_m2')) else None,
        }
    return lookup


def _metric_bin_edges(vals: pd.Series, metric_col: str) -> np.ndarray:
    """Fixed bin edges per metric — consistent across years for comparability."""
    clean = vals.dropna()
    if len(clean) == 0:
        return np.linspace(0, 1, N_BINS + 1)

    if metric_col.startswith('peak_hr_'):
        return np.arange(-0.5, 25.5, 1.0)

    if metric_col == 'fmr':
        pos = clean[clean > 0]
        if len(pos) == 0:
            return np.linspace(0, 1, N_BINS + 1)
        log_lo = float(np.log10(np.nanpercentile(pos, 1)))
        log_hi = float(np.log10(np.nanpercentile(pos, 99)))
        edges = np.concatenate([[0, 1e-3], np.logspace(log_lo, log_hi, N_BINS)])
        return np.unique(edges)

    lo, hi = float(np.nanpercentile(clean, 2)), float(np.nanpercentile(clean, 98))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, N_BINS + 1)


def _cldc_edges(vals: pd.Series) -> np.ndarray:
    clean = vals.dropna()
    if len(clean) == 0:
        return np.linspace(0, 1, N_BINS + 1)
    lo, hi = float(np.nanmin(clean)), float(np.nanmax(clean))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, N_BINS + 1)


def _bin_assignment(val: float, edges: np.ndarray) -> int:
    """Return the bin index (0-based) that val falls into, or -1 if out of range."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return -1
    for i in range(len(edges) - 1):
        if edges[i] <= val < edges[i + 1]:
            return i
    # Include the right edge in the final bin
    if abs(val - edges[-1]) < 1e-9:
        return len(edges) - 2
    return -1


# =============================================================================
# PAYLOAD BUILDERS
# =============================================================================

def _build_distributions_payload(temporal_df: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Histograms + per-bin building membership per (metric, year)."""
    years = sorted(temporal_df['year'].dropna().unique().astype(int))
    all_metrics = [m for group in METRIC_GROUPS.values() for m, _, _ in group]

    meta_lookup = _build_meta_lookup(meta)

    out = {'years': years, 'metrics': {}, 'groups': {}, 'meta': {}}

    for col in all_metrics:
        if col not in temporal_df.columns:
            continue
        edges = _metric_bin_edges(temporal_df[col], col)

        per_year = {}
        for yr in years:
            yr_df = temporal_df[temporal_df['year'] == yr]
            vals = yr_df[col].dropna().values
            counts, _ = np.histogram(vals, bins=edges)

            # Per-bin building membership
            bin_members = [[] for _ in range(len(edges) - 1)]
            for _, row in yr_df.iterrows():
                v = row.get(col)
                if pd.isna(v):
                    continue
                bi = _bin_assignment(float(v), edges)
                if bi < 0:
                    continue
                try:
                    bid_int = int(row['building_id'])
                except (ValueError, TypeError):
                    continue
                bid_str = f'b{bid_int}'
                m = meta_lookup.get(bid_str, {})
                bin_members[bi].append({
                    'id':       bid_str,
                    'label':    m.get('label') or bid_str,
                    'function': m.get('function') or 'n/a',
                    'era':      m.get('era') or 'n/a',
                    'value':    round(float(v), 3),
                })

            stats = {'n': int(len(vals))}
            if len(vals) > 0:
                stats.update({
                    'mean':   round(float(np.mean(vals)), 3),
                    'median': round(float(np.median(vals)), 3),
                })

            per_year[int(yr)] = {
                'counts':  [int(c) for c in counts],
                'members': bin_members,
                'stats':   stats,
            }

        out['metrics'][col] = {
            'edges':    [round(float(e), 4) for e in edges],
            'per_year': per_year,
        }

    for group, items in METRIC_GROUPS.items():
        for col, label, unit in items:
            out['meta'][col] = {'label': label, 'unit': unit, 'group': group}
    out['groups'] = {g: [c for c, _, _ in items] for g, items in METRIC_GROUPS.items()}

    # Per-building per-year value lookup (for rug marks overlaid on histograms)
    per_building_by_year = {}
    for _, row in temporal_df.iterrows():
        try:
            bid_int = int(row['building_id'])
        except (ValueError, TypeError):
            continue
        bid_str = f'b{bid_int}'
        yr = int(row['year']) if pd.notna(row['year']) else None
        if yr is None:
            continue
        per_building_by_year.setdefault(bid_str, {}).setdefault(yr, {})
        for col in all_metrics:
            if col in temporal_df.columns:
                v = row.get(col)
                if pd.notna(v):
                    per_building_by_year[bid_str][yr][col] = round(float(v), 3)

    out['per_building_by_year'] = per_building_by_year
    out['building_labels']      = {bid: info.get('label', bid) for bid, info in meta_lookup.items()}
    return out


def _build_cldc_payload(cldc_df: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Histograms + per-bin building membership per (metric, year, fuel)."""
    years = sorted(cldc_df['year'].dropna().unique().astype(int))
    fuels = ['elec', 'gas']

    meta_lookup = _build_meta_lookup(meta)

    out = {'years': years, 'fuels': fuels, 'metrics': {}}

    for col, label, unit in CLDC_METRICS:
        if col not in cldc_df.columns:
            continue
        edges = _cldc_edges(cldc_df[col])

        by_fuel = {}
        for fuel in fuels:
            by_fuel[fuel] = {}
            fuel_df = cldc_df[cldc_df['fuel'] == fuel]
            for yr in years:
                yr_df = fuel_df[fuel_df['year'] == yr]
                vals = yr_df[col].dropna().values
                counts, _ = np.histogram(vals, bins=edges)

                bin_members = [[] for _ in range(len(edges) - 1)]
                for _, row in yr_df.iterrows():
                    v = row.get(col)
                    if pd.isna(v):
                        continue
                    bi = _bin_assignment(float(v), edges)
                    if bi < 0:
                        continue
                    try:
                        bid_int = int(row['building_id'])
                    except (ValueError, TypeError):
                        continue
                    bid_str = f'b{bid_int}'
                    m = meta_lookup.get(bid_str, {})
                    bin_members[bi].append({
                        'id':       bid_str,
                        'label':    m.get('label') or bid_str,
                        'function': m.get('function') or 'n/a',
                        'era':      m.get('era') or 'n/a',
                        'value':    round(float(v), 4),
                    })

                by_fuel[fuel][int(yr)] = {
                    'counts':  [int(c) for c in counts],
                    'members': bin_members,
                    'n':       int(len(vals)),
                    'mean':    round(float(np.mean(vals)), 4)   if len(vals) > 0 else None,
                    'median':  round(float(np.median(vals)), 4) if len(vals) > 0 else None,
                }

        out['metrics'][col] = {
            'label':   label,
            'unit':    unit,
            'edges':   [round(float(e), 4) for e in edges],
            'by_fuel': by_fuel,
        }

    # Per-building per-(year, fuel) value lookup (for rug marks)
    per_building = {}
    for _, row in cldc_df.iterrows():
        try:
            bid_int = int(row['building_id'])
        except (ValueError, TypeError):
            continue
        bid_str = f'b{bid_int}'
        yr      = int(row['year'])
        fuel    = row['fuel']
        key     = f'{yr}_{fuel}'
        per_building.setdefault(bid_str, {}).setdefault(key, {})
        for col, _, _ in CLDC_METRICS:
            v = row.get(col)
            if pd.notna(v):
                per_building[bid_str][key][col] = round(float(v), 4)

    out['per_building'] = per_building
    return out


def _build_profiles_payload(profiles_df: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Daily profiles grouped by (fuel, year, season, daytype)."""
    meta_lookup = _build_meta_lookup(meta)

    seasons_keep  = {'all', 'summer', 'winter'}
    daytypes_keep = {'all', 'weekday', 'weekend'}

    sub = profiles_df[
        profiles_df['season'].isin(seasons_keep) &
        profiles_df['daytype'].isin(daytypes_keep)
    ]

    fuels    = sorted(sub['fuel'].unique())
    years    = sorted(sub['year'].unique().astype(int))
    seasons  = sorted(sub['season'].unique())
    daytypes = sorted(sub['daytype'].unique())

    out = {
        'fuels':     fuels,
        'years':     years,
        'seasons':   seasons,
        'daytypes':  daytypes,
        'data':      {},
        'functions': sorted(set(
            v['function'] for v in meta_lookup.values() if v.get('function')
        )),
    }

    for fuel in fuels:
        out['data'][fuel] = {}
        for yr in years:
            out['data'][fuel][int(yr)] = {}
            for season in seasons:
                out['data'][fuel][int(yr)][season] = {}
                for daytype in daytypes:
                    rows = sub[
                        (sub['fuel']    == fuel) &
                        (sub['year']    == yr) &
                        (sub['season']  == season) &
                        (sub['daytype'] == daytype)
                    ]
                    entries = []
                    for _, r in rows.iterrows():
                        prof = r.get('profile_sum1')
                        if prof is None or (isinstance(prof, float) and np.isnan(prof)):
                            continue
                        try:
                            bid_int = int(r['building_id'])
                        except (ValueError, TypeError):
                            continue
                        bid_str = f'b{bid_int}'
                        m = meta_lookup.get(bid_str, {})
                        entries.append({
                            'id':       bid_str,
                            'label':    m.get('label') or bid_str,
                            'function': m.get('function'),
                            'era':      m.get('era'),
                            'values':   [round(float(v), 5) for v in prof],
                        })
                    out['data'][fuel][int(yr)][season][daytype] = entries

    return out


def _build_signatures_payload(
    sigs_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    meta: pd.DataFrame,
) -> dict:
    """Per (building, year): scatter points + fitted HS line parameters."""
    years = sorted(sigs_df['year'].dropna().unique().astype(int))
    meta_lookup = _build_meta_lookup(meta)

    hs_cols = ['hs_slope', 'hs_balance', 'hs_q0', 'hs_r2']
    hs_available = all(c in temporal_df.columns for c in hs_cols)

    by_year = {}
    for yr in years:
        sub = sigs_df[sigs_df['year'] == yr]

        points = {}
        for bid, grp in sub.groupby('building_id'):
            try:
                bid_int = int(bid)
            except (ValueError, TypeError):
                continue
            bid_str = f'b{bid_int}'
            m = meta_lookup.get(bid_str, {})
            points[bid_str] = {
                'label':    m.get('label') or bid_str,
                'function': m.get('function') or 'n/a',
                't':        [round(float(x), 2) for x in grp['t_ext'].values],
                'q':        [round(float(x), 2) for x in grp['q_gas'].values],
            }

        fits = {}
        if hs_available:
            yr_fits = temporal_df[temporal_df['year'] == yr]
            for _, row in yr_fits.iterrows():
                try:
                    bid_int = int(row['building_id'])
                except (ValueError, TypeError):
                    continue
                bid_str = f'b{bid_int}'
                if bid_str not in points:
                    continue
                if pd.isna(row.get('hs_slope')):
                    continue
                fits[bid_str] = {
                    't_bal': round(float(row['hs_balance']), 2) if pd.notna(row.get('hs_balance')) else None,
                    'slope': round(float(row['hs_slope']),   2) if pd.notna(row.get('hs_slope'))   else None,
                    'q0':    round(float(row.get('hs_q0', 0) or 0), 2),
                    'r2':    round(float(row['hs_r2']),      3) if pd.notna(row.get('hs_r2'))      else None,
                }

        by_year[int(yr)] = {'points': points, 'fits': fits}

    functions = sorted(set(
        m.get('function') for m in meta_lookup.values() if m.get('function')
    ))

    t_all = sigs_df['t_ext'].values
    q_all = sigs_df['q_gas'].values

    return {
        'years':     years,
        'by_year':   by_year,
        'functions': functions,
        'scales': {
            't_min': float(np.nanpercentile(t_all, 0.5))  if len(t_all) > 0 else -5,
            't_max': float(np.nanpercentile(t_all, 99.5)) if len(t_all) > 0 else 25,
            'q_min': 0.0,
            'q_max': float(np.nanpercentile(q_all, 99))   if len(q_all) > 0 else 1000,
        },
    }


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {_to_jsonable(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None or isinstance(obj, (list, tuple, dict, str, bool)):
        return obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj

# =============================================================================
# HTML GENERATION
# =============================================================================

def _render_html(
    dist_payload: dict,
    cldc_payload: dict,
    prof_payload: dict,
    sig_payload:  dict,
    output_path: str,
) -> None:

    payload_json = json.dumps(_to_jsonable({
        'dist': dist_payload,
        'cldc': cldc_payload,
        'prof': prof_payload,
        'sig':  sig_payload,
    }), separators=(',', ':'))

    plotly_version = get_plotlyjs_version()

    # ------------------------------------------------------------------------
    # CSS — plain string (single braces, normal CSS syntax)
    # ------------------------------------------------------------------------
    css_block = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { width: 100%; height: 100%; font-family: Arial, sans-serif; background: #fafafa; color: #222; }
  #page { display: flex; flex-direction: column; height: 100%; }

  #header { padding: 14px 22px; background: white; border-bottom: 1px solid #ddd; }
  #header h1 { font-size: 18px; font-weight: 600; color: #222; margin-bottom: 2px; }

  #controls {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    padding: 10px 22px;
    background: white;
    border-bottom: 1px solid #ddd;
    position: relative;
  }
  .control-group { display: flex; align-items: center; gap: 8px; }
  .control-group label { font-size: 12px; font-weight: bold; color: #555; }
  select, input[type="range"], input[type="text"] {
    padding: 4px 6px;
    font-size: 12px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background: white;
  }
  input[type="range"] { padding: 0; width: 160px; }
  button {
    padding: 5px 12px;
    font-size: 12px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    background: #3388ff;
    color: white;
  }
  button:hover { background: #2266cc; }
  .year-label {
    display: inline-block;
    min-width: 42px;
    text-align: center;
    font-weight: bold;
    font-size: 14px;
    color: #333;
  }
  .btn-light {
    padding: 5px 10px;
    font-size: 12px;
    background: #f7f7f7;
    border: 1px solid #ccc;
    border-radius: 4px;
    cursor: pointer;
    color: #444;
  }
  .btn-light:hover { background: #eee; }
  .btn-light.has-selection {
    background: #fff0f0;
    border-color: #e31a1c;
    color: #a01820;
    font-weight: bold;
  }

  .panel {
    position: absolute;
    top: 52px;
    right: 22px;
    background: white;
    border: 1px solid #ccc;
    border-radius: 6px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.18);
    padding: 14px 16px;
    z-index: 100;
    min-width: 300px;
    display: none;
    font-size: 12px;
  }
  .panel.open { display: block; }
  .panel label { display: block; font-weight: bold; margin-bottom: 4px; color: #555; }
  .panel textarea {
    width: 100%; padding: 6px; font-size: 12px;
    border: 1px solid #ccc; border-radius: 4px;
    font-family: monospace; resize: vertical; min-height: 60px;
  }
  .panel .hint { font-size: 11px; color: #888; margin: 4px 0 8px; }
  .panel .counter { margin: 6px 0; font-size: 12px; color: #333; }
  .panel .counter b { color: #e31a1c; }
  .panel .close-x {
    position: absolute; top: 8px; right: 10px;
    cursor: pointer; font-size: 16px; color: #888;
    background: none; border: none; padding: 0; font-weight: bold;
  }

  #side-panel {
    position: fixed;
    top: 0; right: 0; bottom: 0;
    width: 380px;
    background: white;
    border-left: 1px solid #ccc;
    box-shadow: -2px 0 12px rgba(0,0,0,0.1);
    padding: 20px 18px;
    overflow-y: auto;
    z-index: 500;
    display: none;
  }
  #side-panel.open { display: block; }
  #side-panel h2 { font-size: 15px; color: #222; margin-bottom: 4px; }
  #side-panel h3 { font-size: 13px; color: #444; margin: 10px 0 4px; }
  #side-panel .meta { font-size: 12px; color: #666; margin-bottom: 10px; }
  #side-panel .sp-close {
    position: absolute; top: 8px; right: 12px;
    cursor: pointer; font-size: 22px; color: #888;
    background: none; border: none; padding: 0;
  }
  #side-panel table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }
  #side-panel th, #side-panel td { padding: 4px 6px; text-align: left; border-bottom: 1px solid #eee; }
  #side-panel th { background: #f5f5f5; font-weight: bold; color: #444; cursor: pointer; user-select: none; }
  #side-panel th:hover { background: #e8e8e8; }
  #side-panel .id-link { color: #3388ff; cursor: pointer; }
  #side-panel .id-link:hover { text-decoration: underline; }
  #side-panel .btn-row { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }

  #description {
    margin: 12px 22px 4px;
    padding: 10px 14px;
    background: white;
    border-left: 4px solid #3388ff;
    border-radius: 3px;
    font-size: 13px;
    color: #444;
    line-height: 1.5;
  }

  #plot-area {
    flex: 1 1 auto;
    padding: 12px 22px;
    overflow: auto;
    min-height: 500px;
  }
  #plot { width: 100%; height: 100%; min-height: 500px; }

  #footer {
    padding: 6px 22px;
    background: white;
    border-top: 1px solid #ddd;
    font-size: 11px;
    color: #888;
  }

  .hidden { display: none !important; }
  .toggle-btn {
    padding: 3px 10px;
    font-size: 11px;
    background: #f0f0f0;
    border: 1px solid #ccc;
    border-radius: 3px;
    cursor: pointer;
    color: #555;
  }
  .toggle-btn.active { background: #3388ff; color: white; border-color: #2266cc; }
"""

    # ------------------------------------------------------------------------
    # HTML body — plain string, no braces to escape
    # ------------------------------------------------------------------------
    html_body = """
<div id="page">

  <div id="header">
    <h1>Cambridge Estates – Non-Spatial Metrics</h1>
  </div>

  <div id="controls">

    <div class="control-group">
      <label>View</label>
      <select id="view-select">
        <option value="dist">Distributions</option>
        <option value="cldc">CLDC Scalars</option>
        <option value="prof">Daily Load Profiles</option>
        <option value="sig">Energy Signatures</option>
      </select>
    </div>

    <div class="control-group view-dist">
      <label>Group</label>
      <select id="dist-group-select"></select>
    </div>

    <div class="control-group view-cldc hidden">
      <label>Metric</label>
      <select id="cldc-metric-select"></select>
    </div>
    <div class="control-group view-cldc hidden">
      <label>Fuel</label>
      <select id="cldc-fuel-select">
        <option value="elec">Electricity</option>
        <option value="gas">Gas</option>
      </select>
    </div>

    <div class="control-group view-prof hidden">
      <label>Fuel</label>
      <select id="prof-fuel-select"></select>
    </div>
    <div class="control-group view-prof hidden">
      <label>Season</label>
      <select id="prof-season-select"></select>
    </div>
    <div class="control-group view-prof hidden">
      <label>Day</label>
      <select id="prof-daytype-select"></select>
    </div>
    <div class="control-group view-prof hidden">
      <label>Highlight by fn</label>
      <select id="prof-function-select">
        <option value="">— none —</option>
      </select>
    </div>

    <div class="control-group view-sig hidden">
      <label>Show estate cloud</label>
      <button id="sig-context-toggle" class="toggle-btn">off</button>
    </div>

    <div class="control-group" id="year-controls">
      <button id="play-btn">▶ Play</button>
      <label>Year</label>
      <input id="year-slider" type="range" min="0" max="0" value="0">
      <span id="year-display" class="year-label"></span>
    </div>

    <div class="control-group">
      <button id="filter-btn" class="btn-light">☰ Filter <span id="filter-btn-count"></span></button>
    </div>
    <div class="control-group view-dist view-cldc">
      <button id="topn-btn" class="btn-light">⇅ Top-N</button>
    </div>
    <div class="control-group">
      <button id="export-btn" class="btn-light">⬇ Export</button>
    </div>

    <!-- Filter panel -->
    <div id="filter-panel" class="panel">
      <button class="close-x" data-close="filter-panel">×</button>
      <label for="filter-textarea">Filter / highlight buildings</label>
      <textarea id="filter-textarea" placeholder="e.g. b7, b42, b150"></textarea>
      <div class="hint">Comma- or space-separated IDs. Applies to all views. Signatures filters; others overlay.</div>
      <div id="filter-counter" class="counter">No buildings selected.</div>
      <button id="filter-clear-btn" class="btn-light">Clear selection</button>
    </div>

    <!-- Top-N panel -->
    <div id="topn-panel" class="panel" style="min-width: 340px;">
      <button class="close-x" data-close="topn-panel">×</button>
      <label>Top-N buildings</label>
      <div class="hint">Ranked by the currently displayed metric for this year.</div>
      <div style="display:flex; gap:8px; align-items:center; margin-bottom:8px;">
        <button id="topn-dir-btn" class="toggle-btn active">↑ Highest</button>
        <span id="topn-metric-label" style="font-size:11px; color:#888;"></span>
      </div>
      <div id="topn-table-wrap"></div>
      <div class="btn-row">
        <button id="topn-filter-btn" class="btn-light">Add these to filter</button>
      </div>
    </div>

    <!-- Export panel -->
    <div id="export-panel" class="panel">
      <button class="close-x" data-close="export-panel">×</button>
      <label>Export current view</label>
      <div class="hint">SVG is vector (editable in Inkscape/Illustrator). PNG is 2× screen resolution.</div>
      <div class="btn-row">
        <button id="export-png-btn" class="btn-light">⬇ PNG (2×)</button>
        <button id="export-svg-btn" class="btn-light">⬇ SVG (vector)</button>
      </div>
    </div>

  </div>

  <div id="description"></div>

  <div id="plot-area">
    <div id="plot"></div>
  </div>

  <div id="footer">
    Click a histogram bar to see which buildings fall in that bin. Use Filter to highlight specific buildings across all views. Use Top-N to rank by the currently displayed metric.
  </div>
</div>

<!-- Side panel (absolute, sliding from right) -->
<div id="side-panel">
  <button class="sp-close" id="side-panel-close">×</button>
  <div id="side-panel-content"></div>
</div>
"""

    # ------------------------------------------------------------------------
    # JavaScript — plain string, single braces throughout.
    # Payload is injected via placeholder replacement.
    # ------------------------------------------------------------------------
    js_block = """
// ══════════════════════════════════════════════════════════════════════════════
// Cambridge Estates non-spatial dashboard
// ══════════════════════════════════════════════════════════════════════════════

var PAYLOAD = __PAYLOAD_JSON__;
var DIST = PAYLOAD.dist;
var CLDC = PAYLOAD.cldc;
var PROF = PAYLOAD.prof;
var SIG  = PAYLOAD.sig;

// ── state ─────────────────────────────────────────────────────────────────────
var state = {
  view: 'dist',
  year: DIST.years[DIST.years.length - 1],
  dist_group:          Object.keys(DIST.groups)[0],
  cldc_metric:         Object.keys(CLDC.metrics)[0],
  cldc_fuel:           'elec',
  prof_fuel:           PROF.fuels[0]    || 'elec',
  prof_season:         PROF.seasons.includes('all')  ? 'all' : PROF.seasons[0],
  prof_daytype:        PROF.daytypes.includes('all') ? 'all' : PROF.daytypes[0],
  prof_highlight_fn:   '',
  sig_show_cloud:      false,
  filter_bids:         [],
  filter_raw:          '',
  topn_direction:      'high',
};
var playTimer = null;

// ── view descriptions ─────────────────────────────────────────────────────────
var DESCRIPTIONS = {
  dist: 'Each subplot is a histogram showing how buildings are distributed on one metric for the selected year. The x-axis is the metric value, the y-axis is the count of buildings. The red vertical line marks the estate-wide median. Click any bar to see which buildings fall in that bin.',
  cldc: 'Histogram of one cumulative-load-duration-curve scalar across all buildings for the selected year and fuel. Each bar counts buildings whose metric value falls in that range. Click any bar to see which buildings are in that bin.',
  prof: 'Each line is the average 24-hour load profile of a building, normalised to sum to 1. Buildings with similar daily operation patterns overlap; distinct patterns separate out. Use "Highlight by fn" to colour one function type in red.',
  sig:  'Each dot is one day of one building. The x-axis is daily mean outdoor temperature; the y-axis is daily gas consumption. By default the plot is filtered — only selected buildings are visible. Each selected building also shows its fitted piecewise-linear heating model (flat above balance point, sloped below). Labels show building IDs directly on the plot.',
};

// ── helpers ───────────────────────────────────────────────────────────────────
function populateSelect(id, values, labels) {
  var el = document.getElementById(id);
  el.innerHTML = '';
  values.forEach(function(v, i) {
    var opt = document.createElement('option');
    opt.value = v;
    opt.textContent = labels ? labels[i] : v;
    el.appendChild(opt);
  });
}

function parseBuildingList(raw) {
  if (!raw) return [];
  return raw.split(/[,\\s]+/)
    .map(function(s) { return s.trim(); })
    .filter(function(s) { return s.length > 0; })
    .map(function(s) {
      var m = s.toLowerCase().replace(/^b/, '');
      return 'b' + m;
    });
}

function allKnownBids() {
  var set = {};
  Object.keys(DIST.per_building_by_year || {}).forEach(function(b) { set[b] = true; });
  Object.keys(CLDC.per_building || {}).forEach(function(b) { set[b] = true; });
  if (SIG && SIG.by_year) {
    Object.keys(SIG.by_year).forEach(function(y) {
      Object.keys(SIG.by_year[y].points || {}).forEach(function(bid) { set[bid] = true; });
    });
  }
  return set;
}
var KNOWN_BIDS = allKnownBids();

function updateFilterState(rawText) {
  state.filter_raw = rawText;
  var requested = parseBuildingList(rawText);
  state.filter_bids = requested.filter(function(bid) { return KNOWN_BIDS[bid]; });

  var matched = state.filter_bids.length;
  var typed   = requested.length;
  var counter = document.getElementById('filter-counter');
  var btnCnt  = document.getElementById('filter-btn-count');
  var btn     = document.getElementById('filter-btn');

  if (typed === 0) {
    counter.textContent = 'No buildings selected.';
    btnCnt.textContent = '';
    btn.classList.remove('has-selection');
  } else {
    counter.innerHTML = '<b>' + matched + '</b> of ' + typed + ' matched.';
    btnCnt.textContent = '(' + matched + ')';
    btn.classList.add('has-selection');
  }
  render();
}

// ── year slider ───────────────────────────────────────────────────────────────
function yearsForView() {
  if (state.view === 'dist') return DIST.years;
  if (state.view === 'cldc') return CLDC.years;
  if (state.view === 'prof') return PROF.years;
  if (state.view === 'sig')  return SIG.years;
  return [];
}
function refreshYearSlider() {
  var ys = yearsForView();
  var slider = document.getElementById('year-slider');
  slider.min = 0;
  slider.max = Math.max(0, ys.length - 1);
  var idx = ys.indexOf(state.year);
  if (idx === -1) idx = ys.length - 1;
  slider.value = idx;
  state.year = ys[idx];
  document.getElementById('year-display').textContent = state.year;
}

// ── view visibility ───────────────────────────────────────────────────────────
function applyViewVisibility() {
  ['dist', 'cldc', 'prof', 'sig'].forEach(function(v) {
    document.querySelectorAll('.view-' + v).forEach(function(el) {
      el.classList.toggle('hidden', state.view !== v);
    });
  });
  document.getElementById('description').textContent = DESCRIPTIONS[state.view] || '';

  var topnBtn = document.getElementById('topn-btn');
  topnBtn.style.display = (state.view === 'dist' || state.view === 'cldc') ? '' : 'none';
}

// ── rug trace helper ──────────────────────────────────────────────────────────
function rugTrace(xValues, texts, xaxis, yaxis) {
  return {
    type: 'scatter', mode: 'markers+text',
    x: xValues, y: xValues.map(function() { return 0; }),
    xaxis: xaxis, yaxis: yaxis,
    marker: {color: '#e31a1c', symbol: 'line-ns', size: 20,
             line: {color: '#e31a1c', width: 3}},
    text: xValues.map(function(_, i) { return i < 5 ? (texts[i] || '').split(' · ')[0] : ''; }),
    textposition: 'top center',
    textfont: {size: 15, color: '#e31a1c'},
    hovertemplate: '%{customdata}<extra></extra>',
    customdata: texts,
    showlegend: false,
  };
}

// ── FIGURE: DISTRIBUTIONS ─────────────────────────────────────────────────────
function figureDist() {
  var group   = state.dist_group;
  var metrics = DIST.groups[group];
  var nCols   = Math.min(3, metrics.length);
  var nRows   = Math.ceil(metrics.length / nCols);

  var fig = {
    data: [],
    layout: {
      grid: {rows: nRows, columns: nCols, pattern: 'independent'},
      margin: {l: 55, r: 15, t: 60, b: 55},
      paper_bgcolor: '#fafafa', plot_bgcolor: 'white',
      font: {family: 'Arial, sans-serif', size: 11, color: '#333'},
      showlegend: false,
      annotations: [],
      autosize: true,
    },
    config: {displaylogo: false, responsive: true,
             modeBarButtonsToRemove: ['lasso2d', 'select2d']}
  };

  metrics.forEach(function(m, idx) {
    var md = DIST.metrics[m];
    if (!md) return;
    var meta = DIST.meta[m];
    var suffix = (idx === 0) ? '' : String(idx + 1);
    var bucket = md.per_year[state.year] || {counts: [], members: [], stats: {n: 0}};

    var centres = [], widths = [];
    for (var i = 0; i < md.edges.length - 1; i++) {
      centres.push((md.edges[i] + md.edges[i+1]) / 2);
      widths.push(md.edges[i+1] - md.edges[i]);
    }

    var hoverTexts = bucket.counts.map(function(cnt, bi) {
      var mbrs = (bucket.members || [])[bi] || [];
      var lo = md.edges[bi].toFixed(2);
      var hi = md.edges[bi+1].toFixed(2);
      var head = '<b>' + lo + ' – ' + hi + '</b><br>' + cnt + ' building' + (cnt === 1 ? '' : 's');
      if (mbrs.length === 0) return head;
      var preview = mbrs.slice(0, 5).map(function(mb) {
        return mb.id + ' (' + mb.label + ', ' + mb.value + ')';
      }).join('<br>');
      var extra = mbrs.length > 5 ? '<br><i>+ ' + (mbrs.length - 5) + ' more (click bar)</i>' : '';
      return head + '<br>' + preview + extra;
    });

    fig.data.push({
      type: 'bar', x: centres, y: bucket.counts, width: widths,
      xaxis: 'x' + suffix, yaxis: 'y' + suffix,
      marker: {color: '#3388ff', line: {color: '#1a4d99', width: 1}},
      hovertemplate: '%{customdata}<extra></extra>',
      customdata: hoverTexts,
      name: m,
    });

    if (state.filter_bids.length > 0) {
      var rugX = [], rugText = [];
      state.filter_bids.forEach(function(bid) {
        var yrMetrics = ((DIST.per_building_by_year || {})[bid] || {})[state.year] || {};
        var v = yrMetrics[m];
        if (v === null || v === undefined) return;
        rugX.push(v);
        var lbl = (DIST.building_labels || {})[bid] || bid;
        rugText.push(bid + ' · ' + lbl + ' · ' + v.toFixed(2));
      });
      if (rugX.length > 0) {
        fig.data.push(rugTrace(rugX, rugText, 'x' + suffix, 'y' + suffix));
      }
    }

    var xTitle = meta.label + ' (' + meta.unit + ')';
    var xCfg = {
      title: {text: xTitle, font: {size: 15, color: '#444'}, standoff: 4},
      showgrid: false, zeroline: false, tickfont: {size: 13},
    };
    if (m === 'fmr') xCfg.type = 'log';
    if (m.indexOf('peak_hr') === 0) {
      xCfg.tickmode = 'array';
      xCfg.tickvals = [0, 6, 12, 18, 24];
      xCfg.ticktext = ['00','06','12','18','24'];
    }
    fig.layout['xaxis' + suffix] = xCfg;
    fig.layout['yaxis' + suffix] = {
      title: {text: 'Number of buildings', font: {size: 15, color: '#444'}, standoff: 4},
      showgrid: true, gridcolor: '#eee', zeroline: false, tickfont: {size: 13}
    };

    if (bucket.stats.n > 0) {
      var yMax = Math.max.apply(null, bucket.counts);
      fig.data.push({
        type: 'scatter', mode: 'lines',
        x: [bucket.stats.median, bucket.stats.median], y: [0, yMax],
        xaxis: 'x' + suffix, yaxis: 'y' + suffix,
        line: {color: '#e31a1c', width: 2, dash: 'dot'},
        hoverinfo: 'skip', showlegend: false,
      });
      fig.layout.annotations.push({
        text: 'n=' + bucket.stats.n + '<br>med=' + bucket.stats.median.toFixed(2),
        xref: 'x' + suffix + ' domain', yref: 'y' + suffix + ' domain',
        x: 0.98, y: 0.95, xanchor: 'right', yanchor: 'top',
        showarrow: false, font: {size: 9, color: '#555'},
        bgcolor: 'rgba(255,255,255,0.85)', borderpad: 3
      });
    }
  });

  return fig;
}

// ── FIGURE: CLDC ──────────────────────────────────────────────────────────────
function figureCldc() {
  var metricKey = state.cldc_metric;
  var fuel      = state.cldc_fuel;
  var md        = CLDC.metrics[metricKey];
  if (!md) return {data: [], layout: {}};

  var bucket = (md.by_fuel[fuel] || {})[state.year] || {counts: [], members: [], n: 0};

  var centres = [], widths = [];
  for (var i = 0; i < md.edges.length - 1; i++) {
    centres.push((md.edges[i] + md.edges[i+1]) / 2);
    widths.push(md.edges[i+1] - md.edges[i]);
  }

  var hoverTexts = bucket.counts.map(function(cnt, bi) {
    var mbrs = (bucket.members || [])[bi] || [];
    var lo = md.edges[bi].toFixed(3);
    var hi = md.edges[bi+1].toFixed(3);
    var head = '<b>' + lo + ' – ' + hi + '</b><br>' + cnt + ' building' + (cnt === 1 ? '' : 's');
    if (mbrs.length === 0) return head;
    var preview = mbrs.slice(0, 5).map(function(mb) {
      return mb.id + ' (' + mb.label + ', ' + mb.value + ')';
    }).join('<br>');
    var extra = mbrs.length > 5 ? '<br><i>+ ' + (mbrs.length - 5) + ' more (click bar)</i>' : '';
    return head + '<br>' + preview + extra;
  });

  var fig = {
    data: [{
      type: 'bar', x: centres, y: bucket.counts, width: widths,
      marker: {color: '#3388ff', line: {color: '#1a4d99', width: 1}},
      hovertemplate: '%{customdata}<extra></extra>',
      customdata: hoverTexts,
    }],
    layout: {
      title: {
        text: md.label + ' — ' + fuel + ', ' + state.year,
        font: {size: 14}, x: 0.02, xanchor: 'left'
      },
      margin: {l: 60, r: 20, t: 50, b: 60},
      paper_bgcolor: '#fafafa', plot_bgcolor: 'white',
      xaxis: {
        title: {text: md.label + ' (' + md.unit + ')', font: {size: 15, color: '#444'}},
        showgrid: false, tickfont: {size: 13}
      },
      yaxis: {
        title: {text: 'Number of buildings', font: {size: 15, color: '#444'}},
        gridcolor: '#eee', tickfont: {size: 13}
      },
      annotations: [],
      showlegend: false,
      autosize: true,
    },
    config: {displaylogo: false, responsive: true,
             modeBarButtonsToRemove: ['lasso2d', 'select2d']}
  };

  if (state.filter_bids.length > 0) {
    var key = state.year + '_' + fuel;
    var rugX = [], rugText = [];
    state.filter_bids.forEach(function(bid) {
      var pb = ((CLDC.per_building || {})[bid] || {})[key];
      if (!pb) return;
      var v = pb[metricKey];
      if (v === null || v === undefined) return;
      rugX.push(v);
      var lbl = (DIST.building_labels || {})[bid] || bid;
      rugText.push(bid + ' · ' + lbl + ' · ' + v.toFixed(3));
    });
    if (rugX.length > 0) fig.data.push(rugTrace(rugX, rugText));
  }

  if (bucket.n > 0) {
    var yMax = Math.max.apply(null, bucket.counts);
    fig.data.push({
      type: 'scatter', mode: 'lines',
      x: [bucket.median, bucket.median], y: [0, yMax],
      line: {color: '#e31a1c', width: 2, dash: 'dot'},
      hoverinfo: 'skip', showlegend: false,
    });
    fig.layout.annotations.push({
      text: 'n=' + bucket.n +
            '<br>median=' + bucket.median.toFixed(3) +
            '<br>mean=' + bucket.mean.toFixed(3),
      xref: 'paper', yref: 'paper',
      x: 0.98, y: 0.97, xanchor: 'right', yanchor: 'top',
      showarrow: false, font: {size: 15, color: '#555'},
      bgcolor: 'rgba(255,255,255,0.9)', borderpad: 4
    });
  }

  return fig;
}

// ── FIGURE: PROFILES ──────────────────────────────────────────────────────────
function figureProf() {
  var entries = (((PROF.data[state.prof_fuel] || {})[state.year] || {})[state.prof_season] || {})[state.prof_daytype] || [];
  var hlFn = state.prof_highlight_fn;
  var filterBids = state.filter_bids;
  var hasFilter = filterBids.length > 0;

  var hours = [];
  for (var h = 0; h < 24; h++) hours.push(h);

  var palette = ['#e31a1c', '#1f78b4', '#33a02c', '#ff7f00', '#6a3d9a',
                 '#b15928', '#8c564b', '#e377c2', '#17becf', '#bcbd22'];

  var traces = [];
  var bgX = [], bgY = [];
  var highlighted = [];
  entries.forEach(function(e) {
    var inFilter = hasFilter && filterBids.indexOf(e.id) >= 0;
    var inFunction = (hlFn !== '') && (e.function === hlFn);

    if (inFilter || inFunction) {
      highlighted.push(e);
    } else {
      for (var i = 0; i < 24; i++) {
        bgX.push(i);
        bgY.push(e.values[i]);
      }
      bgX.push(null); bgY.push(null);
    }
  });

  if (bgX.length > 0) {
    traces.push({
      type: 'scatter', mode: 'lines',
      x: bgX, y: bgY,
      name: 'all buildings (' + (entries.length - highlighted.length) + ')',
      line: {color: 'rgba(60, 120, 200, 0.14)', width: 1},
      hoverinfo: 'skip',
      showlegend: true,
    });
  }

  highlighted.forEach(function(e, idx) {
    var inFilter = hasFilter && filterBids.indexOf(e.id) >= 0;
    var colour = inFilter ? palette[idx % palette.length] : '#e31a1c';
    traces.push({
      type: 'scatter', mode: 'lines',
      x: hours, y: e.values,
      name: e.id + ' · ' + e.label,
      line: {color: colour, width: 2.4},
      hovertemplate: '<b>' + e.label + '</b> (' + (e.function || 'n/a') + ')<br>' +
                     '%{x}:00 → %{y:.4f}<extra></extra>',
      showlegend: true,
    });
  });

  var subtitle = state.prof_fuel + ' · ' + state.prof_season + ' · ' + state.prof_daytype +
                 ' · ' + state.year + ' · ' + entries.length + ' buildings' +
                 (hlFn ? ' · highlight: ' + hlFn : '') +
                 (hasFilter ? ' · filter: ' + filterBids.length : '');

  return {
    data: traces,
    layout: {
      title: {text: 'Daily load profiles — ' + subtitle, font: {size: 14},
              x: 0.02, xanchor: 'left'},
      margin: {l: 65, r: 250, t: 50, b: 60},
      paper_bgcolor: '#fafafa', plot_bgcolor: 'white',
      xaxis: {
        title: {text: 'Hour of day', font: {size: 15, color: '#444'}},
        tickvals: [0, 6, 12, 18, 24],
        ticktext: ['00', '06', '12', '18', '24'],
        gridcolor: '#eee',
      },
      yaxis: {
        title: {text: 'Normalised load (sum-to-1 over 24h)', font: {size: 15, color: '#444'}},
        gridcolor: '#eee'
      },
      showlegend: true,
      legend: {
        x: 1.02, y: 1, xanchor: 'left',
        font: {size: 15},
        bgcolor: 'rgba(255,255,255,0.85)',
        bordercolor: '#ddd', borderwidth: 1,
      },
      autosize: true,
    },
    config: {displaylogo: false, responsive: true,
             modeBarButtonsToRemove: ['lasso2d', 'select2d']}
  };
}

// ── FIGURE: SIGNATURES ────────────────────────────────────────────────────────
function figureSig() {
  var yr = state.year;
  var yrData = SIG.by_year[yr] || {points: {}, fits: {}};
  var allBids = Object.keys(yrData.points);
  var filter = state.filter_bids;
  var hasFilter = filter.length > 0;
  var showCloud = state.sig_show_cloud || !hasFilter;

  var traces = [];

  if (showCloud) {
    var bgT = [], bgQ = [];
    allBids.forEach(function(bid) {
      if (hasFilter && filter.indexOf(bid) >= 0) return;
      var d = yrData.points[bid];
      bgT = bgT.concat(d.t);
      bgQ = bgQ.concat(d.q);
    });
    if (bgT.length > 0) {
      traces.push({
        type: 'scattergl', mode: 'markers',
        x: bgT, y: bgQ,
        marker: {color: 'rgba(120, 120, 120, 0.15)', size: 3},
        hoverinfo: 'skip',
        name: 'estate (' + (allBids.length - filter.length) + ')',
        showlegend: true,
      });
    }
  }

  var palette = ['#e31a1c', '#1f78b4', '#33a02c', '#ff7f00', '#6a3d9a',
                 '#b15928', '#8c564b', '#e377c2', '#17becf', '#bcbd22'];
  var shownBids = hasFilter ? filter.filter(function(b) { return allBids.indexOf(b) >= 0; }) : [];

  shownBids.forEach(function(bid, idx) {
    var d = yrData.points[bid];
    if (!d) return;
    var colour = palette[idx % palette.length];

    traces.push({
      type: 'scatter', mode: 'markers',
      x: d.t, y: d.q,
      marker: {color: colour, size: 6, opacity: 0.75},
      name: bid + ' (' + d.function + ')',
      hovertemplate: '<b>' + bid + '</b> · ' + d.label +
                     '<br>T=%{x:.1f}°C · Q=%{y:.1f} kWh/day<extra></extra>',
      showlegend: true,
    });

    var lastIdx = d.t.length - 1;
    if (lastIdx >= 0) {
      var maxT = -Infinity, maxI = 0;
      for (var k = 0; k < d.t.length; k++) {
        if (d.t[k] > maxT) { maxT = d.t[k]; maxI = k; }
      }
      traces.push({
        type: 'scatter', mode: 'text',
        x: [d.t[maxI]], y: [d.q[maxI]],
        text: [bid],
        textposition: 'middle right',
        textfont: {color: colour, size: 12, family: 'Arial, sans-serif'},
        hoverinfo: 'skip',
        showlegend: false,
      });
    }

    var f = yrData.fits[bid];
    if (f && f.t_bal !== null && f.slope !== null) {
      var tMin = SIG.scales.t_min, tMax = SIG.scales.t_max;
      var lineT = [tMin, f.t_bal, tMax];
      var lineQ = lineT.map(function(t) {
        return f.q0 + f.slope * Math.max(0, f.t_bal - t);
      });
      traces.push({
        type: 'scatter', mode: 'lines',
        x: lineT, y: lineQ,
        line: {color: colour, width: 2.5},
        name: bid + ' fit (R²=' + (f.r2 !== null ? f.r2.toFixed(2) : 'n/a') + ')',
        hovertemplate: '<b>' + bid + ' fit</b><br>T_bal=' + f.t_bal.toFixed(1) +
                       '°C, slope=' + f.slope.toFixed(1) + ' kWh/°C·day<extra></extra>',
        showlegend: true,
      });
    }
  });

  var subtitle;
  if (hasFilter) {
    subtitle = yr + ' · showing ' + shownBids.length + ' selected building' +
               (shownBids.length === 1 ? '' : 's') +
               (showCloud ? ' (with estate cloud)' : ' (filter mode)');
  } else {
    subtitle = yr + ' · ' + allBids.length + ' buildings with gas data · no filter active';
  }

  return {
    data: traces,
    layout: {
      title: {text: 'Energy signatures — ' + subtitle, font: {size: 15},
              x: 0.02, xanchor: 'left'},
      margin: {l: 70, r: 250, t: 50, b: 60},
      paper_bgcolor: '#fafafa', plot_bgcolor: 'white',
      xaxis: {
        title: {text: 'Daily mean outdoor temperature (°C)', font: {size: 15, color: '#444'}},
        gridcolor: '#eee', zeroline: false,
        range: [SIG.scales.t_min, SIG.scales.t_max],
      },
      yaxis: {
        title: {text: 'Daily gas consumption (kWh/day)', font: {size: 15, color: '#444'}},
        gridcolor: '#eee', zeroline: false,
        range: [SIG.scales.q_min, SIG.scales.q_max],
      },
      showlegend: true,
      legend: {
        x: 1.02, y: 1, xanchor: 'left',
        font: {size: 15},
        bgcolor: 'rgba(255,255,255,0.85)',
        bordercolor: '#ddd', borderwidth: 1,
      },
    },
    config: {displaylogo: false, responsive: true,
             modeBarButtonsToRemove: ['lasso2d', 'select2d']}
  };
}

// ── master render ─────────────────────────────────────────────────────────────
function render() {
  var fig;
  if      (state.view === 'dist') fig = figureDist();
  else if (state.view === 'cldc') fig = figureCldc();
  else if (state.view === 'prof') fig = figureProf();
  else                            fig = figureSig();
  Plotly.react('plot', fig.data, fig.layout, fig.config).then(attachBarClickHandler);
}

// ── SIDE PANEL: bin contents ──────────────────────────────────────────────────
var sidePanelSort = {col: 'value', dir: 'desc'};
var lastBinMembers = [];
var lastBinMeta = null;

function openSidePanel() {
  document.getElementById('side-panel').classList.add('open');
}
function closeSidePanel() {
  document.getElementById('side-panel').classList.remove('open');
}

function renderSidePanelBin(members, binMeta) {
  lastBinMembers = members;
  lastBinMeta = binMeta;

  if (!members || members.length === 0) {
    document.getElementById('side-panel-content').innerHTML =
      '<h2>No buildings in this bin</h2>';
    openSidePanel();
    return;
  }

  var sorted = members.slice().sort(function(a, b) {
    var av = a[sidePanelSort.col], bv = b[sidePanelSort.col];
    if (typeof av === 'string') { av = av.toLowerCase(); bv = (bv || '').toLowerCase(); }
    if (av < bv) return sidePanelSort.dir === 'asc' ? -1 : 1;
    if (av > bv) return sidePanelSort.dir === 'asc' ? 1 : -1;
    return 0;
  });

  var html = '';
  html += '<h2>' + binMeta.title + '</h2>';
  html += '<div class="meta">' + binMeta.subtitle + ' · ' + members.length + ' building' +
          (members.length === 1 ? '' : 's') + '</div>';

  html += '<table>';
  html += '<thead><tr>' +
          '<th data-sort="id">ID</th>' +
          '<th data-sort="label">Name</th>' +
          '<th data-sort="function">Function</th>' +
          '<th data-sort="era">Era</th>' +
          '<th data-sort="value">Value</th>' +
          '</tr></thead><tbody>';
  sorted.forEach(function(mb) {
    html += '<tr>' +
            '<td><span class="id-link" data-bid="' + mb.id + '">' + mb.id + '</span></td>' +
            '<td>' + mb.label + '</td>' +
            '<td>' + (mb.function || 'n/a') + '</td>' +
            '<td>' + (mb.era || 'n/a') + '</td>' +
            '<td>' + mb.value + '</td>' +
            '</tr>';
  });
  html += '</tbody></table>';

  html += '<div class="btn-row">';
  html += '<button id="add-to-filter-btn" class="btn-light">Add all to filter</button>';
  html += '</div>';

  document.getElementById('side-panel-content').innerHTML = html;
  openSidePanel();

  document.querySelectorAll('#side-panel th').forEach(function(th) {
    th.addEventListener('click', function() {
      var col = th.getAttribute('data-sort');
      if (sidePanelSort.col === col) {
        sidePanelSort.dir = sidePanelSort.dir === 'asc' ? 'desc' : 'asc';
      } else {
        sidePanelSort.col = col;
        sidePanelSort.dir = (col === 'value') ? 'desc' : 'asc';
      }
      renderSidePanelBin(lastBinMembers, lastBinMeta);
    });
  });

  document.querySelectorAll('#side-panel .id-link').forEach(function(el) {
    el.addEventListener('click', function() {
      var bid = el.getAttribute('data-bid');
      var current = state.filter_raw ? state.filter_raw.split(/[,\\s]+/).filter(function(s) { return s.length > 0; }) : [];
      if (current.indexOf(bid) < 0) current.push(bid);
      var newRaw = current.join(', ');
      document.getElementById('filter-textarea').value = newRaw;
      updateFilterState(newRaw);
    });
  });

  document.getElementById('add-to-filter-btn').addEventListener('click', function() {
    var current = state.filter_raw ? state.filter_raw.split(/[,\\s]+/).filter(function(s) { return s.length > 0; }) : [];
    members.forEach(function(mb) {
      if (current.indexOf(mb.id) < 0) current.push(mb.id);
    });
    var newRaw = current.join(', ');
    document.getElementById('filter-textarea').value = newRaw;
    updateFilterState(newRaw);
  });
}

// ── SIDE PANEL: Top-N ─────────────────────────────────────────────────────────
var lastTopN = [];

function currentMetricForTopN() {
  if (state.view === 'dist') {
    var group = state.dist_group;
    var metrics = DIST.groups[group];
    return metrics[0];
  }
  if (state.view === 'cldc') {
    return state.cldc_metric;
  }
  return null;
}

function renderTopNPanel() {
  var metricKey = currentMetricForTopN();
  if (!metricKey) return;

  var rows = [];
  var metricLabel = '';

  if (state.view === 'dist') {
    var meta = DIST.meta[metricKey];
    metricLabel = meta.label + ' (' + meta.unit + ')';
    Object.keys(DIST.per_building_by_year).forEach(function(bid) {
      var yrVals = DIST.per_building_by_year[bid];
      var v = (yrVals[state.year] || {})[metricKey];
      if (v === null || v === undefined) return;
      rows.push({
        id: bid,
        label: (DIST.building_labels || {})[bid] || bid,
        value: v,
      });
    });
  } else if (state.view === 'cldc') {
    var md = CLDC.metrics[metricKey];
    metricLabel = md.label + ' (' + md.unit + ', ' + state.cldc_fuel + ')';
    var key = state.year + '_' + state.cldc_fuel;
    Object.keys(CLDC.per_building).forEach(function(bid) {
      var pb = (CLDC.per_building[bid] || {})[key];
      if (!pb) return;
      var v = pb[metricKey];
      if (v === null || v === undefined) return;
      rows.push({
        id: bid,
        label: (DIST.building_labels || {})[bid] || bid,
        value: v,
      });
    });
  }

  rows.sort(function(a, b) {
    return state.topn_direction === 'high' ? b.value - a.value : a.value - b.value;
  });
  rows = rows.slice(0, 10);
  lastTopN = rows;

  document.getElementById('topn-metric-label').textContent = metricLabel + ' · year ' + state.year;

  var html = '<table>';
  html += '<thead><tr><th>Rank</th><th>ID</th><th>Name</th><th>Value</th></tr></thead><tbody>';
  rows.forEach(function(r, i) {
    html += '<tr>' +
            '<td>' + (i + 1) + '</td>' +
            '<td>' + r.id + '</td>' +
            '<td>' + r.label + '</td>' +
            '<td>' + r.value + '</td>' +
            '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('topn-table-wrap').innerHTML = html;
}

// ── bar-click handler ─────────────────────────────────────────────────────────
function attachBarClickHandler() {
  var plotDiv = document.getElementById('plot');
  if (!plotDiv || !plotDiv.on) return;

  plotDiv.removeAllListeners && plotDiv.removeAllListeners('plotly_click');

  plotDiv.on('plotly_click', function(data) {
    if (!data.points || data.points.length === 0) return;
    var pt = data.points[0];
    if (pt.data.type !== 'bar') return;

    if (state.view === 'dist') {
      var xaxis = pt.data.xaxis || 'x';
      var suffixMatch = xaxis.match(/^x(\\d*)$/);
      var idx = suffixMatch && suffixMatch[1] ? parseInt(suffixMatch[1]) - 1 : 0;
      var metrics = DIST.groups[state.dist_group];
      var metricKey = metrics[idx];
      if (!metricKey) return;
      var md = DIST.metrics[metricKey];
      var yrData = md.per_year[state.year];
      if (!yrData) return;
      var binIdx = pt.pointIndex;
      var members = (yrData.members || [])[binIdx] || [];
      var meta = DIST.meta[metricKey];
      var lo = md.edges[binIdx].toFixed(2);
      var hi = md.edges[binIdx + 1].toFixed(2);
      renderSidePanelBin(members, {
        title: meta.label,
        subtitle: meta.unit + ' · bin ' + lo + ' – ' + hi + ' · year ' + state.year,
      });
    } else if (state.view === 'cldc') {
      var md = CLDC.metrics[state.cldc_metric];
      var bucket = (md.by_fuel[state.cldc_fuel] || {})[state.year];
      if (!bucket) return;
      var binIdx = pt.pointIndex;
      var members = (bucket.members || [])[binIdx] || [];
      var lo = md.edges[binIdx].toFixed(3);
      var hi = md.edges[binIdx + 1].toFixed(3);
      renderSidePanelBin(members, {
        title: md.label,
        subtitle: md.unit + ' · ' + state.cldc_fuel + ' · bin ' + lo + ' – ' + hi + ' · year ' + state.year,
      });
    }
  });
}

// ── export ────────────────────────────────────────────────────────────────────
function exportPlot(format) {
  var plotDiv = document.getElementById('plot');
  var filename = 'non_spatial_' + state.view + '_' + state.year;

  if (format === 'png') {
    Plotly.downloadImage(plotDiv, {
      format: 'png',
      filename: filename,
      scale: 2,
      width:  plotDiv.clientWidth,
      height: plotDiv.clientHeight,
    });
  } else if (format === 'svg') {
    Plotly.downloadImage(plotDiv, {
      format: 'svg',
      filename: filename,
      width:  plotDiv.clientWidth,
      height: plotDiv.clientHeight,
    });
  }
}

// ── populate selects ──────────────────────────────────────────────────────────
populateSelect('dist-group-select', Object.keys(DIST.groups));
populateSelect('cldc-metric-select',
  Object.keys(CLDC.metrics),
  Object.keys(CLDC.metrics).map(function(k) { return CLDC.metrics[k].label; })
);
populateSelect('prof-fuel-select',    PROF.fuels);
populateSelect('prof-season-select',  PROF.seasons);
populateSelect('prof-daytype-select', PROF.daytypes);
PROF.functions.forEach(function(fn) {
  var opt = document.createElement('option');
  opt.value = fn;
  opt.textContent = fn;
  document.getElementById('prof-function-select').appendChild(opt);
});

// ── wire events ───────────────────────────────────────────────────────────────
document.getElementById('view-select').addEventListener('change', function() {
  state.view = this.value;
  applyViewVisibility();
  refreshYearSlider();
  closeSidePanel();
  render();
});

document.getElementById('dist-group-select').addEventListener('change', function() {
  state.dist_group = this.value; render();
});
document.getElementById('cldc-metric-select').addEventListener('change', function() {
  state.cldc_metric = this.value; render();
});
document.getElementById('cldc-fuel-select').addEventListener('change', function() {
  state.cldc_fuel = this.value; render();
});
document.getElementById('prof-fuel-select').addEventListener('change', function() {
  state.prof_fuel = this.value; render();
});
document.getElementById('prof-season-select').addEventListener('change', function() {
  state.prof_season = this.value; render();
});
document.getElementById('prof-daytype-select').addEventListener('change', function() {
  state.prof_daytype = this.value; render();
});
document.getElementById('prof-function-select').addEventListener('change', function() {
  state.prof_highlight_fn = this.value; render();
});

document.getElementById('sig-context-toggle').addEventListener('click', function() {
  state.sig_show_cloud = !state.sig_show_cloud;
  this.classList.toggle('active', state.sig_show_cloud);
  this.textContent = state.sig_show_cloud ? 'on' : 'off';
  render();
});

var yearSlider  = document.getElementById('year-slider');
var yearDisplay = document.getElementById('year-display');
var playBtn     = document.getElementById('play-btn');

yearSlider.addEventListener('input', function() {
  var ys = yearsForView();
  state.year = ys[parseInt(this.value)];
  yearDisplay.textContent = state.year;
  render();
});

playBtn.addEventListener('click', function() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    playBtn.textContent = '▶ Play';
  } else {
    playBtn.textContent = '⏹ Stop';
    var idx = parseInt(yearSlider.value);
    playTimer = setInterval(function() {
      var ys = yearsForView();
      idx++;
      if (idx >= ys.length) idx = 0;
      yearSlider.value = idx;
      state.year = ys[idx];
      yearDisplay.textContent = state.year;
      render();
    }, 1000);
  }
});

document.getElementById('filter-btn').addEventListener('click', function(e) {
  e.stopPropagation();
  document.querySelectorAll('.panel').forEach(function(p) {
    if (p.id !== 'filter-panel') p.classList.remove('open');
  });
  document.getElementById('filter-panel').classList.toggle('open');
});
document.getElementById('filter-textarea').addEventListener('input', function() {
  updateFilterState(this.value);
});
document.getElementById('filter-clear-btn').addEventListener('click', function() {
  document.getElementById('filter-textarea').value = '';
  updateFilterState('');
});

document.getElementById('topn-btn').addEventListener('click', function(e) {
  e.stopPropagation();
  document.querySelectorAll('.panel').forEach(function(p) {
    if (p.id !== 'topn-panel') p.classList.remove('open');
  });
  var panel = document.getElementById('topn-panel');
  panel.classList.toggle('open');
  if (panel.classList.contains('open')) renderTopNPanel();
});
document.getElementById('topn-dir-btn').addEventListener('click', function() {
  state.topn_direction = (state.topn_direction === 'high') ? 'low' : 'high';
  this.textContent = state.topn_direction === 'high' ? '↑ Highest' : '↓ Lowest';
  renderTopNPanel();
});
document.getElementById('topn-filter-btn').addEventListener('click', function() {
  var current = state.filter_raw ? state.filter_raw.split(/[,\\s]+/).filter(function(s) { return s.length > 0; }) : [];
  lastTopN.forEach(function(r) {
    if (current.indexOf(r.id) < 0) current.push(r.id);
  });
  var newRaw = current.join(', ');
  document.getElementById('filter-textarea').value = newRaw;
  updateFilterState(newRaw);
});

document.getElementById('export-btn').addEventListener('click', function(e) {
  e.stopPropagation();
  document.querySelectorAll('.panel').forEach(function(p) {
    if (p.id !== 'export-panel') p.classList.remove('open');
  });
  document.getElementById('export-panel').classList.toggle('open');
});
document.getElementById('export-png-btn').addEventListener('click', function() { exportPlot('png'); });
document.getElementById('export-svg-btn').addEventListener('click', function() { exportPlot('svg'); });

document.querySelectorAll('.close-x').forEach(function(btn) {
  btn.addEventListener('click', function() {
    var target = btn.getAttribute('data-close');
    document.getElementById(target).classList.remove('open');
  });
});
document.getElementById('side-panel-close').addEventListener('click', closeSidePanel);

document.addEventListener('click', function(e) {
  document.querySelectorAll('.panel.open').forEach(function(p) {
    if (p.contains(e.target)) return;
    var triggerIds = {'filter-panel': 'filter-btn', 'topn-panel': 'topn-btn', 'export-panel': 'export-btn'};
    var triggerId = triggerIds[p.id];
    if (triggerId && document.getElementById(triggerId).contains(e.target)) return;
    p.classList.remove('open');
  });
});

window.addEventListener('resize', function() {
  Plotly.Plots.resize(document.getElementById('plot'));
});

// ── init ──────────────────────────────────────────────────────────────────────
applyViewVisibility();
refreshYearSlider();
render();
"""

    # ------------------------------------------------------------------------
    # Inject payload and assemble final HTML via concatenation (no f-string)
    # ------------------------------------------------------------------------
    js_final = js_block.replace('__PAYLOAD_JSON__', payload_json)

    html = (
        '<!DOCTYPE html>\n'
        '<html>\n'
        '<head>\n'
        '<meta charset="utf-8"/>\n'
        '<title>Cambridge Estates – Non-Spatial Metrics</title>\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<script src="https://cdn.plot.ly/plotly-' + plotly_version + '.min.js"></script>\n'
        '<style>\n' +
        css_block +
        '\n</style>\n'
        '</head>\n'
        '<body>\n' +
        html_body +
        '\n<script>\n' +
        js_final +
        '\n</script>\n'
        '</body>\n'
        '</html>\n'
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'\nDashboard saved to: {output_path}')


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':

    print('Loading inputs...')
    temporal = _load_or_compute_temporal()
    meta     = _load_or_compute_metadata()

    print('\nBuilding distributions payload...')
    dist_df      = _load_or_compute_distributions(temporal)
    dist_payload = _build_distributions_payload(temporal, meta)

    print('\nBuilding CLDC payload...')
    cldc_df      = _load_or_compute_cldc()
    cldc_payload = _build_cldc_payload(cldc_df, meta)

    print('\nBuilding profiles payload...')
    profiles_df  = _load_or_compute_profiles()
    prof_payload = _build_profiles_payload(profiles_df, meta)

    print('\nBuilding signatures payload...')
    sigs_df      = _load_or_compute_signatures()
    sig_payload  = _build_signatures_payload(sigs_df, temporal, meta)

    print('\nRendering HTML...')
    _render_html(dist_payload, cldc_payload, prof_payload, sig_payload, OUTPUT_HTML)