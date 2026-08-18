"""
gis_plotting.py
---------------
Single-page HTML map of the Cambridge estate, matching the UI chrome of
`non_spatial_plots.py` and `similarity_viewing.py`:

    Header → Controls bar → Description strip → Map (+ left side-panel) → Footer

Views:
    - Metric Map         : circles coloured by a chosen metric for one year
    - Year-on-year Change: circles coloured by Δ(metric) versus the previous year

Features:
    - View selector (Metric Map / YoY Change)
    - Metric selector grouped by theme (Intensity, Load shape, Fuel, Timing, Heating)
    - Year slider with Play button
    - Building filter panel (off-filter buildings fade to grey)
    - Building side-panel on click: metadata + all metrics for the selected year
    - Export panel: SVG (vectors only) and PNG (with basemap)
    - Short description per view
    - Heating Sensitivity and Balance-point temperature are truncated at 2022
      (weather data coverage beyond that is unreliable for the HS fit)

RUN
---
    pixi run python gis_plotting.py

Depends on:
    metricsV3.compute_all_buildings_temporal  (or `metrics_results_temporal.csv` cache)
    non_spatial_metrics._load_metadata_table  (or `metadata_table.csv` cache)
    CENTROID_CSV_PATH                         (from config.py)
"""

import os
import json
import numpy as np
import pandas as pd

from metricsV3           import compute_all_buildings_temporal, DATA_DIR
from non_spatial_metrics import _load_metadata_table
from config              import CENTROID_CSV_PATH

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_HTML     = 'gis_map.html'
TEMPORAL_CACHE  = 'metrics_results_temporal.csv'
METADATA_CACHE  = 'metadata_table.csv'
JOIN_ID_PREFIX  = 'b'

# Weather data reliability — HS fit becomes noisy after this cut-off, so
# `hs_slope` and `hs_balance` are suppressed for years > HS_MAX_YEAR.
HS_MAX_YEAR = 2022
HS_METRICS  = {'hs_slope', 'hs_balance'}

# (col, label, unit, palette, group)
METRICS = [
    ('eui_combined', 'EUI – Combined',          'kWh/m²/yr',  'sequential', 'Intensity'),
    ('eui_elec',     'EUI – Electricity',       'kWh/m²/yr',  'sequential', 'Intensity'),
    ('eui_gas',      'EUI – Gas',               'kWh/m²/yr',  'sequential', 'Intensity'),
    ('bi_elec',      'BI – Electricity',        'W/m²',       'sequential', 'Intensity'),
    ('bi_gas',       'BI – Gas',                'W/m²',       'sequential', 'Intensity'),
    ('pli_elec',     'PLI – Electricity',       'W/m²',       'sequential', 'Intensity'),
    ('pli_gas',      'PLI – Gas',               'W/m²',       'sequential', 'Intensity'),
    ('lf_elec',      'Load Factor – Elec',      '—',          'sequential', 'Load shape'),
    ('lf_gas',       'Load Factor – Gas',       '—',          'sequential', 'Load shape'),
    ('fmr',          'Fuel Mix Ratio',          'gas/elec',   'diverging',  'Fuel & carbon'),
    ('peak_hr_elec', 'Peak Hour – Elec',        'hour',       'cyclic',     'Timing'),
    ('peak_hr_gas',  'Peak Hour – Gas',         'hour',       'cyclic',     'Timing'),
    ('hs_slope',     'Heating Sensitivity',     'kWh/°C·day', 'sequential', 'Heating'),
    ('hs_balance',   'Balance-point temp.',     '°C',         'sequential', 'Heating'),
]

METRIC_GROUPS_ORDER = ['Intensity', 'Load shape', 'Fuel & carbon', 'Timing', 'Heating']

# Metrics that can drive circle SIZE. Magnitude-type quantities only —
# sizing by a diverging ratio (FMR) or a cyclic hour would be misleading,
# so those are excluded. 'gia' is a special-cased static option (default).
SIZE_METRICS = [
    ('gia',          'GIA (building size)'),
    ('eui_gas',      'EUI – Gas'),
    ('eui_elec',     'EUI – Electricity'),
    ('eui_combined', 'EUI – Combined'),
    ('bi_gas',       'BI – Gas'),
    ('bi_elec',      'BI – Electricity'),
    ('pli_gas',      'PLI – Gas'),
    ('pli_elec',     'PLI – Electricity'),
    ('hs_slope',     'Heating Sensitivity'),
]

# Circle radius range in pixels (min metric value → RADIUS_MIN, max → RADIUS_MAX).
RADIUS_MIN = 5.0
RADIUS_MAX = 20.0

# Colour palettes
SEQUENTIAL_SCALE = ['#ffffcc', '#fed976', '#fd8d3c', '#e31a1c', '#800026']
DIVERGING_SCALE  = ['#2166ac', '#67a9cf', '#f7f7f7', '#ef8a62', '#b2182b']
CYCLIC_SCALE = [
    '#2c1a4d',  # 00:00
    '#5a4a9f',  # 04:00
    '#f0e442',  # 08:00
    '#e69f00',  # 12:00
    '#d55e00',  # 16:00
    '#5a4a9f',  # 20:00
    '#2c1a4d',  # 24:00 (wraps to 00:00)
]
YOY_SCALE = ['#2166ac', '#67a9cf', '#f7f7f7', '#ef8a62', '#b2182b']

# View descriptions
DESCRIPTIONS = {
    'metric': (
        'Each circle is one building, positioned at its centroid. Circle area is '
        'proportional to Gross Internal Area; colour encodes the selected metric '
        'for the selected year. Grey circles have no data for this metric/year. '
        'Hover for a quick summary; click a circle to open full details on the left.'
    ),
    'yoy': (
        'Each circle shows how the selected metric changed from the previous year. '
        'Blue = decreased, red = increased, white ≈ unchanged. For Peak Hour metrics, '
        'the change uses circular arithmetic so 23:00 → 01:00 is a +2 h shift, not '
        '−22 h. Circles with no previous-year value are hidden.'
    ),
}


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_or_compute_temporal() -> pd.DataFrame:
    if os.path.exists(TEMPORAL_CACHE):
        print(f'  loading {TEMPORAL_CACHE}')
        return pd.read_csv(TEMPORAL_CACHE)
    print('  computing temporal metrics (no cache)')
    df = compute_all_buildings_temporal(data_dir=DATA_DIR)
    df.to_csv(TEMPORAL_CACHE, index=False)
    return df


def _load_or_compute_metadata() -> pd.DataFrame:
    if os.path.exists(METADATA_CACHE):
        print(f'  loading {METADATA_CACHE}')
        return pd.read_csv(METADATA_CACHE, index_col=0)
    print('  loading metadata (no cache)')
    meta = _load_metadata_table()
    meta.to_csv(METADATA_CACHE)
    return meta


def _build_dataframe() -> pd.DataFrame:
    """Join temporal metrics to centroid coordinates and enforce the HS cut-off."""
    df = _load_or_compute_temporal()

    # HS cut-off: weather-driven fit is unreliable after HS_MAX_YEAR.
    mask = df['year'] > HS_MAX_YEAR
    for col in HS_METRICS:
        if col in df.columns:
            df.loc[mask, col] = np.nan
    if 'hs_r2' in df.columns:
        df.loc[mask, 'hs_r2'] = np.nan

    df['_join_key'] = JOIN_ID_PREFIX + df['building_id'].astype(str)

    centroid_df = pd.read_csv(CENTROID_CSV_PATH)
    centroid_df['building_id'] = centroid_df['building_id'].astype(str).str.strip()

    merged = df.merge(centroid_df, left_on='_join_key',
                      right_on='building_id', how='left')

    for col in ['building_id_x', 'building_id_y', '_join_key']:
        if col in merged.columns:
            if col == 'building_id_x':
                merged.rename(columns={'building_id_x': 'building_id'}, inplace=True)
            else:
                merged.drop(columns=[col], errors='ignore', inplace=True)

    if 'year_x' in merged.columns:
        merged.rename(columns={'year_x': 'year'}, inplace=True)
        merged.drop(columns=['year_y'], errors='ignore', inplace=True)

    print(f'Rows: {len(merged)}')
    print(f'Years: {sorted(merged["year"].dropna().unique().astype(int).tolist())}')
    return merged


# =============================================================================
# METRIC SCALES
# =============================================================================

def _circular_diff(a: float, b: float, period: float = 24.0) -> float:
    """Signed smallest-arc difference on a circular scale (−period/2, period/2]."""
    d = (a - b) % period
    if d > period / 2:
        d -= period
    return d


def _build_colour_scales(df: pd.DataFrame) -> dict:
    """Compute vmin/vmax per metric (across all years) and YoY-change ranges."""
    scales = {}
    for col, _, _, palette, _ in METRICS:
        clean = df[col].dropna()
        entry = {'palette': palette}

        if palette == 'cyclic':
            entry.update({'vmin': 0.0, 'vmax': 24.0})

        elif palette == 'diverging':
            # FMR: symmetric on log scale around 1.0
            if len(clean) > 0:
                pos = clean[clean > 0]
                log_vals = np.log(pos) if len(pos) > 0 else np.array([])
                extent = (float(np.nanpercentile(np.abs(log_vals), 95))
                          if len(log_vals) > 0 else 1.0)
            else:
                extent = 1.0
            entry.update({
                'vmin': float(np.exp(-extent)),
                'vmax': float(np.exp( extent)),
                'vmid': 1.0,
            })

        else:  # sequential
            if len(clean) > 0:
                entry.update({
                    'vmin': float(np.nanpercentile(clean, 2)),
                    'vmax': float(np.nanpercentile(clean, 98)),
                })
            else:
                entry.update({'vmin': 0.0, 'vmax': 1.0})

        # Year-on-year change extent
        yoy_vals = []
        by_bid = df.dropna(subset=[col, 'year']).groupby('building_id')
        for _, g in by_bid:
            g = g.sort_values('year')
            vs = g[col].values
            if palette == 'cyclic':
                diffs = [_circular_diff(vs[i], vs[i-1]) for i in range(1, len(vs))]
            else:
                diffs = (vs[1:] - vs[:-1]).tolist()
            yoy_vals.extend(diffs)
        if len(yoy_vals) > 0:
            yoy_ext = float(np.nanpercentile(np.abs(yoy_vals), 95))
            if yoy_ext <= 0:
                yoy_ext = 1.0
        else:
            yoy_ext = 1.0
        entry['yoy_ext'] = yoy_ext

        scales[col] = entry
    return scales


def _value_to_colour(val: float, scale: dict) -> str:
    palette = scale.get('palette', 'sequential')

    if palette == 'cyclic':
        t = (float(val) % 24) / 24.0
        stops = CYCLIC_SCALE

    elif palette == 'diverging':
        vmin, vmax = scale['vmin'], scale['vmax']
        v = float(val)
        if v <= 0:
            t = 0.0
        else:
            lv, lmin, lmax = np.log(v), np.log(vmin), np.log(vmax)
            t = (lv - lmin) / (lmax - lmin) if lmax > lmin else 0.5
        t = max(0.0, min(1.0, t))
        stops = DIVERGING_SCALE

    else:  # sequential
        vmin, vmax = scale['vmin'], scale['vmax']
        t = ((float(val) - vmin) / (vmax - vmin)) if vmax > vmin else 0.5
        t = max(0.0, min(1.0, t))
        stops = SEQUENTIAL_SCALE

    return _blend(stops, t)


def _yoy_to_colour(delta: float, scale: dict) -> str:
    ext = scale.get('yoy_ext', 1.0)
    t = 0.5 + 0.5 * (float(delta) / ext) if ext > 0 else 0.5
    t = max(0.0, min(1.0, t))
    return _blend(YOY_SCALE, t)


def _blend(stops: list, t: float) -> str:
    n     = len(stops) - 1
    seg   = min(int(t * n), n - 1)
    local = t * n - seg

    def _h2r(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    c1, c2  = _h2r(stops[seg]), _h2r(stops[seg + 1])
    blended = tuple(int(c1[i] + local * (c2[i] - c1[i])) for i in range(3))
    return '#{:02x}{:02x}{:02x}'.format(*blended)


# =============================================================================
# METADATA HELPERS
# =============================================================================

def _build_meta_lookup(meta: pd.DataFrame) -> dict:
    """bid_str (without 'b' prefix) → metadata dict."""
    lookup = {}
    for bid_raw, row in meta.iterrows():
        key = str(bid_raw).lower().lstrip('b').strip()
        lookup[key] = {
            'name':      str(row.get('name_summary') or row.get('name_cebd') or f'b{key}'),
            'function':  row.get('function')   if pd.notna(row.get('function'))   else None,
            'era':       row.get('era')        if pd.notna(row.get('era'))        else None,
            'tenure':    row.get('tenure')     if pd.notna(row.get('tenure'))     else None,
            'date_built':row.get('date_built') if pd.notna(row.get('date_built')) else None,
            'n_floors':  int(row.get('n_floors')) if pd.notna(row.get('n_floors')) else None,
            'gia_m2':    float(row.get('gia_m2')) if pd.notna(row.get('gia_m2')) else None,
            'has_elec':  bool(row.get('has_elec')) if pd.notna(row.get('has_elec')) else None,
            'has_gas':   bool(row.get('has_gas'))  if pd.notna(row.get('has_gas'))  else None,
        }
    return lookup


# =============================================================================
# PAYLOAD
# =============================================================================

def _build_payload(df: pd.DataFrame, scales: dict, meta_lookup: dict) -> dict:
    """Build the JSON payload consumed by the front-end."""
    gia_clean = df['gia_m2'].dropna()
    gia_min   = float(np.nanpercentile(gia_clean, 2))  if len(gia_clean) > 0 else 0.0
    gia_max   = float(np.nanpercentile(gia_clean, 98)) if len(gia_clean) > 0 else 1.0

    # Per-metric 2nd/98th-percentile bounds for circle-size scaling. Robust
    # to outliers, mirroring the GIA treatment above. Values outside the
    # [2, 98] band are clamped to the radius limits at render time.
    def _radius(val, lo, hi):
        if val is None or pd.isna(val) or hi <= lo:
            return RADIUS_MIN
        frac = (float(val) - lo) / (hi - lo)
        frac = min(max(frac, 0.0), 1.0)
        return round(RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * frac, 1)

    size_bounds = {}
    for col, _ in SIZE_METRICS:
        if col == 'gia':
            size_bounds['gia'] = (gia_min, gia_max)
            continue
        s = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        if len(s) > 0:
            size_bounds[col] = (float(np.nanpercentile(s, 2)),
                                float(np.nanpercentile(s, 98)))
        else:
            size_bounds[col] = (0.0, 1.0)

    # Normalise the mixed int/float bid representation from pandas iterrows.
    def _bid_str(raw) -> str:
        try:
            return str(int(float(raw)))
        except (TypeError, ValueError):
            return str(raw).strip().lstrip('b')

    # Per-building point record, enriched with metadata.
    points = {}
    for bid, g in df.groupby('building_id'):
        lat = g['latitude'].dropna().mean()
        lon = g['longitude'].dropna().mean()
        if pd.isna(lat) or pd.isna(lon):
            continue
        gia_vals = g['gia_m2'].dropna()
        gia = float(gia_vals.iloc[0]) if len(gia_vals) > 0 else None
        radius = 7.0
        if gia is not None and gia_max > gia_min:
            radius = round(5 + 15 * (gia - gia_min) / (gia_max - gia_min), 1)

        bid_key = _bid_str(bid)
        m = meta_lookup.get(bid_key, {})
        points[bid_key] = {
            'lat':       round(float(lat), 6),
            'lon':       round(float(lon), 6),
            'gia':       round(gia, 1) if gia is not None else None,
            'r':         radius,
            'name':      m.get('name') or f'b{bid_key}',
            'function':  m.get('function'),
            'era':       m.get('era'),
            'tenure':    m.get('tenure'),
            'date_built':m.get('date_built'),
            'n_floors':  m.get('n_floors'),
            'has_elec':  m.get('has_elec'),
            'has_gas':   m.get('has_gas'),
        }

    # Previous-year lookup for YoY
    df_sorted = df.sort_values(['building_id', 'year'])
    prev_values = {}
    for _, row in df_sorted.iterrows():
        bid = _bid_str(row['building_id'])
        yr  = int(row['year']) if pd.notna(row['year']) else None
        if yr is None:
            continue
        prev_values.setdefault(bid, {})[yr] = {c: row.get(c) for c, _, _, _, _ in METRICS}

    # Per-year × per-building × per-metric
    by_year = {}
    for _, row in df.iterrows():
        bid = _bid_str(row['building_id'])
        if bid not in points:
            continue
        yr = int(row['year']) if pd.notna(row['year']) else None
        if yr is None:
            continue

        year_dict = by_year.setdefault(yr, {})
        bid_dict  = year_dict.setdefault(bid, {})

        # Per-metric radius for circle-size toggle. GIA is static (from the
        # point record); the rest are read from this row's per-year value.
        radii = {'gia': points[bid]['r']}
        for scol, _ in SIZE_METRICS:
            if scol == 'gia':
                continue
            lo, hi = size_bounds.get(scol, (0.0, 1.0))
            radii[scol] = _radius(row.get(scol), lo, hi)
        bid_dict['_r'] = radii

        for col, _, _, palette, _ in METRICS:
            v = row.get(col)
            entry = {'v': None, 'c': None, 'd': None, 'dc': None}
            if pd.notna(v):
                entry['v'] = round(float(v), 3)
                entry['c'] = _value_to_colour(float(v), scales[col])

                if col in HS_METRICS:
                    r2 = row.get('hs_r2')
                    if pd.isna(r2) or float(r2) < 0.5:
                        entry['poor_fit'] = True

                prev_yr = yr - 1
                prev = prev_values.get(bid, {}).get(prev_yr, {}).get(col)
                if prev is not None and pd.notna(prev):
                    if palette == 'cyclic':
                        delta = _circular_diff(float(v), float(prev))
                    else:
                        delta = float(v) - float(prev)
                    entry['d']  = round(delta, 3)
                    entry['dc'] = _yoy_to_colour(delta, scales[col])

            bid_dict[col] = entry

    # Tooltips — concise summary (original first-draft format).
    tooltips = {}
    for yr, yr_dict in by_year.items():
        tooltips[yr] = {}
        for bid, metrics_dict in yr_dict.items():
            pt  = points[bid]
            tip = f'<b>Building b{bid}</b> &nbsp;|&nbsp; <b>{yr}</b><br>'
            if pt['gia'] is not None:
                tip += f'GIA: {pt["gia"]:.0f} m²<br>'
            tip += '<hr style="margin:3px 0">'
            for col, label, unit, _, _ in METRICS:
                e = metrics_dict.get(col, {})
                v = e.get('v')
                if v is None:
                    tip += f'<b>{label}:</b> N/A<br>'
                else:
                    tip += f'<b>{label}:</b> {v:.2f} {unit}<br>'
            tooltips[yr][bid] = tip

    years = sorted(by_year.keys())

    return {
        'points':   points,
        'by_year':  by_year,
        'tooltips': tooltips,
        'years':    years,
        'scales':   scales,
        'metrics':  [
            {'col': c, 'label': l, 'unit': u, 'palette': p, 'group': g}
            for c, l, u, p, g in METRICS
        ],
        'size_metrics': [{'col': c, 'label': l} for c, l in SIZE_METRICS],
        'groups':   METRIC_GROUPS_ORDER,
        'palettes': {
            'sequential': SEQUENTIAL_SCALE,
            'diverging':  DIVERGING_SCALE,
            'cyclic':     CYCLIC_SCALE,
            'yoy':        YOY_SCALE,
        },
        'descriptions': DESCRIPTIONS,
        'hs_max_year':  HS_MAX_YEAR,
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
    if obj is None or isinstance(obj, (list, tuple, dict, str, bool, int, float)):
        return obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


# =============================================================================
# HTML
# =============================================================================

def _render_html(df: pd.DataFrame, payload: dict, output_path: str) -> None:

    valid = df.dropna(subset=['latitude', 'longitude'])
    centre_lat = float(valid['latitude'].mean())
    centre_lon = float(valid['longitude'].mean())

    payload_json = json.dumps(_to_jsonable(payload), separators=(',', ':'))

    # ------------------------------------------------------------------------
    # CSS
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
    z-index: 1100;
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
  .panel .btn-row { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }

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
    overflow: hidden;
    min-height: 500px;
    position: relative;
    display: flex;
    gap: 12px;
  }

  /* LEFT-HAND side panel for building details */
  #side-panel {
    flex: 0 0 340px;
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 16px 18px;
    overflow-y: auto;
    position: relative;
    display: none;
    font-size: 12px;
  }
  #side-panel.open { display: block; }
  #side-panel h2 { font-size: 15px; color: #222; margin-bottom: 4px; }
  #side-panel h3 {
    font-size: 12px; color: #444; margin: 12px 0 4px;
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  #side-panel .meta { font-size: 12px; color: #666; margin-bottom: 10px; line-height: 1.5; }
  #side-panel .meta span { display: inline-block; margin-right: 10px; }
  #side-panel .sp-close {
    position: absolute; top: 8px; right: 12px;
    cursor: pointer; font-size: 22px; color: #888;
    background: none; border: none; padding: 0;
  }
  #side-panel table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }
  #side-panel th, #side-panel td { padding: 4px 6px; text-align: left; border-bottom: 1px solid #eee; }
  #side-panel th { background: #f5f5f5; font-weight: bold; color: #444; }
  #side-panel td.val { text-align: right; font-variant-numeric: tabular-nums; }
  #side-panel .delta-up   { color: #b2182b; font-size: 11px; }
  #side-panel .delta-down { color: #2166ac; font-size: 11px; }
  #side-panel .delta-flat { color: #888;    font-size: 11px; }
  #side-panel .tag {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    background: #e8f0fe;
    color: #1a4d99;
    font-size: 11px;
    margin-right: 4px;
  }
  #side-panel .hs-note {
    font-size: 11px; color: #888; margin-top: 4px; font-style: italic;
  }

  #map-wrap {
    flex: 1 1 auto;
    position: relative;
    min-width: 0;
  }
  #map {
    width: 100%;
    height: 100%;
    min-height: 500px;
    border-radius: 4px;
    border: 1px solid #ddd;
  }

  #legend {
    position: absolute;
    bottom: 14px;
    right: 14px;
    z-index: 900;
    background: white;
    padding: 10px 14px;
    border-radius: 6px;
    border: 1px solid #ccc;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.18);
    font-size: 12px;
    min-width: 210px;
  }
  #legend-title { margin-bottom: 4px; color: #222; }
  #legend-bar {
    height: 12px;
    border-radius: 3px;
    margin: 6px 0 4px 0;
    border: 1px solid #ccc;
  }
  #legend-labels {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #555;
  }
  #legend-note {
    margin-top: 6px;
    font-size: 11px;
    color: #888;
    line-height: 1.4;
  }

  #footer {
    padding: 6px 22px;
    background: white;
    border-top: 1px solid #ddd;
    font-size: 11px;
    color: #888;
  }

  .hidden { display: none !important; }

  .leaflet-tooltip {
    font-family: Arial, sans-serif;
    font-size: 12px;
    max-width: 280px;
    white-space: normal;
  }
  .leaflet-interactive { cursor: pointer; }
"""

    # ------------------------------------------------------------------------
    # HTML body
    # ------------------------------------------------------------------------
    html_body = """
<div id="page">

  <div id="header">
    <h1>Cambridge Estates – Spatial Metrics Map</h1>
  </div>

  <div id="controls">

    <div class="control-group">
      <label>View</label>
      <select id="view-select">
        <option value="metric">Metric Map</option>
        <option value="yoy">Year-on-year Change</option>
      </select>
    </div>

    <div class="control-group">
      <label>Metric</label>
      <select id="metric-select"></select>
    </div>

    <div class="control-group">
      <label>Size by</label>
      <select id="size-select"></select>
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
    <div class="control-group">
      <button id="export-btn" class="btn-light">⬇ Export</button>
    </div>

    <!-- Filter panel -->
    <div id="filter-panel" class="panel">
      <button class="close-x" data-close="filter-panel">×</button>
      <label for="filter-textarea">Filter / highlight buildings</label>
      <textarea id="filter-textarea" placeholder="e.g. b7, b42, b150"></textarea>
      <div class="hint">Comma- or space-separated IDs. Matched buildings stay full-colour; others fade to grey.</div>
      <div id="filter-counter" class="counter">No buildings selected.</div>
      <button id="filter-clear-btn" class="btn-light">Clear selection</button>
    </div>

    <!-- Export panel -->
    <div id="export-panel" class="panel">
      <button class="close-x" data-close="export-panel">×</button>
      <label>Export current view</label>
      <div class="hint">SVG is vector (circles + legend, no basemap tiles — editable in Inkscape/Illustrator). PNG captures the full map including the basemap.</div>
      <div class="btn-row">
        <button id="export-svg-btn" class="btn-light">⬇ SVG (vector)</button>
        <button id="export-png-btn" class="btn-light">⬇ PNG (with basemap)</button>
      </div>
      <div class="hint" id="export-status" style="margin-top:8px;"></div>
    </div>

  </div>

  <div id="description"></div>

  <div id="plot-area">
    <div id="side-panel">
      <button class="sp-close" id="side-panel-close">×</button>
      <div id="side-panel-content"></div>
    </div>
    <div id="map-wrap">
      <div id="map"></div>
      <div id="legend">
        <div id="legend-title"><b>Legend</b></div>
        <div id="legend-bar"></div>
        <div id="legend-labels">
          <span id="legend-min"></span>
          <span id="legend-mid"></span>
          <span id="legend-max"></span>
        </div>
        <div id="legend-note">Circle size ∝ building GIA. Faded grey = no data.</div>
      </div>
    </div>
  </div>

  <div id="footer">
    Hover for a summary; click a circle for full details on the left.
    Use Filter to focus on specific buildings. Heating Sensitivity &amp; Balance-point temp. are truncated at <span id="footer-hs-year">2022</span>.
  </div>

</div>
"""

    # ------------------------------------------------------------------------
    # JavaScript
    # ------------------------------------------------------------------------
    js_block = r"""
// ══════════════════════════════════════════════════════════════════════════════
// Cambridge Estates GIS map
// ══════════════════════════════════════════════════════════════════════════════

var PAYLOAD      = __PAYLOAD_JSON__;
var POINTS       = PAYLOAD.points;
var BY_YEAR      = PAYLOAD.by_year;
var TOOLTIPS     = PAYLOAD.tooltips;
var YEARS        = PAYLOAD.years;
var SCALES       = PAYLOAD.scales;
var METRICS      = PAYLOAD.metrics;
var SIZE_METRICS = PAYLOAD.size_metrics;
var GROUPS       = PAYLOAD.groups;
var PALETTES     = PAYLOAD.palettes;
var DESCRIPTIONS = PAYLOAD.descriptions;
var HS_MAX_YEAR  = PAYLOAD.hs_max_year;

var HS_COLS = {hs_slope: true, hs_balance: true};

var state = {
  view:          'metric',
  metric:        METRICS[0].col,
  size_by:       'gia',
  year:          YEARS[YEARS.length - 1],
  filter_bids:   [],
  filter_raw:    '',
  selected_bid:  null,
};
var playTimer = null;
var circles   = [];

document.getElementById('footer-hs-year').textContent = HS_MAX_YEAR;

// ── map ───────────────────────────────────────────────────────────────────────
var map = L.map('map', {preferCanvas: false}).setView([__CENTRE_LAT__, __CENTRE_LON__], 15);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO',
  subdomains:  'abcd',
  maxZoom:     20,
}).addTo(map);

// ── metric dropdown (grouped) ─────────────────────────────────────────────────
(function populateMetricSelect() {
  var el = document.getElementById('metric-select');
  GROUPS.forEach(function(grp) {
    var og = document.createElement('optgroup');
    og.label = grp;
    METRICS.filter(function(m) { return m.group === grp; }).forEach(function(m) {
      var opt = document.createElement('option');
      opt.value       = m.col;
      opt.textContent = m.label + ' (' + m.unit + ')';
      og.appendChild(opt);
    });
    el.appendChild(og);
  });
})();

// ── size-by dropdown ──────────────────────────────────────────────────────────
(function populateSizeSelect() {
  var el = document.getElementById('size-select');
  SIZE_METRICS.forEach(function(m) {
    var opt = document.createElement('option');
    opt.value       = m.col;
    opt.textContent = m.label;
    el.appendChild(opt);
  });
  el.value = state.size_by;
})();

// ── year slider ───────────────────────────────────────────────────────────────
(function initYearSlider() {
  var slider = document.getElementById('year-slider');
  slider.min = 0;
  slider.max = Math.max(0, YEARS.length - 1);
  slider.value = YEARS.length - 1;
  document.getElementById('year-display').textContent = state.year;
})();

// ── filter parsing ────────────────────────────────────────────────────────────
function parseBuildingList(raw) {
  if (!raw) return [];
  return raw.split(/[,\s]+/)
    .map(function(s) { return s.trim(); })
    .filter(function(s) { return s.length > 0; })
    .map(function(s) { return s.toLowerCase().replace(/^b/, ''); });
}

function updateFilterState(rawText) {
  state.filter_raw = rawText;
  var requested    = parseBuildingList(rawText);
  state.filter_bids = requested.filter(function(bid) { return POINTS[bid]; });

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

// ── drawing ───────────────────────────────────────────────────────────────────
function clearCircles() {
  circles.forEach(function(c) { map.removeLayer(c); });
  circles = [];
}

function render() {
  clearCircles();
  document.getElementById('description').textContent = DESCRIPTIONS[state.view] || '';

  var yrData = BY_YEAR[state.year] || {};
  var tips   = TOOLTIPS[state.year] || {};
  var activeCol = state.metric;
  var hasFilter = state.filter_bids.length > 0;
  var filterSet = {};
  state.filter_bids.forEach(function(b) { filterSet[b] = true; });

  Object.keys(POINTS).forEach(function(bid) {
    var pt  = POINTS[bid];
    var rec = (yrData[bid] || {})[activeCol] || {};

    // Circle radius from the chosen size metric for this building-year.
    // Falls back to the static GIA radius if unavailable.
    var radii  = (yrData[bid] || {})['_r'] || {};
    var radius = radii[state.size_by];
    if (radius === undefined || radius === null) radius = pt.r;

    var value, colour;

    if (state.view === 'yoy') {
      value  = rec.d;
      colour = rec.dc;
    } else {
      value  = rec.v;
      colour = rec.c;
    }

    var hasData    = (value !== null && value !== undefined && colour);
    var dimmed     = hasFilter && !filterSet[bid];
    var isSelected = (state.selected_bid === bid);

    var fillColour = hasData ? colour : '#cccccc';
    var fillOpacity, strokeOpacity, strokeColour, strokeWeight;

    if (!hasData) {
      fillOpacity = 0.15;  strokeOpacity = 0.25;
    } else if (dimmed) {
      fillColour = '#cccccc'; fillOpacity = 0.25; strokeOpacity = 0.35;
    } else {
      fillOpacity = 0.85;  strokeOpacity = 1.0;
    }
    if (hasData && !dimmed && rec.poor_fit) {
      fillOpacity = 0.35;  strokeOpacity = 0.55;
    }

    strokeColour = isSelected ? '#111' : '#444';
    strokeWeight = isSelected ? 3      : 1;

    var c = L.circleMarker([pt.lat, pt.lon], {
      radius:      radius,
      color:       strokeColour,
      weight:      strokeWeight,
      opacity:     strokeOpacity,
      fillColor:   fillColour,
      fillOpacity: fillOpacity,
    });

    var tip = tips[bid];
    if (tip) c.bindTooltip(tip, {sticky: true, maxWidth: 280});

    c.on('click', function(e) {
      L.DomEvent.stopPropagation(e);
      selectBuilding(bid);
    });

    c.addTo(map);
    circles.push(c);
  });

  updateLegend();
}

// ── side panel ────────────────────────────────────────────────────────────────
function selectBuilding(bid) {
  state.selected_bid = bid;
  renderSidePanel(bid);
  render();
}

function closeSidePanel() {
  state.selected_bid = null;
  document.getElementById('side-panel').classList.remove('open');
  setTimeout(function() { map.invalidateSize(); }, 50);
  render();
}

function _arrow(d) {
  if (d === null || d === undefined) return '';
  if (d > 0)  return '<span class="delta-up">▲ +'   + d.toFixed(2) + '</span>';
  if (d < 0)  return '<span class="delta-down">▼ '  + d.toFixed(2) + '</span>';
  return          '<span class="delta-flat">◆ 0</span>';
}

function renderSidePanel(bid) {
  var pt = POINTS[bid];
  if (!pt) return;
  var yrData = (BY_YEAR[state.year] || {})[bid] || {};

  var html = '';
  html += '<h2>' + _htmlEscape(pt.name || ('Building b' + bid)) + '</h2>';
  html += '<div class="meta"><span class="tag">b' + bid + '</span>';
  if (pt.function)  html += '<span class="tag">' + _htmlEscape(pt.function) + '</span>';
  if (pt.era)       html += '<span class="tag">' + _htmlEscape(pt.era) + '</span>';
  html += '</div>';

  var bits = [];
  if (pt.gia !== null && pt.gia !== undefined) bits.push('<span><b>GIA:</b> ' + pt.gia.toFixed(0) + ' m²</span>');
  if (pt.n_floors)  bits.push('<span><b>Floors:</b> ' + pt.n_floors + '</span>');
  if (pt.date_built) bits.push('<span><b>Built:</b> ' + _htmlEscape(String(pt.date_built)) + '</span>');
  if (pt.tenure)    bits.push('<span><b>Tenure:</b> ' + _htmlEscape(pt.tenure) + '</span>');
  if (bits.length) html += '<div class="meta">' + bits.join('') + '</div>';

  var fuels = [];
  if (pt.has_elec) fuels.push('electricity');
  if (pt.has_gas)  fuels.push('gas');
  if (fuels.length) html += '<div class="meta"><b>Supplied:</b> ' + fuels.join(', ') + '</div>';

  html += '<h3>Metrics — ' + state.year + '</h3>';
  html += '<table><thead><tr><th>Metric</th><th style="text-align:right">Value</th><th style="text-align:right">Δ vs prev.</th></tr></thead><tbody>';

  var anyHsSuppressed = false;
  GROUPS.forEach(function(grp) {
    html += '<tr><td colspan="3" style="background:#fafafa; font-weight:bold; color:#666; padding-top:6px;">' + grp + '</td></tr>';
    METRICS.filter(function(m) { return m.group === grp; }).forEach(function(m) {
      var e = yrData[m.col] || {};
      var v = e.v, d = e.d;

      // HS suppression after HS_MAX_YEAR
      if (HS_COLS[m.col] && state.year > HS_MAX_YEAR) {
        anyHsSuppressed = true;
        html += '<tr><td>' + m.label + ' <span style="color:#888">(' + m.unit + ')</span></td>'
             +  '<td class="val" style="color:#aaa">—</td>'
             +  '<td class="val" style="color:#aaa">—</td></tr>';
        return;
      }

      var valStr = (v === null || v === undefined) ? '—' : (v.toFixed(2));
      var poor   = e.poor_fit ? ' <span style="color:#888; font-size:11px">(R²&lt;0.5)</span>' : '';
      html += '<tr><td>' + m.label + ' <span style="color:#888">(' + m.unit + ')</span>' + poor + '</td>';
      html += '<td class="val">' + valStr + '</td>';
      html += '<td class="val">' + _arrow(d === undefined ? null : d) + '</td>';
      html += '</tr>';
    });
  });
  html += '</tbody></table>';

  if (anyHsSuppressed) {
    html += '<div class="hs-note">Heating Sensitivity metrics are not shown for years after ' + HS_MAX_YEAR + ' (weather-data coverage).</div>';
  }

  html += '<div style="margin-top:10px; display:flex; gap:8px;">';
  html += '<button id="sp-add-filter" class="btn-light">Add to filter</button>';
  html += '<button id="sp-zoom" class="btn-light">Zoom to</button>';
  html += '</div>';

  document.getElementById('side-panel-content').innerHTML = html;
  document.getElementById('side-panel').classList.add('open');
  setTimeout(function() { map.invalidateSize(); }, 50);

  document.getElementById('sp-add-filter').addEventListener('click', function() {
    var ta = document.getElementById('filter-textarea');
    var cur = ta.value.trim();
    var list = cur ? cur.split(/[,\s]+/).filter(function(s){return s;}) : [];
    var tag = 'b' + bid;
    if (list.indexOf(tag) < 0 && list.indexOf(bid) < 0) list.push(tag);
    ta.value = list.join(', ');
    updateFilterState(ta.value);
  });
  document.getElementById('sp-zoom').addEventListener('click', function() {
    map.setView([pt.lat, pt.lon], Math.max(map.getZoom(), 18));
  });
}

function _htmlEscape(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;').replace(/'/g, '&apos;');
}

// ── legend ────────────────────────────────────────────────────────────────────
function updateLegend() {
  var meta    = METRICS.find(function(m) { return m.col === state.metric; });
  var scale   = SCALES[state.metric] || {};
  var palette = meta ? meta.palette : 'sequential';
  var isYoY   = state.view === 'yoy';

  var bar    = document.getElementById('legend-bar');
  var minLbl = document.getElementById('legend-min');
  var midLbl = document.getElementById('legend-mid');
  var maxLbl = document.getElementById('legend-max');
  var title  = document.getElementById('legend-title');
  var note   = document.getElementById('legend-note');

  // Human-readable label for the current circle-size metric.
  var sizeMeta = SIZE_METRICS.find(function(m) { return m.col === state.size_by; });
  var sizeLbl  = 'Circle size ∝ ' + (sizeMeta ? sizeMeta.label : 'GIA');

  if (isYoY) {
    title.innerHTML = '<b>Δ ' + meta.label + ' (' + meta.unit + ') &nbsp;— vs previous yr</b>';
    var ext = scale.yoy_ext || 1.0;
    bar.style.background = 'linear-gradient(to right,' + PALETTES.yoy.join(',') + ')';
    minLbl.textContent = '−' + ext.toFixed(2);
    midLbl.textContent = '0';
    maxLbl.textContent = '+' + ext.toFixed(2);
    midLbl.style.display = 'inline';
    note.innerHTML = sizeLbl + '. Grey = no previous-year value.<br>Range clipped at 95th pctile |Δ|.';
    return;
  }

  title.innerHTML = '<b>' + meta.label + ' (' + meta.unit + ')</b>';

  if (palette === 'cyclic') {
    bar.style.background = 'linear-gradient(to right,' + PALETTES.cyclic.join(',') + ')';
    minLbl.textContent = '00:00';
    midLbl.textContent = '12:00';
    maxLbl.textContent = '24:00';
    midLbl.style.display = 'inline';
    note.innerHTML = 'Peak Hour = modal hour of the top-5% readings.<br>Palette wraps: 00:00 ≡ 24:00.';
  } else if (palette === 'diverging') {
    bar.style.background = 'linear-gradient(to right,' + PALETTES.diverging.join(',') + ')';
    minLbl.textContent = '«elec (' + (scale.vmin || 0).toFixed(2) + ')';
    midLbl.textContent = '1.0';
    maxLbl.textContent = 'gas» (' + (scale.vmax || 0).toFixed(2) + ')';
    midLbl.style.display = 'inline';
    note.innerHTML = sizeLbl + '. Symmetric on log scale around FMR = 1.0.';
  } else {
    bar.style.background = 'linear-gradient(to right,' + PALETTES.sequential.join(',') + ')';
    minLbl.textContent = (scale.vmin || 0).toFixed(1);
    maxLbl.textContent = (scale.vmax || 0).toFixed(1);
    midLbl.style.display = 'none';
    note.innerHTML = sizeLbl + '. Range = 2nd–98th pctile across all years.';
  }
}

// ── EXPORT: SVG (vectors only) ────────────────────────────────────────────────
function exportSVG() {
  var status = document.getElementById('export-status');
  status.textContent = 'Building SVG…';

  var bounds = map.getBounds();
  var sw = bounds.getSouthWest();
  var ne = bounds.getNorthEast();

  var W = 900, H = 700;
  var latMid = (ne.lat + sw.lat) / 2;
  var lonMid = (ne.lng + sw.lng) / 2;
  var kx     = Math.cos(latMid * Math.PI / 180);
  var scale  = Math.min(W / ((ne.lng - sw.lng) * kx), H / (ne.lat - sw.lat)) * 0.92;
  var cx0 = W / 2, cy0 = H / 2;

  function proj(lat, lon) {
    return [
      cx0 + (lon - lonMid) * kx * scale,
      cy0 - (lat - latMid) * scale,
    ];
  }

  var yrData    = BY_YEAR[state.year] || {};
  var activeCol = state.metric;
  var hasFilter = state.filter_bids.length > 0;
  var filterSet = {};
  state.filter_bids.forEach(function(b) { filterSet[b] = true; });

  var meta = METRICS.find(function(m) { return m.col === activeCol; });
  var titleStr = (state.view === 'yoy' ? 'Δ ' : '') + meta.label + ' (' + meta.unit + ')  —  ' + state.year;

  var svg = [];
  svg.push('<?xml version="1.0" encoding="UTF-8"?>');
  svg.push('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + W + ' ' + (H + 80) + '" font-family="Arial, sans-serif">');
  svg.push('<rect x="0" y="0" width="' + W + '" height="' + (H + 80) + '" fill="#fafafa"/>');
  svg.push('<rect x="10" y="10" width="' + (W - 20) + '" height="' + H + '" fill="white" stroke="#ccc"/>');

  svg.push('<text x="20" y="35" font-size="16" font-weight="bold" fill="#222">' + _xmlEscape(titleStr) + '</text>');
  svg.push('<text x="20" y="55" font-size="12" fill="#666">Cambridge Estates — ' +
           (state.view === 'yoy' ? 'year-on-year change' : 'metric map') + '</text>');

  // Draw dimmed/no-data circles first, filtered-in on top
  var bidsSorted = Object.keys(POINTS).sort(function(a, b) {
    var aIn = hasFilter && filterSet[a];
    var bIn = hasFilter && filterSet[b];
    return (aIn ? 1 : 0) - (bIn ? 1 : 0);
  });

  bidsSorted.forEach(function(bid) {
    var pt  = POINTS[bid];
    var rec = (yrData[bid] || {})[activeCol] || {};
    var eRadii  = (yrData[bid] || {})['_r'] || {};
    var eRadius = eRadii[state.size_by];
    if (eRadius === undefined || eRadius === null) eRadius = pt.r;
    var value  = (state.view === 'yoy') ? rec.d : rec.v;
    var colour = (state.view === 'yoy') ? rec.dc : rec.c;
    var hasData = (value !== null && value !== undefined && colour);
    var dimmed  = hasFilter && !filterSet[bid];
    var p = proj(pt.lat, pt.lon);
    if (p[0] < 10 || p[0] > W - 10 || p[1] < 10 || p[1] > H + 10) return;

    var fill = hasData ? colour : '#cccccc';
    var fo   = 0.85;
    var stroke = '#444';
    var sw2 = 1;
    if (!hasData)    fo = 0.15;
    else if (dimmed) { fill = '#cccccc'; fo = 0.25; }
    if (hasData && !dimmed && rec.poor_fit) fo = 0.35;
    if (state.selected_bid === bid) { stroke = '#111'; sw2 = 2.5; }

    svg.push('<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="' + eRadius + '" ' +
             'fill="' + fill + '" fill-opacity="' + fo + '" stroke="' + stroke + '" stroke-width="' + sw2 + '"/>');
  });

  // Legend
  var legY = H + 20;
  svg.push('<text x="20" y="' + legY + '" font-size="12" font-weight="bold" fill="#222">' + _xmlEscape(_legendTitle()) + '</text>');
  var gradId = 'leg_' + Math.random().toString(36).slice(2, 8);
  var stops  = _legendStops();
  svg.push('<defs><linearGradient id="' + gradId + '" x1="0" x2="1" y1="0" y2="0">');
  stops.forEach(function(s, i) {
    svg.push('<stop offset="' + (i / (stops.length - 1)) + '" stop-color="' + s + '"/>');
  });
  svg.push('</linearGradient></defs>');
  svg.push('<rect x="20" y="' + (legY + 8) + '" width="260" height="14" fill="url(#' + gradId + ')" stroke="#ccc"/>');
  var lbls = _legendLabels();
  svg.push('<text x="20"  y="' + (legY + 40) + '" font-size="11" fill="#555">' + _xmlEscape(lbls.min) + '</text>');
  if (lbls.mid) {
    svg.push('<text x="150" y="' + (legY + 40) + '" font-size="11" fill="#555" text-anchor="middle">' + _xmlEscape(lbls.mid) + '</text>');
  }
  svg.push('<text x="280" y="' + (legY + 40) + '" font-size="11" fill="#555" text-anchor="end">' + _xmlEscape(lbls.max) + '</text>');
  svg.push('<text x="320" y="' + (legY + 20) + '" font-size="10" fill="#888">Circle area ∝ GIA</text>');
  svg.push('<text x="320" y="' + (legY + 35) + '" font-size="10" fill="#888">Basemap (CARTO) omitted from SVG export</text>');

  svg.push('</svg>');

  var blob = new Blob([svg.join('\n')], {type: 'image/svg+xml;charset=utf-8'});
  _downloadBlob(blob, 'gis_map_' + state.view + '_' + state.metric + '_' + state.year + '.svg');
  status.textContent = 'SVG downloaded.';
}

function _legendTitle() {
  var meta = METRICS.find(function(m) { return m.col === state.metric; });
  if (state.view === 'yoy') return 'Δ ' + meta.label + ' (' + meta.unit + ')  —  vs previous year';
  return meta.label + ' (' + meta.unit + ')';
}
function _legendStops() {
  var meta  = METRICS.find(function(m) { return m.col === state.metric; });
  var palette = meta ? meta.palette : 'sequential';
  if (state.view === 'yoy') return PALETTES.yoy;
  if (palette === 'cyclic')    return PALETTES.cyclic;
  if (palette === 'diverging') return PALETTES.diverging;
  return PALETTES.sequential;
}
function _legendLabels() {
  var meta  = METRICS.find(function(m) { return m.col === state.metric; });
  var scale = SCALES[state.metric] || {};
  var palette = meta ? meta.palette : 'sequential';
  if (state.view === 'yoy') {
    var ext = scale.yoy_ext || 1;
    return {min: '−' + ext.toFixed(2), mid: '0', max: '+' + ext.toFixed(2)};
  }
  if (palette === 'cyclic')    return {min: '00:00', mid: '12:00', max: '24:00'};
  if (palette === 'diverging') return {min: (scale.vmin || 0).toFixed(2), mid: '1.0', max: (scale.vmax || 0).toFixed(2)};
  return {min: (scale.vmin || 0).toFixed(1), mid: '', max: (scale.vmax || 0).toFixed(1)};
}
function _xmlEscape(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;').replace(/'/g, '&apos;');
}
function _downloadBlob(blob, name) {
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(function() { URL.revokeObjectURL(url); }, 1000);
}

// ── EXPORT: PNG (full map incl. basemap via dom-to-image) ─────────────────────
function exportPNG() {
  var status = document.getElementById('export-status');
  if (typeof domtoimage === 'undefined') {
    status.textContent = 'PNG library not loaded.';
    return;
  }
  status.textContent = 'Capturing map (this may take a few seconds)…';
  var node = document.getElementById('map-wrap');
  domtoimage.toPng(node, {
    bgcolor: '#fafafa',
    width:   node.offsetWidth  * 2,
    height:  node.offsetHeight * 2,
    style:   {transform: 'scale(2)', transformOrigin: 'top left'},
    cacheBust: true,
  }).then(function(dataUrl) {
    var a = document.createElement('a');
    a.href = dataUrl;
    a.download = 'gis_map_' + state.view + '_' + state.metric + '_' + state.year + '.png';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    status.textContent = 'PNG downloaded.';
  }).catch(function(err) {
    status.textContent = 'PNG export failed (likely a basemap CORS issue). Try SVG instead.';
    console.error(err);
  });
}

// ── wire events ───────────────────────────────────────────────────────────────
document.getElementById('view-select').addEventListener('change', function() {
  state.view = this.value;
  render();
  if (state.selected_bid) renderSidePanel(state.selected_bid);
});

document.getElementById('metric-select').addEventListener('change', function() {
  state.metric = this.value;
  render();
});

document.getElementById('size-select').addEventListener('change', function() {
  state.size_by = this.value;
  render();
  updateLegend();
});

var yearSlider  = document.getElementById('year-slider');
var yearDisplay = document.getElementById('year-display');
var playBtn     = document.getElementById('play-btn');

yearSlider.addEventListener('input', function() {
  state.year = YEARS[parseInt(this.value)];
  yearDisplay.textContent = state.year;
  render();
  if (state.selected_bid) renderSidePanel(state.selected_bid);
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
      idx++;
      if (idx >= YEARS.length) idx = 0;
      yearSlider.value = idx;
      state.year = YEARS[idx];
      yearDisplay.textContent = state.year;
      render();
      if (state.selected_bid) renderSidePanel(state.selected_bid);
    }, 900);
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

document.getElementById('export-btn').addEventListener('click', function(e) {
  e.stopPropagation();
  document.querySelectorAll('.panel').forEach(function(p) {
    if (p.id !== 'export-panel') p.classList.remove('open');
  });
  document.getElementById('export-panel').classList.toggle('open');
});
document.getElementById('export-svg-btn').addEventListener('click', exportSVG);
document.getElementById('export-png-btn').addEventListener('click', exportPNG);

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
    var triggerIds = {'filter-panel': 'filter-btn', 'export-panel': 'export-btn'};
    var triggerId = triggerIds[p.id];
    if (triggerId && document.getElementById(triggerId).contains(e.target)) return;
    p.classList.remove('open');
  });
});

window.addEventListener('resize', function() { map.invalidateSize(); });

// ── init ──────────────────────────────────────────────────────────────────────
document.getElementById('description').textContent = DESCRIPTIONS[state.view] || '';
render();
setTimeout(function() { map.invalidateSize(); }, 100);
"""

    js_final = (js_block
                .replace('__PAYLOAD_JSON__', payload_json)
                .replace('__CENTRE_LAT__', f'{centre_lat:.6f}')
                .replace('__CENTRE_LON__', f'{centre_lon:.6f}'))

    html = (
        '<!DOCTYPE html>\n'
        '<html>\n'
        '<head>\n'
        '<meta charset="utf-8"/>\n'
        '<title>Cambridge Estates – Spatial Metrics Map</title>\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>\n'
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>\n'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/dom-to-image/2.6.0/dom-to-image.min.js"></script>\n'
        '<style>\n' + css_block + '\n</style>\n'
        '</head>\n'
        '<body>\n' + html_body +
        '\n<script>\n' + js_final + '\n</script>\n'
        '</body>\n'
        '</html>\n'
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'\nMap saved to: {output_path}')
    print(f'Years: {payload["years"][0]} – {payload["years"][-1]}')
    print(f'Buildings with coordinates: {len(payload["points"])}')
    print(f'HS metrics truncated at: {HS_MAX_YEAR}')
    print('Open the HTML file in your browser.')


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':

    print('Loading data...')
    df          = _build_dataframe()
    meta        = _load_or_compute_metadata()
    meta_lookup = _build_meta_lookup(meta)

    print('\nBuilding colour scales...')
    scales = _build_colour_scales(df)

    print('\nBuilding payload...')
    payload = _build_payload(df, scales, meta_lookup)

    n_records = sum(len(m) for yr in payload['by_year'].values() for m in yr.values())
    print(f'Total metric records: {n_records:,}')

    print('\nRendering HTML...')
    _render_html(df, payload, OUTPUT_HTML)