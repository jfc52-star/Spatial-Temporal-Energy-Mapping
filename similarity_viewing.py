"""
similarity_viewing.py
---------------------
Single-page HTML for the building-similarity view.

Views:
  - HEATMAP (default) : reordered distance matrix with cluster colour strips
  - MDS SCATTER       : 2D embedding of the same distance matrix
  - DENDROGRAM        : truncated hierarchical-clustering tree

Controls:
  - Distance metric dropdown
  - Reference year dropdown (matrices and embeddings recomputed per year)
  - Colour-by (function / era / cluster)
  - K slider (number of clusters)
  - Building lookup (type an id to jump to a building)

Side panel:
  - Default state (no building picked, colour-by != cluster):
      "Most outlying buildings on this metric" — sorted table with scores
  - Colour-by = cluster (no building picked):
      Collapsible cluster contents + CSV export
  - Building picked (single): metadata + nearest neighbours + comparison
                              bar chart + daily profile chart
  - Pair picked (two buildings): side-by-side comparison + profile chart

Reads `similarity_cache.pkl` produced by similarity_metrics.py.

RUN
---
    pixi run python similarity_viewing.py
"""

import os
import json
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance  import squareform
from plotly.offline          import get_plotlyjs_version

from metricsV3          import compute_all_buildings_temporal, DATA_DIR
from non_spatial_metrics import _load_metadata_table
from similarity_metrics  import (
    CACHE_PATH,
    compute_all_years,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_HTML    = 'similarity.html'
TEMPORAL_CACHE = 'metrics_results_temporal.csv'
METADATA_CACHE = 'metadata_table.csv'
PROFILES_CACHE = 'daily_profiles.pkl'

METRIC_LABELS = {
    'gower':                  'Gower (mixed features)',
    'euclidean':              'Normalised Euclidean (operational)',
    'profile_euclidean_elec': 'Load Profile Euclidean – Elec',
    'profile_euclidean_gas':  'Load Profile Euclidean – Gas',
    'profile_kl_elec':        'KL Divergence – Elec',
    'profile_kl_gas':         'KL Divergence – Gas',
    'peak_timing_elec':       'Peak Timing – Elec',
    'peak_timing_gas':        'Peak Timing – Gas',
}

COMPARISON_METRICS = [
    ('eui_combined',  'EUI (combined)'),
    ('bi_elec',       'BI elec'),
    ('bi_gas',        'BI gas'),
    ('lf_elec',       'LF elec'),
    ('fmr',           'FMR'),
    ('peak_hr_elec',  'Peak hr elec'),
    ('peak_hr_gas',   'Peak hr gas'),
]


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_or_compute_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        print(f'  loading {CACHE_PATH}')
        return pd.read_pickle(CACHE_PATH)

    print('  no similarity cache found — recomputing (this will take several minutes)')
    temporal = pd.read_csv(TEMPORAL_CACHE) if os.path.exists(TEMPORAL_CACHE) \
               else compute_all_buildings_temporal(data_dir=DATA_DIR)
    meta     = pd.read_csv(METADATA_CACHE, index_col=0) if os.path.exists(METADATA_CACHE) \
               else _load_metadata_table()
    if not os.path.exists(PROFILES_CACHE):
        raise FileNotFoundError(
            f'{PROFILES_CACHE} not found. Run non_spatial_metrics.py first.'
        )
    profiles = pd.read_pickle(PROFILES_CACHE)

    payload = compute_all_years(temporal, profiles, meta)
    pd.to_pickle(payload, CACHE_PATH)
    return payload


def _load_temporal() -> pd.DataFrame:
    if os.path.exists(TEMPORAL_CACHE):
        return pd.read_csv(TEMPORAL_CACHE)
    df = compute_all_buildings_temporal(data_dir=DATA_DIR)
    df.to_csv(TEMPORAL_CACHE, index=False)
    return df


def _load_metadata() -> pd.DataFrame:
    if os.path.exists(METADATA_CACHE):
        return pd.read_csv(METADATA_CACHE, index_col=0)
    meta = _load_metadata_table()
    meta.to_csv(METADATA_CACHE)
    return meta


# =============================================================================
# PAYLOAD BUILDERS
# =============================================================================

def _build_building_info_by_year(
    all_bids: list,
    temporal_df: pd.DataFrame,
    meta: pd.DataFrame,
    years: list,
) -> dict:
    """Per-building info containing static metadata + per-year metric values."""
    yr_lookup = {}
    for yr in years:
        sub = temporal_df[temporal_df['year'] == yr].copy()
        sub['bid_str'] = 'b' + sub['building_id'].astype(str)
        yr_lookup[int(yr)] = sub.set_index('bid_str').to_dict(orient='index')

    out = {}
    for bid in all_bids:
        m = meta.loc[bid] if bid in meta.index else None
        base = {
            'id':       bid,
            'label':    str(m.get('name_summary') or m.get('name_cebd') or bid) if m is not None else bid,
            'function': (m.get('function') if m is not None else None) or 'n/a',
            'era':      (m.get('era')      if m is not None else None) or 'n/a',
            'tenure':   (m.get('tenure')   if m is not None else None) or 'n/a',
            'gia_m2':   float(m.get('gia_m2')) if (m is not None and pd.notna(m.get('gia_m2'))) else None,
            'metrics_by_year': {},
        }
        for yr in years:
            t = yr_lookup[int(yr)].get(bid, {})
            year_metrics = {}
            for col, _ in COMPARISON_METRICS:
                v = t.get(col)
                year_metrics[col] = round(float(v), 3) if v is not None and pd.notna(v) else None
            base['metrics_by_year'][int(yr)] = year_metrics
        out[bid] = base
    return out


def _build_profile_payload(profiles_df: pd.DataFrame, building_ids: list, years: list) -> dict:
    """Per-building per-year sum-to-1 profiles."""
    out = {}
    for yr in years:
        sub = profiles_df[profiles_df['year'] == yr]
        year_out = {}
        for bid_int, grp in sub.groupby('building_id'):
            bid_str = f'b{int(bid_int)}'
            if bid_str not in building_ids:
                continue
            entry = {}
            for _, row in grp.iterrows():
                fuel    = row['fuel']
                season  = row['season']
                daytype = row['daytype']
                prof    = row.get('profile_sum1')
                if prof is None or (isinstance(prof, float) and np.isnan(prof)):
                    continue
                entry.setdefault(fuel, {}).setdefault(season, {})[daytype] = \
                    [round(float(v), 5) for v in prof]
            if entry:
                year_out[bid_str] = entry
        out[int(yr)] = year_out
    return out


def _reorder_matrix(mat: np.ndarray) -> np.ndarray:
    if len(mat) < 2:
        return np.arange(len(mat))
    condensed = squareform(mat, checks=False)
    Z         = linkage(condensed, method='average')
    return leaves_list(Z)


def _colourscale_bounds(mat: np.ndarray) -> tuple:
    n = len(mat)
    if n < 2:
        return 0.0, 1.0
    mask = ~np.eye(n, dtype=bool)
    vals = mat[mask]
    lo = float(np.nanpercentile(vals,  2))
    hi = float(np.nanpercentile(vals, 98))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _compute_outlier_scores(mat: np.ndarray) -> list:
    """Mean distance from each building to all others — higher = more outlying."""
    if len(mat) < 2:
        return []
    n = len(mat)
    # Sum each row, subtract self-distance (0) and divide by (n-1)
    return (mat.sum(axis=1) / (n - 1)).tolist()


def _build_payload(cache: dict, temporal_df, meta, profiles_df) -> dict:
    """Assemble everything the HTML needs."""
    years = cache['years']

    # Union of all building IDs across all (metric, year)
    all_bids = sorted({
        bid
        for yr_data in cache['per_year'].values()
        for bid_list in yr_data['building_ids'].values()
        for bid in bid_list
    })

    building_info   = _build_building_info_by_year(all_bids, temporal_df, meta, years)
    profile_payload = _build_profile_payload(profiles_df, all_bids, years)

    # Build the per-(year, metric) payload used by the JS
    per_year = {}
    for yr, yr_data in cache['per_year'].items():
        yr_payload = {}
        for metric, mat in yr_data['distances'].items():
            bids     = yr_data['building_ids'][metric]
            emb_data = yr_data['embeddings'][metric]
            emb      = emb_data['embedding']
            clusters = emb_data['clusters']
            dend     = emb_data['dendrogram']

            order          = _reorder_matrix(mat)
            ordered_matrix = mat[np.ix_(order, order)]
            ordered_bids   = [bids[i] for i in order]

            vmin, vmax = _colourscale_bounds(mat)

            top5 = []
            for i in range(len(bids)):
                distances = mat[i].copy()
                distances[i] = np.inf
                nearest = np.argsort(distances)[:5]
                top5.append([
                    {'id': bids[j], 'dist': round(float(mat[i, j]), 4)}
                    for j in nearest
                ])

            outlier_scores = _compute_outlier_scores(mat)

            yr_payload[metric] = {
                'n_buildings':  len(bids),
                'building_ids': bids,
                'ordered_ids':  ordered_bids,
                'order':        [int(x) for x in order],
                'embedding':    [[round(float(x), 3), round(float(y), 3)] for x, y in emb],
                'clusters':     {str(k): [int(c) for c in v] for k, v in clusters.items()},
                'heatmap':      [[round(float(v), 4) for v in row] for row in ordered_matrix],
                'colourscale':  {'vmin': round(vmin, 4), 'vmax': round(vmax, 4)},
                'top5':         top5,
                'outlier_scores': [round(float(s), 4) for s in outlier_scores],
                'dendrogram':   dend,
            }
        per_year[int(yr)] = yr_payload

    metric_labels = {m: METRIC_LABELS.get(m, m) for m in cache['feature_info_static']}
    metric_infos  = cache['feature_info_static']

    return {
        'years':               years,
        'per_year':            per_year,
        'metric_labels':       metric_labels,
        'metric_infos':        metric_infos,
        'metric_order':        list(cache['feature_info_static'].keys()),
        'building_info':       building_info,
        'profiles':            profile_payload,
        'comparison_keys':     [c for c, _ in COMPARISON_METRICS],
        'comparison_labels':   {c: l for c, l in COMPARISON_METRICS},
        'cluster_ks':          sorted({int(k) for yr_data in per_year.values()
                                              for m in yr_data.values()
                                              for k in m['clusters']}),
    }


# =============================================================================
# JSON SANITISER
# =============================================================================

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

def _render_html(payload: dict, output_path: str) -> None:
    payload_json   = json.dumps(_to_jsonable(payload), separators=(',', ':'))
    plotly_version = get_plotlyjs_version()
    default_year   = payload['years'][-1]
    default_metric = payload['metric_order'][0]
    cluster_ks     = payload['cluster_ks']
    default_k      = 6 if 6 in cluster_ks else cluster_ks[len(cluster_ks) // 2]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Cambridge Estates – Building Similarity</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.plot.ly/plotly-{plotly_version}.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: 100%; height: 100%; font-family: Arial, sans-serif; background: #fafafa; color: #222; }}
  #page {{ display: flex; flex-direction: column; height: 100%; }}

  #header {{ padding: 14px 22px; background: white; border-bottom: 1px solid #ddd; }}
  #header h1 {{ font-size: 18px; font-weight: 600; color: #222; margin-bottom: 2px; }}
  #header .sub {{ font-size: 12px; color: #888; }}

  #controls {{
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    padding: 10px 22px;
    background: white;
    border-bottom: 1px solid #ddd;
  }}
  .control-group {{ display: flex; align-items: center; gap: 6px; }}
  .control-group label {{ font-size: 12px; font-weight: bold; color: #555; }}
  select, input[type="range"], input[type="text"] {{
    padding: 4px 6px;
    font-size: 12px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background: white;
  }}
  input[type="range"] {{ padding: 0; width: 120px; }}
  input[type="text"]  {{ width: 120px; }}

  .view-toggle {{
    display: flex;
    gap: 0;
    border: 1px solid #ccc;
    border-radius: 4px;
    overflow: hidden;
  }}
  .view-toggle button {{
    padding: 5px 12px;
    font-size: 12px;
    background: #f7f7f7;
    border: none;
    cursor: pointer;
    color: #444;
    border-right: 1px solid #ccc;
  }}
  .view-toggle button:last-child {{ border-right: none; }}
  .view-toggle button.active {{ background: #3388ff; color: white; font-weight: bold; }}

  #lookup-results {{
    position: absolute;
    background: white;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.1);
    max-height: 240px;
    overflow: auto;
    z-index: 100;
    min-width: 200px;
    display: none;
  }}
  #lookup-results .item {{
    padding: 5px 10px;
    cursor: pointer;
    font-size: 12px;
    border-bottom: 1px solid #f0f0f0;
  }}
  #lookup-results .item:hover {{ background: #f0f8ff; }}
  #lookup-results .item:last-child {{ border-bottom: none; }}

  #main {{
    flex: 1 1 auto;
    display: flex;
    min-height: 500px;
    overflow: hidden;
  }}
  #plot-area {{
    flex: 1 1 auto;
    padding: 8px 12px;
    overflow: hidden;
    position: relative;
  }}
  #plot {{ width: 100%; height: 100%; min-height: 500px; }}

  #side-panel {{
    width: 380px;
    flex: 0 0 380px;
    background: white;
    border-left: 1px solid #ddd;
    padding: 14px 18px;
    overflow: auto;
  }}
  #side-panel h2 {{ font-size: 15px; color: #222; margin-bottom: 4px; }}
  #side-panel h3 {{ font-size: 13px; color: #444; margin: 12px 0 4px; }}
  #side-panel .meta {{ font-size: 12px; color: #666; margin-bottom: 12px; }}
  #side-panel .meta span {{ display: inline-block; margin-right: 10px; }}
  #side-panel .placeholder {{ color: #888; font-size: 13px; padding: 12px 0; }}

  #neighbour-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 6px;
  }}
  #neighbour-table th, #neighbour-table td {{
    padding: 4px 6px;
    text-align: left;
    border-bottom: 1px solid #eee;
  }}
  #neighbour-table th {{ background: #f5f5f5; font-weight: bold; color: #444; }}

  #comparison-plot {{ width: 100%; height: 220px; margin-top: 8px; }}
  #profile-plot    {{ width: 100%; height: 200px; margin-top: 4px; }}

  #footer {{
    padding: 6px 22px;
    background: white;
    border-top: 1px solid #ddd;
    font-size: 11px;
    color: #888;
  }}
  .note {{ font-size: 11px; color: #888; margin-left: auto; }}
</style>
</head>
<body>

<div id="page">

  <div id="header">
    <h1>Cambridge Estates – Building Similarity</h1>
    <div class="sub" id="view-subtitle">
      Heatmap: reordered pairwise distance matrix. Click a row/col to pick a building.
    </div>
  </div>

  <div id="controls">
    <div class="control-group">
      <label>View</label>
      <div class="view-toggle">
        <button id="view-heatmap" class="active">Heatmap</button>
        <button id="view-mds">MDS scatter</button>
        <button id="view-dendro">Dendrogram</button>
      </div>
    </div>

    <div class="control-group">
      <label>Year</label>
      <select id="year-select"></select>
    </div>

    <div class="control-group">
      <label>Distance metric</label>
      <select id="metric-select"></select>
    </div>

    <div class="control-group">
      <label>Colour by</label>
      <select id="colour-select">
        <option value="function">Function type</option>
        <option value="era">Era</option>
        <option value="cluster">Cluster (K)</option>
      </select>
    </div>

    <div class="control-group" id="k-control">
      <label>K</label>
      <input id="k-slider" type="range" min="0" max="0" value="0">
      <span id="k-display" style="font-weight:bold; min-width:22px; text-align:center;">{default_k}</span>
    </div>

    <div class="control-group" style="position:relative;">
      <label>Find</label>
      <input id="lookup-input" type="text" placeholder="e.g. b37" autocomplete="off">
      <div id="lookup-results"></div>
    </div>

    <div class="note" id="n-buildings-display"></div>
  </div>

  <div id="main">
    <div id="plot-area">
      <div id="plot"></div>
    </div>
    <div id="side-panel">
      <div id="panel-content" class="placeholder">
        Click a building in the plot to see its details and nearest neighbours.
      </div>
    </div>
  </div>

  <div id="footer">
    Heatmap: rows/cols hierarchically reordered (darker cells = more similar). MDS: close points are similar. Dendrogram: truncated to top 20 merges.
  </div>
</div>

<script>
// ── data ──────────────────────────────────────────────────────────────────────
var PAYLOAD    = {payload_json};
var YEARS      = PAYLOAD.years;
var PER_YEAR   = PAYLOAD.per_year;
var INFO       = PAYLOAD.building_info;
var PROFILES   = PAYLOAD.profiles || {{}};
var CMP_KEYS   = PAYLOAD.comparison_keys;
var CMP_LABELS = PAYLOAD.comparison_labels;
var CLUSTER_KS = PAYLOAD.cluster_ks;
var METRIC_LABELS = PAYLOAD.metric_labels;
var METRIC_INFOS  = PAYLOAD.metric_infos;

var state = {{
  view:      'heatmap',
  year:      {default_year},
  metric:    '{default_metric}',
  colour:    'function',
  k:         {default_k},
  selected:  null,
  selectedA: null,
  selectedB: null,
}};

var profileState = {{
  fuel:    null,
  season:  'winter',
  daytype: 'weekday',
}};

// ── helpers ───────────────────────────────────────────────────────────────────
function currentMetricData() {{
  return (PER_YEAR[state.year] || {{}})[state.metric] || null;
}}
function metricExistsForYear(metric, year) {{
  return PER_YEAR[year] && PER_YEAR[year][metric];
}}
function buildingInfoMetrics(bid) {{
  var info = INFO[bid];
  if (!info) return {{}};
  var yrMetrics = (info.metrics_by_year || {{}})[state.year] || {{}};
  return yrMetrics;
}}

// Profile helpers
function hasFuel(bid, fuel) {{
  var yearProfiles = PROFILES[state.year] || {{}};
  var b = yearProfiles[bid];
  if (!b) return false;
  var f = b[fuel];
  if (!f) return false;
  for (var s in f) for (var d in f[s]) if (f[s][d]) return true;
  return false;
}}
function defaultFuelFor(metricKey, bids) {{
  if (metricKey && metricKey.endsWith('_elec')) return 'elec';
  if (metricKey && metricKey.endsWith('_gas'))  return 'gas';
  for (var i = 0; i < bids.length; i++) {{
    if (hasFuel(bids[i], 'elec')) return 'elec';
  }}
  return 'gas';
}}
function profileFor(bid, fuel, season, daytype) {{
  var yearProfiles = PROFILES[state.year] || {{}};
  var b = yearProfiles[bid];
  if (!b) return null;
  var f = b[fuel];
  if (!f) return null;
  var s = f[season];
  if (!s) return null;
  return s[daytype] || null;
}}

// Categorical palette
var PALETTE = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#5254a3', '#ad494a', '#8ca252', '#bd9e39', '#6b6ecf',
];
function colourFor(value, domain) {{
  var idx = domain.indexOf(value);
  if (idx < 0) idx = domain.length;
  return PALETTE[idx % PALETTE.length];
}}

var DIST_COLORSCALE = [
  [0.0, '#800026'], [0.2, '#e31a1c'], [0.4, '#fd8d3c'],
  [0.6, '#fed976'], [0.8, '#ffffcc'], [1.0, '#ffffff'],
];

// ── populate controls ─────────────────────────────────────────────────────────
var yearSelect = document.getElementById('year-select');
YEARS.forEach(function(y) {{
  var opt = document.createElement('option');
  opt.value = y; opt.textContent = y;
  yearSelect.appendChild(opt);
}});
yearSelect.value = state.year;

var metricSelect = document.getElementById('metric-select');
function rebuildMetricSelect() {{
  metricSelect.innerHTML = '';
  PAYLOAD.metric_order.forEach(function(m) {{
    if (!metricExistsForYear(m, state.year)) return;
    var opt = document.createElement('option');
    opt.value = m;
    opt.textContent = METRIC_LABELS[m] || m;
    metricSelect.appendChild(opt);
  }});
  // If current metric is missing for this year, pick the first available
  if (!metricExistsForYear(state.metric, state.year) && metricSelect.options.length > 0) {{
    state.metric = metricSelect.options[0].value;
  }}
  metricSelect.value = state.metric;
}}
rebuildMetricSelect();

var kSlider = document.getElementById('k-slider');
kSlider.min   = 0;
kSlider.max   = CLUSTER_KS.length - 1;
var defaultKIdx = CLUSTER_KS.indexOf(state.k);
kSlider.value = defaultKIdx >= 0 ? defaultKIdx : 0;
document.getElementById('k-display').textContent = state.k;

// ── view toggle ───────────────────────────────────────────────────────────────
function setView(viewName, subtitle) {{
  state.view = viewName;
  ['heatmap', 'mds', 'dendro'].forEach(function(v) {{
    var btn = document.getElementById('view-' + v);
    if (v === viewName) btn.classList.add('active');
    else btn.classList.remove('active');
  }});
  document.getElementById('view-subtitle').textContent = subtitle;
  render();
}}
document.getElementById('view-heatmap').addEventListener('click', function() {{
  setView('heatmap', 'Heatmap: reordered pairwise distance matrix. Click a row/col to pick a building.');
}});
document.getElementById('view-mds').addEventListener('click', function() {{
  setView('mds', '2D MDS projection of the distance matrix. Click a point to see its neighbours.');
}});
document.getElementById('view-dendro').addEventListener('click', function() {{
  setView('dendro', 'Truncated dendrogram: top 20 merges. Leaves with (N) represent merged sub-clusters.');
}});

// ── HEATMAP ───────────────────────────────────────────────────────────────────
function buildHeatmapFigure() {{
  var m = currentMetricData();
  if (!m) return {{data: [], layout: {{}}}};

  var orderedIds = m.ordered_ids;
  var z          = m.heatmap;
  var vmin       = m.colourscale.vmin;
  var vmax       = m.colourscale.vmax;

  var clusterOrigOrder = m.clusters[state.k] || m.clusters[CLUSTER_KS[0]];
  var clusterOrdered   = m.order.map(function(origIdx) {{ return clusterOrigOrder[origIdx]; }});

  var traces = [{{
    type: 'heatmap', z: z, x: orderedIds, y: orderedIds,
    xgap: 0, ygap: 0,
    colorscale: DIST_COLORSCALE, zmin: vmin, zmax: vmax,
    hovertemplate: '<b>%{{y}}</b> ↔ <b>%{{x}}</b><br>distance = %{{z:.3f}}<extra></extra>',
    colorbar: {{title: 'distance', thickness: 12, len: 0.6}},
  }}];

  var stripValues = orderedIds.map(function(bid, i) {{
    var info = INFO[bid] || {{}};
    if (state.colour === 'cluster') return 'Cluster ' + clusterOrdered[i];
    return info[state.colour] || 'n/a';
  }});
  var stripText = orderedIds.map(function(bid, i) {{
    return stripValues[i] + ' · ' + ((INFO[bid] || {{}}).label || bid);
  }});
  var domain = Array.from(new Set(stripValues)).sort();
  var stripColours = stripValues.map(function(v) {{ return colourFor(v, domain); }});

  traces.push({{
    type: 'scatter', mode: 'markers',
    x: orderedIds, y: Array(orderedIds.length).fill(0),
    xaxis: 'x', yaxis: 'y2',
    marker: {{color: stripColours, size: 9, symbol: 'square'}},
    hovertemplate: '%{{text}}<extra></extra>', text: stripText, showlegend: false,
  }});
  traces.push({{
    type: 'scatter', mode: 'markers',
    x: Array(orderedIds.length).fill(0), y: orderedIds,
    xaxis: 'x2', yaxis: 'y',
    marker: {{color: stripColours, size: 9, symbol: 'square'}},
    hovertemplate: '%{{text}}<extra></extra>', text: stripText, showlegend: false,
  }});

  domain.forEach(function(v) {{
    traces.push({{
      type: 'scatter', mode: 'markers',
      x: [null], y: [null],
      marker: {{color: colourFor(v, domain), size: 10, symbol: 'square'}},
      name: String(v), showlegend: true,
    }});
  }});

  var shapes = [];
  function addHighlight(bid) {{
    var idx = orderedIds.indexOf(bid);
    if (idx < 0) return;
    shapes.push({{type: 'rect', xref: 'x', yref: 'y',
                 x0: idx - 0.5, x1: idx + 0.5, y0: -0.5, y1: orderedIds.length - 0.5,
                 fillcolor: 'rgba(0, 0, 0, 0.12)', line: {{width: 0}}, layer: 'above'}});
    shapes.push({{type: 'rect', xref: 'x', yref: 'y',
                 x0: -0.5, x1: orderedIds.length - 0.5, y0: idx - 0.5, y1: idx + 0.5,
                 fillcolor: 'rgba(0, 0, 0, 0.12)', line: {{width: 0}}, layer: 'above'}});
  }}
  if (state.selectedA) addHighlight(state.selectedA);
  if (state.selectedB) addHighlight(state.selectedB);

  return {{
    data: traces,
    layout: {{
      margin: {{l: 70, r: 20, t: 40, b: 50}},
      paper_bgcolor: '#fafafa', plot_bgcolor: 'white',
      xaxis: {{domain: [0.025, 1.0], showticklabels: false, ticks: '',
               showgrid: false, zeroline: false}},
      yaxis: {{domain: [0.0, 0.975], showticklabels: false, ticks: '',
               autorange: 'reversed', showgrid: false, zeroline: false,
               scaleanchor: 'x', scaleratio: 1}},
      xaxis2: {{domain: [0.0, 0.02], showticklabels: false, ticks: '',
                showgrid: false, zeroline: false, range: [-0.5, 0.5], fixedrange: true}},
      yaxis2: {{domain: [0.98, 1.0], showticklabels: false, ticks: '',
                showgrid: false, zeroline: false, range: [-0.5, 0.5], fixedrange: true}},
      shapes: shapes, showlegend: true,
      legend: {{font: {{size: 10}}, x: 1.08, y: 1, xanchor: 'left'}},
      hovermode: 'closest',
    }},
    config: {{displaylogo: false, responsive: true,
             modeBarButtonsToRemove: ['lasso2d', 'select2d']}},
  }};
}}

// ── MDS ───────────────────────────────────────────────────────────────────────
function buildMdsFigure() {{
  var m = currentMetricData();
  if (!m) return {{data: [], layout: {{}}}};

  var bids = m.building_ids;
  var emb  = m.embedding;
  var clusterLabels = m.clusters[state.k] || m.clusters[CLUSTER_KS[0]];

  var series = {{}};
  bids.forEach(function(bid, i) {{
    var info = INFO[bid] || {{}};
    var value;
    if (state.colour === 'cluster') value = 'Cluster ' + clusterLabels[i];
    else value = info[state.colour] || 'n/a';

    if (!series[value]) series[value] = {{x: [], y: [], text: [], ids: []}};
    series[value].x.push(emb[i][0]);
    series[value].y.push(emb[i][1]);
    series[value].ids.push(bid);
    series[value].text.push(
      '<b>' + info.label + '</b><br>id: ' + bid +
      '<br>function: ' + (info['function'] || 'n/a') +
      '<br>era: ' + (info['era'] || 'n/a') +
      (info['gia_m2'] ? ('<br>GIA: ' + info['gia_m2'].toFixed(0) + ' m²') : '')
    );
  }});

  var domain = Object.keys(series).sort();
  var traces = domain.map(function(v) {{
    var s = series[v];
    return {{
      type: 'scatter', mode: 'markers',
      x: s.x, y: s.y, text: s.text, customdata: s.ids,
      hovertemplate: '%{{text}}<extra></extra>',
      marker: {{color: colourFor(v, domain), size: 9, line: {{color: 'white', width: 0.8}}}},
      name: v,
    }};
  }});

  if (state.selected) {{
    var selIdx = bids.indexOf(state.selected);
    if (selIdx >= 0) {{
      traces.push({{
        type: 'scatter', mode: 'markers',
        x: [emb[selIdx][0]], y: [emb[selIdx][1]],
        marker: {{color: 'black', size: 18, symbol: 'circle-open', line: {{width: 3}}}},
        hoverinfo: 'skip', showlegend: false,
      }});
      var neighbours = m.top5[selIdx];
      var nx = [], ny = [], ntext = [];
      neighbours.forEach(function(n) {{
        var idx = bids.indexOf(n.id);
        if (idx < 0) return;
        nx.push(emb[idx][0]);
        ny.push(emb[idx][1]);
        ntext.push(n.id + '  d=' + n.dist.toFixed(3));
      }});
      traces.push({{
        type: 'scatter', mode: 'markers',
        x: nx, y: ny,
        marker: {{color: 'rgba(230, 30, 30, 0.9)', size: 14, symbol: 'diamond-open', line: {{width: 2}}}},
        text: ntext, hovertemplate: '<b>Neighbour</b><br>%{{text}}<extra></extra>',
        showlegend: false,
      }});
    }}
  }}

  return {{
    data: traces,
    layout: {{
      margin: {{l: 40, r: 20, t: 20, b: 40}},
      paper_bgcolor: '#fafafa', plot_bgcolor: 'white',
      xaxis: {{title: 'MDS-1', gridcolor: '#eee', zeroline: false}},
      yaxis: {{title: 'MDS-2', gridcolor: '#eee', zeroline: false, scaleanchor: 'x', scaleratio: 1}},
      showlegend: true,
      legend: {{font: {{size: 11}}, x: 1.02, y: 1, xanchor: 'left'}},
      hovermode: 'closest',
    }},
    config: {{displaylogo: false, responsive: true,
             modeBarButtonsToRemove: ['lasso2d', 'select2d']}}
  }};
}}

// ── DENDROGRAM ────────────────────────────────────────────────────────────────
function buildDendroFigure() {{
  var m = currentMetricData();
  if (!m || !m.dendrogram) return {{data: [], layout: {{}}}};

  var dend = m.dendrogram;
  var traces = [];

  // Each branch is a U shape: 4 points forming two vertical lines and a horizontal top
  dend.segments.forEach(function(seg) {{
    traces.push({{
      type: 'scatter', mode: 'lines',
      x: seg.x, y: seg.y,
      line: {{color: '#3388ff', width: 1.5}},
      hoverinfo: 'skip', showlegend: false,
    }});
  }});

  // Leaf labels at the bottom — tick positions at 5, 15, 25, 35, ... matching dendrogram convention
  var leafX = dend.leaf_labels.map(function(_, i) {{ return 5 + 10 * i; }});
  var leafLabels = dend.leaf_labels;

  return {{
    data: traces,
    layout: {{
      margin: {{l: 60, r: 20, t: 40, b: 80}},
      paper_bgcolor: '#fafafa', plot_bgcolor: 'white',
      xaxis: {{
        title: 'Sub-clusters (leaf counts shown)',
        tickvals: leafX,
        ticktext: leafLabels,
        tickfont: {{size: 10}},
        tickangle: -45,
        showgrid: false, zeroline: false,
      }},
      yaxis: {{
        title: 'Merge distance',
        gridcolor: '#eee', zeroline: false,
        rangemode: 'tozero',
      }},
      showlegend: false,
      hovermode: 'closest',
    }},
    config: {{displaylogo: false, responsive: true,
             modeBarButtonsToRemove: ['lasso2d', 'select2d']}}
  }};
}}

// ── PROFILE CHART (used by side-panel details) ────────────────────────────────
function renderProfileChart(bids, containerId, availableFuels) {{
  if (!availableFuels || availableFuels.length === 0) return;
  if (!availableFuels.includes(profileState.fuel)) {{
    profileState.fuel = availableFuels[0];
  }}

  var ctrlHtml =
    '<div style="display:flex; gap:6px; align-items:center; margin:8px 0 4px; font-size:11px; flex-wrap:wrap;">' +
    '<label style="font-weight:bold; color:#555;">Slice:</label>';

  if (availableFuels.length > 1) {{
    ctrlHtml += '<select id="prof-fuel-sel" style="padding:2px 4px; font-size:11px;">';
    availableFuels.forEach(function(f) {{
      ctrlHtml += '<option value="' + f + '"' + (f === profileState.fuel ? ' selected' : '') + '>' + f + '</option>';
    }});
    ctrlHtml += '</select>';
  }} else {{
    ctrlHtml += '<span style="color:#666;">' + availableFuels[0] + '</span>';
  }}
  ['winter', 'summer', 'all', 'shoulder'].forEach(function() {{ }});

  var seasons = ['winter', 'summer', 'all', 'shoulder'];
  ctrlHtml += '<select id="prof-season-sel" style="padding:2px 4px; font-size:11px;">';
  seasons.forEach(function(s) {{
    ctrlHtml += '<option value="' + s + '"' + (s === profileState.season ? ' selected' : '') + '>' + s + '</option>';
  }});
  ctrlHtml += '</select>';

  var daytypes = ['weekday', 'weekend', 'all'];
  ctrlHtml += '<select id="prof-daytype-sel" style="padding:2px 4px; font-size:11px;">';
  daytypes.forEach(function(d) {{
    ctrlHtml += '<option value="' + d + '"' + (d === profileState.daytype ? ' selected' : '') + '>' + d + '</option>';
  }});
  ctrlHtml += '</select>';
  ctrlHtml += '</div>';
  ctrlHtml += '<div id="' + containerId + '" style="width:100%; height:200px;"></div>';

  var placeholder = document.getElementById('profile-panel');
  if (placeholder) placeholder.innerHTML = ctrlHtml;

  function draw() {{
    var hours = [];
    for (var h = 0; h < 24; h++) hours.push(h);
    var traces = bids.map(function(bid, i) {{
      var prof = profileFor(bid, profileState.fuel, profileState.season, profileState.daytype);
      return {{
        type: 'scatter', mode: 'lines',
        x: hours, y: prof || Array(24).fill(null),
        name: bid,
        line: {{width: i === 0 ? 2.2 : 1.4}},
        hovertemplate: '<b>' + bid + '</b><br>%{{x}}:00 → %{{y:.4f}}<extra></extra>',
      }};
    }});
    Plotly.newPlot(containerId, traces, {{
      margin: {{l: 40, r: 10, t: 10, b: 30}},
      paper_bgcolor: 'white', plot_bgcolor: 'white',
      xaxis: {{tickvals: [0,6,12,18,24], ticktext: ['00','06','12','18','24'],
              tickfont: {{size: 10}}, gridcolor: '#eee'}},
      yaxis: {{title: 'sum-to-1', tickfont: {{size: 10}}, gridcolor: '#eee'}},
      legend: {{font: {{size: 9}}, orientation: 'h', y: -0.25}},
      showlegend: true,
    }}, {{displaylogo: false, responsive: true, displayModeBar: false}});
  }}
  draw();

  var fsel = document.getElementById('prof-fuel-sel');
  if (fsel) fsel.addEventListener('change', function() {{ profileState.fuel = this.value; draw(); }});
  document.getElementById('prof-season-sel').addEventListener('change', function() {{
    profileState.season = this.value; draw();
  }});
  document.getElementById('prof-daytype-sel').addEventListener('change', function() {{
    profileState.daytype = this.value; draw();
  }});
}}

// ── SIDE PANEL: single building ───────────────────────────────────────────────
function renderSidePanelSingle(bid) {{
  var panel = document.getElementById('panel-content');
  var m = currentMetricData();
  var selIdx = m.building_ids.indexOf(bid);
  if (selIdx < 0) {{
    panel.innerHTML = '<div class="placeholder">Selected building not in this metric/year.</div>';
    return;
  }}
  var info = INFO[bid] || {{}};
  var neighbours = m.top5[selIdx];

  var html = '';
  html += '<h2>' + info.label + '</h2>';
  html += '<div class="meta">';
  html += '<span><b>id:</b> ' + bid + '</span>';
  html += '<span><b>fn:</b> ' + (info['function'] || 'n/a') + '</span>';
  html += '<span><b>era:</b> ' + (info['era'] || 'n/a') + '</span>';
  if (info.gia_m2) html += '<span><b>GIA:</b> ' + info.gia_m2.toFixed(0) + ' m²</span>';
  html += '</div>';

  html += '<h3>Nearest neighbours</h3>';
  html += '<table id="neighbour-table">';
  html += '<thead><tr><th>id</th><th>label</th><th>fn</th><th>dist</th></tr></thead><tbody>';
  neighbours.forEach(function(n) {{
    var nInfo = INFO[n.id] || {{label: n.id, 'function': 'n/a'}};
    html += '<tr><td>' + n.id + '</td><td>' + nInfo.label +
            '</td><td>' + (nInfo['function'] || 'n/a') + '</td>' +
            '<td>' + n.dist.toFixed(3) + '</td></tr>';
  }});
  html += '</tbody></table>';

  html += '<h3>Metric comparison</h3>';
  html += '<div id="comparison-plot"></div>';
  html += '<h3>Daily load profiles</h3>';
  html += '<div id="profile-panel"></div>';
  panel.innerHTML = html;

  var buildings = [{{id: bid}}].concat(neighbours.map(function(n) {{ return {{id: n.id}}; }}));
  var traces = CMP_KEYS.map(function(key) {{
    return {{
      type: 'bar',
      x: buildings.map(function(b) {{ return b.id; }}),
      y: buildings.map(function(b) {{
        var v = buildingInfoMetrics(b.id)[key];
        return (v === null || v === undefined) ? null : v;
      }}),
      name: CMP_LABELS[key],
      hovertemplate: '<b>' + CMP_LABELS[key] + '</b><br>%{{x}}: %{{y:.2f}}<extra></extra>',
    }};
  }});
  Plotly.newPlot('comparison-plot', traces, {{
    barmode: 'group',
    margin: {{l: 40, r: 10, t: 10, b: 40}},
    paper_bgcolor: 'white', plot_bgcolor: 'white',
    xaxis: {{tickfont: {{size: 10}}}},
    yaxis: {{tickfont: {{size: 10}}, gridcolor: '#eee'}},
    legend: {{font: {{size: 9}}, orientation: 'h', y: -0.25}},
    showlegend: true,
  }}, {{displaylogo: false, responsive: true, displayModeBar: false}});

  var profileBids = [bid].concat(neighbours.map(function(n) {{ return n.id; }}));
  var availFuels = [];
  ['elec', 'gas'].forEach(function(f) {{
    if (profileBids.some(function(b) {{ return hasFuel(b, f); }})) availFuels.push(f);
  }});
  if (profileState.fuel === null) profileState.fuel = defaultFuelFor(state.metric, profileBids);
  renderProfileChart(profileBids, 'profile-plot', availFuels);
}}

// ── SIDE PANEL: pair ──────────────────────────────────────────────────────────
function renderSidePanelPair(bidA, bidB) {{
  var panel = document.getElementById('panel-content');
  var m = currentMetricData();
  var idxA = m.building_ids.indexOf(bidA);
  var idxB = m.building_ids.indexOf(bidB);
  if (idxA < 0 || idxB < 0) {{
    renderSidePanelSingle(bidA);
    return;
  }}

  var orderMap = {{}};
  m.order.forEach(function(origIdx, ordIdx) {{ orderMap[origIdx] = ordIdx; }});
  var dist = m.heatmap[orderMap[idxA]][orderMap[idxB]];

  var infoA = INFO[bidA] || {{}};
  var infoB = INFO[bidB] || {{}};

  var html = '';
  html += '<h2>Pair: ' + bidA + ' ↔ ' + bidB + '</h2>';
  html += '<div class="meta"><span><b>distance:</b> ' + dist.toFixed(3) + '</span></div>';
  html += '<table id="neighbour-table">';
  html += '<thead><tr><th></th><th>' + bidA + '</th><th>' + bidB + '</th></tr></thead><tbody>';
  html += '<tr><td><b>name</b></td><td>' + infoA.label + '</td><td>' + infoB.label + '</td></tr>';
  html += '<tr><td><b>function</b></td><td>' + (infoA['function'] || 'n/a') + '</td><td>' + (infoB['function'] || 'n/a') + '</td></tr>';
  html += '<tr><td><b>era</b></td><td>' + (infoA['era'] || 'n/a') + '</td><td>' + (infoB['era'] || 'n/a') + '</td></tr>';
  if (infoA.gia_m2 || infoB.gia_m2) {{
    html += '<tr><td><b>GIA</b></td><td>' +
            (infoA.gia_m2 ? infoA.gia_m2.toFixed(0) : 'n/a') + '</td><td>' +
            (infoB.gia_m2 ? infoB.gia_m2.toFixed(0) : 'n/a') + '</td></tr>';
  }}
  html += '</tbody></table>';

  html += '<h3>Metric comparison</h3>';
  html += '<div id="comparison-plot"></div>';
  html += '<h3>Daily load profiles</h3>';
  html += '<div id="profile-panel"></div>';
  panel.innerHTML = html;

  var traces = CMP_KEYS.map(function(key) {{
    return {{
      type: 'bar',
      x: [bidA, bidB],
      y: [
        buildingInfoMetrics(bidA)[key] || null,
        buildingInfoMetrics(bidB)[key] || null,
      ],
      name: CMP_LABELS[key],
      hovertemplate: '<b>' + CMP_LABELS[key] + '</b><br>%{{x}}: %{{y:.2f}}<extra></extra>',
    }};
  }});
  Plotly.newPlot('comparison-plot', traces, {{
    barmode: 'group',
    margin: {{l: 40, r: 10, t: 10, b: 40}},
    paper_bgcolor: 'white', plot_bgcolor: 'white',
    xaxis: {{tickfont: {{size: 10}}}},
    yaxis: {{tickfont: {{size: 10}}, gridcolor: '#eee'}},
    legend: {{font: {{size: 9}}, orientation: 'h', y: -0.25}},
    showlegend: true,
  }}, {{displaylogo: false, responsive: true, displayModeBar: false}});

  var availFuels = [];
  ['elec', 'gas'].forEach(function(f) {{
    if (hasFuel(bidA, f) || hasFuel(bidB, f)) availFuels.push(f);
  }});
  if (profileState.fuel === null) profileState.fuel = defaultFuelFor(state.metric, [bidA, bidB]);
  renderProfileChart([bidA, bidB], 'profile-plot', availFuels);
}}

// ── SIDE PANEL: cluster contents ──────────────────────────────────────────────
function renderClusterContents() {{
  var panel = document.getElementById('panel-content');
  var m = currentMetricData();
  if (!m) {{ panel.innerHTML = '<div class="placeholder">No data for this metric/year.</div>'; return; }}
  var clusterLabels = m.clusters[state.k] || m.clusters[CLUSTER_KS[0]];
  var bids = m.building_ids;

  var byCluster = {{}};
  bids.forEach(function(bid, i) {{
    var c = clusterLabels[i];
    if (!byCluster[c]) byCluster[c] = [];
    byCluster[c].push(bid);
  }});

  var clusterIds = Object.keys(byCluster).map(Number).sort(function(a, b) {{ return a - b; }});
  var domainLabels = clusterIds.map(function(c) {{ return 'Cluster ' + c; }});

  function summariseCluster(members) {{
    var euiVals = [], funcs = {{}}, eras = {{}};
    members.forEach(function(bid) {{
      var info = INFO[bid] || {{}};
      var eui = buildingInfoMetrics(bid).eui_combined;
      if (eui !== null && eui !== undefined && !isNaN(eui)) euiVals.push(eui);
      var fn  = info['function'] || 'n/a';
      var era = info['era']      || 'n/a';
      funcs[fn] = (funcs[fn] || 0) + 1;
      eras[era] = (eras[era] || 0) + 1;
    }});
    function dominant(counts) {{
      var best = null, bestN = 0, total = 0;
      Object.keys(counts).forEach(function(k) {{
        total += counts[k];
        if (counts[k] > bestN) {{ best = k; bestN = counts[k]; }}
      }});
      return {{label: best, pct: total > 0 ? Math.round(100 * bestN / total) : 0}};
    }}
    var avgEui = euiVals.length > 0 ? euiVals.reduce(function(a,b){{return a+b;}}, 0) / euiVals.length : null;
    return {{count: members.length, avgEui: avgEui, domFn: dominant(funcs), domEra: dominant(eras)}};
  }}

  var html = '';
  html += '<div style="display:flex; align-items:center; justify-content:space-between;">';
  html += '<h2 style="margin:0;">Cluster contents (K=' + state.k + ')</h2>';
  html += '<button id="cluster-export-btn" style="padding:4px 10px; font-size:11px; background:#3388ff; color:white; border:none; border-radius:3px; cursor:pointer;">⬇ CSV</button>';
  html += '</div>';
  html += '<div class="meta" style="margin-top:4px;"><span>' + clusterIds.length + ' clusters across ' + bids.length + ' buildings</span></div>';
  html += '<div style="font-size:11px; color:#888; margin-bottom:10px;">Click a cluster header to expand/collapse.</div>';

  clusterIds.forEach(function(cid) {{
    var members = byCluster[cid];
    var swatch  = colourFor('Cluster ' + cid, domainLabels);
    var s       = summariseCluster(members);

    var summaryLine = s.count + ' buildings';
    if (s.avgEui !== null) summaryLine += ' · avg EUI ' + s.avgEui.toFixed(0);
    if (s.domFn.label && s.domFn.label !== 'n/a') {{
      summaryLine += ' · mostly ' + s.domFn.label + ' (' + s.domFn.pct + '%)';
    }}
    if (s.domEra.label && s.domEra.label !== 'n/a' && s.domEra.pct >= 40) {{
      summaryLine += ' · ' + s.domEra.label;
    }}

    html += '<div class="cluster-block" data-cid="' + cid + '" style="margin-bottom: 10px; border: 1px solid #eee; border-radius: 4px; overflow: hidden;">';
    html += '<div class="cluster-header" style="padding:8px 10px; background:#fafafa; cursor:pointer; display:flex; align-items:center; gap:8px;" data-cid="' + cid + '">';
    html += '<span class="cluster-caret" style="color:#888; font-size:10px; width:10px;">▶</span>';
    html += '<span style="display:inline-block; width:12px; height:12px; background:' + swatch + '; border-radius:2px; flex-shrink:0;"></span>';
    html += '<div style="flex:1;">';
    html += '<div style="font-weight:bold; font-size:13px; color:#222;">Cluster ' + cid + '</div>';
    html += '<div style="font-size:11px; color:#666;">' + summaryLine + '</div>';
    html += '</div></div>';

    html += '<div class="cluster-body" data-cid="' + cid + '" style="display:none;">';
    html += '<table id="neighbour-table" style="margin:0;">';
    html += '<thead><tr><th>id</th><th>label</th><th>fn</th><th>era</th></tr></thead><tbody>';
    members.forEach(function(bid) {{
      var info = INFO[bid] || {{}};
      html += '<tr>';
      html += '<td><a href="#" data-bid="' + bid + '" class="cluster-member-link" style="color:#3388ff; text-decoration:none;">' + bid + '</a></td>';
      html += '<td>' + (info.label || bid) + '</td>';
      html += '<td>' + (info['function'] || 'n/a') + '</td>';
      html += '<td>' + (info['era'] || 'n/a') + '</td>';
      html += '</tr>';
    }});
    html += '</tbody></table></div></div>';
  }});

  panel.innerHTML = html;

  document.querySelectorAll('.cluster-header').forEach(function(hdr) {{
    hdr.addEventListener('click', function() {{
      var cid = hdr.getAttribute('data-cid');
      var body  = document.querySelector('.cluster-body[data-cid="' + cid + '"]');
      var caret = hdr.querySelector('.cluster-caret');
      if (!body) return;
      var isHidden = body.style.display === 'none';
      body.style.display = isHidden ? 'block' : 'none';
      caret.textContent  = isHidden ? '▼' : '▶';
    }});
  }});
  document.querySelectorAll('.cluster-member-link').forEach(function(a) {{
    a.addEventListener('click', function(e) {{
      e.preventDefault();
      selectBuilding(a.getAttribute('data-bid'));
    }});
  }});
  document.getElementById('cluster-export-btn').addEventListener('click', function() {{
    exportClusterCsv(byCluster, clusterIds);
  }});
}}

function exportClusterCsv(byCluster, clusterIds) {{
  var headers = ['building_id', 'cluster_id', 'name', 'function', 'era', 'gia_m2', 'eui_combined'];
  var rows = [headers];
  clusterIds.forEach(function(cid) {{
    byCluster[cid].forEach(function(bid) {{
      var info = INFO[bid] || {{}};
      var metrics = buildingInfoMetrics(bid);
      rows.push([
        bid, cid, info.label || '',
        info['function'] || '', info['era'] || '',
        (info.gia_m2 !== null && info.gia_m2 !== undefined) ? info.gia_m2 : '',
        (metrics.eui_combined !== null && metrics.eui_combined !== undefined) ? metrics.eui_combined : '',
      ]);
    }});
  }});
  function csvEscape(v) {{
    var s = String(v);
    if (s.indexOf(',') >= 0 || s.indexOf('"') >= 0 || s.indexOf('\\n') >= 0) {{
      return '"' + s.replace(/"/g, '""') + '"';
    }}
    return s;
  }}
  var csv = rows.map(function(r) {{ return r.map(csvEscape).join(','); }}).join('\\n');
  var fname = 'clusters_' + state.metric + '_y' + state.year + '_k' + state.k + '.csv';
  var blob = new Blob([csv], {{type: 'text/csv;charset=utf-8;'}});
  var url  = URL.createObjectURL(blob);
  var a    = document.createElement('a');
  a.href   = url; a.download = fname;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}}

// ── SIDE PANEL: outlier table ─────────────────────────────────────────────────
function renderOutlierTable() {{
  var panel = document.getElementById('panel-content');
  var m = currentMetricData();
  if (!m || !m.outlier_scores) {{
    panel.innerHTML = '<div class="placeholder">No outlier data for this metric/year.</div>';
    return;
  }}

  var bids   = m.building_ids;
  var scores = m.outlier_scores;

  var indexed = bids.map(function(bid, i) {{ return {{bid: bid, score: scores[i]}}; }});
  indexed.sort(function(a, b) {{ return b.score - a.score; }});
  var top = indexed.slice(0, 10);

  // Mini bar chart alongside table — top 10 outlier scores
  var html = '';
  html += '<h2>Most outlying buildings</h2>';
  html += '<div class="meta"><span>' + METRIC_LABELS[state.metric] + ' · ' + state.year + '</span></div>';
  html += '<div style="font-size:11px; color:#888; margin-bottom:8px;">Score = mean distance to all other buildings. Higher = more distinctive.</div>';
  html += '<div id="outlier-chart" style="width:100%; height:180px;"></div>';
  html += '<table id="neighbour-table" style="margin-top:10px;">';
  html += '<thead><tr><th>rank</th><th>id</th><th>label</th><th>fn</th><th>score</th></tr></thead><tbody>';
  top.forEach(function(entry, i) {{
    var info = INFO[entry.bid] || {{}};
    html += '<tr>';
    html += '<td>' + (i + 1) + '</td>';
    html += '<td><a href="#" data-bid="' + entry.bid + '" class="outlier-link" style="color:#3388ff; text-decoration:none;">' + entry.bid + '</a></td>';
    html += '<td>' + (info.label || entry.bid) + '</td>';
    html += '<td>' + (info['function'] || 'n/a') + '</td>';
    html += '<td>' + entry.score.toFixed(3) + '</td>';
    html += '</tr>';
  }});
  html += '</tbody></table>';
  html += '<div style="font-size:11px; color:#888; margin-top:10px;">Pick a different metric or year to see outliers under that view.</div>';

  panel.innerHTML = html;

  // Horizontal bar chart
  Plotly.newPlot('outlier-chart', [{{
    type: 'bar', orientation: 'h',
    x: top.map(function(e) {{ return e.score; }}).reverse(),
    y: top.map(function(e) {{ return e.bid; }}).reverse(),
    marker: {{color: '#e31a1c'}},
    hovertemplate: '<b>%{{y}}</b><br>score = %{{x:.3f}}<extra></extra>',
  }}], {{
    margin: {{l: 60, r: 10, t: 5, b: 30}},
    paper_bgcolor: 'white', plot_bgcolor: 'white',
    xaxis: {{tickfont: {{size: 10}}, gridcolor: '#eee'}},
    yaxis: {{tickfont: {{size: 10}}, automargin: true}},
  }}, {{displaylogo: false, responsive: true, displayModeBar: false}});

  document.querySelectorAll('.outlier-link').forEach(function(a) {{
    a.addEventListener('click', function(e) {{
      e.preventDefault();
      selectBuilding(a.getAttribute('data-bid'));
    }});
  }});
}}

// ── SIDE PANEL ROUTER ─────────────────────────────────────────────────────────
function renderSidePanel() {{
  var panel = document.getElementById('panel-content');

  if (state.view === 'dendro') {{
    panel.innerHTML = '<div class="placeholder">Dendrogram view — no building selection. Switch to Heatmap or MDS to interact with individual buildings.</div>';
    return;
  }}

  if (state.colour === 'cluster' && !state.selected && !state.selectedA) {{
    renderClusterContents();
    return;
  }}

  if (state.view === 'mds') {{
    if (!state.selected) {{
      renderOutlierTable();
      return;
    }}
    renderSidePanelSingle(state.selected);
    return;
  }}

  if (state.selectedA && state.selectedB && state.selectedA !== state.selectedB) {{
    renderSidePanelPair(state.selectedA, state.selectedB);
    return;
  }}
  if (state.selectedA) {{
    renderSidePanelSingle(state.selectedA);
    return;
  }}

  // Nothing selected in heatmap, colour-by != cluster: show outliers
  renderOutlierTable();
}}

// ── shared select logic ───────────────────────────────────────────────────────
function selectBuilding(bid) {{
  if (state.view === 'heatmap') {{
    state.selectedA = bid;
    state.selectedB = null;
  }} else {{
    state.selected = bid;
  }}
  render();
}}

// ── master render ─────────────────────────────────────────────────────────────
function render() {{
  var fig;
  if (state.view === 'heatmap')   fig = buildHeatmapFigure();
  else if (state.view === 'mds')  fig = buildMdsFigure();
  else                            fig = buildDendroFigure();

  Plotly.react('plot', fig.data, fig.layout, fig.config);

  var m = currentMetricData();
  document.getElementById('n-buildings-display').textContent =
    m ? (m.n_buildings + ' buildings · ' + (METRIC_INFOS[state.metric] ? (METRIC_INFOS[state.metric].notes || '') : '')) : '';
  document.getElementById('k-control').style.opacity =
    state.colour === 'cluster' ? 1 : 0.4;

  renderSidePanel();
  setTimeout(attachClickHandler, 50);
}}

// ── events ────────────────────────────────────────────────────────────────────
yearSelect.addEventListener('change', function() {{
  state.year = parseInt(this.value);
  rebuildMetricSelect();
  state.selected = null; state.selectedA = null; state.selectedB = null;
  profileState.fuel = null;   // re-pick default for the new year
  render();
}});

metricSelect.addEventListener('change', function() {{
  state.metric = this.value;
  state.selected = null; state.selectedA = null; state.selectedB = null;
  profileState.fuel = null;
  render();
}});

document.getElementById('colour-select').addEventListener('change', function() {{
  state.colour = this.value;
  render();
}});

kSlider.addEventListener('input', function() {{
  state.k = CLUSTER_KS[parseInt(this.value)];
  document.getElementById('k-display').textContent = state.k;
  if (state.colour === 'cluster') render();
}});

// Building lookup
var lookupInput   = document.getElementById('lookup-input');
var lookupResults = document.getElementById('lookup-results');

function matchingBuildings(query) {{
  var q = query.trim().toLowerCase();
  if (!q) return [];
  var m = currentMetricData();
  if (!m) return [];
  var matches = [];
  m.building_ids.forEach(function(bid) {{
    var info = INFO[bid] || {{}};
    var idMatch    = bid.toLowerCase().includes(q);
    var labelMatch = (info.label || '').toLowerCase().includes(q);
    if (idMatch || labelMatch) matches.push({{bid: bid, label: info.label || bid}});
  }});
  return matches.slice(0, 10);
}}

lookupInput.addEventListener('input', function() {{
  var matches = matchingBuildings(this.value);
  if (matches.length === 0) {{
    lookupResults.style.display = 'none';
    return;
  }}
  lookupResults.innerHTML = matches.map(function(m) {{
    return '<div class="item" data-bid="' + m.bid + '"><b>' + m.bid + '</b> — ' + m.label + '</div>';
  }}).join('');
  lookupResults.style.display = 'block';
  lookupResults.querySelectorAll('.item').forEach(function(el) {{
    el.addEventListener('click', function() {{
      selectBuilding(el.getAttribute('data-bid'));
      lookupInput.value = '';
      lookupResults.style.display = 'none';
    }});
  }});
}});

lookupInput.addEventListener('blur', function() {{
  setTimeout(function() {{ lookupResults.style.display = 'none'; }}, 150);
}});

// Heatmap / MDS click handler
function attachClickHandler() {{
  var plotDiv = document.getElementById('plot');
  if (!plotDiv || !plotDiv.on) return;
  plotDiv.removeAllListeners && plotDiv.removeAllListeners('plotly_click');
  plotDiv.on('plotly_click', function(data) {{
    if (!data.points || data.points.length === 0) return;
    var pt = data.points[0];

    if (state.view === 'heatmap') {{
      if (pt.data && pt.data.type === 'heatmap') {{
        var xId = pt.x, yId = pt.y;
        if (xId === yId) {{
          state.selectedA = xId; state.selectedB = null;
        }} else {{
          state.selectedA = yId; state.selectedB = xId;
        }}
        render();
        return;
      }}
      if (pt.data && pt.data.type === 'scatter') {{
        var bid = (pt.xaxis === 'x2') ? pt.y : pt.x;
        if (bid) {{
          state.selectedA = bid; state.selectedB = null;
          render();
        }}
      }}
    }} else if (state.view === 'mds') {{
      var bid = pt.customdata;
      if (!bid) return;
      state.selected = (state.selected === bid) ? null : bid;
      render();
    }}
  }});
}}

window.addEventListener('resize', function() {{
  Plotly.Plots.resize(document.getElementById('plot'));
}});

// ── init ──────────────────────────────────────────────────────────────────────
render();

</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'\nSimilarity view saved to: {output_path}')


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print('Loading similarity cache...')
    cache = _load_or_compute_cache()

    print('\nLoading temporal + metadata + profiles for payload...')
    temporal = _load_temporal()
    meta     = _load_metadata()

    if not os.path.exists(PROFILES_CACHE):
        raise FileNotFoundError(
            f'{PROFILES_CACHE} not found. Run non_spatial_metrics.py first.'
        )
    profiles = pd.read_pickle(PROFILES_CACHE)

    print('\nBuilding payload...')
    payload = _build_payload(cache, temporal, meta, profiles)
    print(f'  {len(payload["years"])} years × {len(payload["metric_order"])} metrics')
    print(f'  {len(payload["building_info"])} buildings with tooltip info')

    print('\nRendering HTML...')
    _render_html(payload, OUTPUT_HTML)