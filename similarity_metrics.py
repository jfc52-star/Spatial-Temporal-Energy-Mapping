"""
similarity_metrics.py
---------------------
Pairwise distance matrices between buildings, computed across ALL years
of the dataset. Hierarchical clustering, 2D MDS embedding, and truncated
dendrogram per (year, metric).

Distances implemented:
  - Gower           : mixed continuous + categorical (uses metadata)
  - Euclidean       : normalised feature vector over operational metrics
  - Peak Timing     : circular single-feature distance (per fuel)
  - Load Profile    : Euclidean on concatenated seasonal/daytype shapes
  - KL              : KL divergence between sum-to-1 daily load profiles

Single output pickle `similarity_cache.pkl` with structure:
    {
      'years':    [2018, 2019, 2020, ...],
      'per_year': {
        year: {
          'distances':    {metric: ndarray},
          'building_ids': {metric: [bid, ...]},
          'feature_info': {metric: dict},
          'embeddings':   {metric: {'embedding': Nx2,
                                    'clusters': {k: labels},
                                    'dendrogram': dict}},
        },
      },
      'feature_info_static': {metric: notes},
    }
"""

import os
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance  import squareform
from sklearn.manifold        import MDS

from metricsV3           import compute_all_buildings_temporal, DATA_DIR
from non_spatial_metrics import _load_metadata_table, compute_all_profiles
from config              import CENTROID_CSV_PATH
# =============================================================================
# 1. CONFIGURATION
# =============================================================================

CACHE_PATH = 'similarity_cache.pkl'

EUCLIDEAN_FEATURES = [
    'eui_elec', 'eui_gas',
    'bi_elec',  'bi_gas',
    'pli_elec', 'pli_gas',
    'lf_elec',  'lf_gas',
    'fmr',
    'peak_hr_elec', 'peak_hr_gas',
]

GOWER_CONTINUOUS = [
    'eui_elec', 'eui_gas',
    'bi_elec',  'bi_gas',
    'pli_elec', 'pli_gas',
    'lf_elec',  'lf_gas',
    'fmr',
    'gia_m2',
    'date_built',
]
GOWER_CATEGORICAL = ['function', 'tenure', 'era']

PROFILE_SLICES = [
    ('winter', 'weekday'),
    ('winter', 'weekend'),
    ('summer', 'weekday'),
    ('summer', 'weekend'),
]

KL_EPSILON            = 1e-9
DENDROGRAM_TRUNCATE_P = 20

CENTROID_PATH = CENTROID_CSV_PATH


def _normalise_bid(bid) -> str:
    """Force any building ID to the canonical 'b<int>' form, idempotently."""
    s = str(bid).strip()
    if s.startswith('b'):
        return s
    # Strip leading zeros and a trailing '.0' from numeric strings
    try:
        return f'b{int(float(s))}'
    except (ValueError, TypeError):
        return f'b{s}'


def _normalise_meta_index(meta: pd.DataFrame) -> pd.DataFrame:
    """Ensure the metadata index is canonical 'b<int>' format."""
    meta = meta.copy()
    if meta.index.name != 'building_id':
        if 'building_id' in meta.columns:
            meta = meta.set_index('building_id')
    meta.index = [_normalise_bid(b) for b in meta.index]
    meta.index.name = 'building_id'
    return meta


def _load_centroids() -> pd.DataFrame:
    """Load building centroid coordinates, indexed by 'b<id>'."""
    if not os.path.exists(CENTROID_PATH):
        print(f'  WARNING: centroid file not found at {CENTROID_PATH} - '
              f'archetype_geo will run without geography.')
        return pd.DataFrame(columns=['lat', 'lon'])

    df = pd.read_csv(CENTROID_PATH)
    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ('latitude', 'lat'):              rename[c] = 'lat'
        elif lc in ('longitude', 'lon', 'long'):   rename[c] = 'lon'
        elif lc in ('building_id', 'bid', 'id'):   rename[c] = 'building_id'
    df = df.rename(columns=rename)

    df['building_id'] = df['building_id'].apply(_normalise_bid)
    return df.set_index('building_id')[['lat', 'lon']]


# =============================================================================
# 1b. ARCHETYPE ASSIGNMENT
# =============================================================================

ARCHETYPE_QUANTILES = {
    'high':      0.75,
    'low':       0.25,
    'very_high': 0.90,
}


def _q(series, q):
    if series is None:
        return np.nan
    s = pd.to_numeric(pd.Series(series), errors='coerce').dropna()
    return s.quantile(q) if len(s) else np.nan

def _robust_z_params(features: pd.DataFrame, columns: list) -> dict:
    """Compute estate-wide robust standardisation parameters (median, IQR)
    for each requested column, ignoring NaNs.

    Returns {col: (median, iqr)}. IQR is floored to a small positive value so
    that constant or near-constant features cannot produce divide-by-zero or
    explode the z-score. Median/IQR are used instead of mean/SD so the small
    number of extreme process outliers (e.g. b59, b62) do not distort the
    standardisation of every other building.
    """
    params = {}
    for c in columns:
        if c not in features.columns:
            params[c] = (np.nan, np.nan)
            continue
        s = pd.to_numeric(features[c], errors='coerce').dropna()
        if len(s) < 2:
            params[c] = (np.nan, np.nan)
            continue
        med = float(s.median())
        iqr = float(s.quantile(0.75) - s.quantile(0.25))
        # Floor IQR to avoid blow-ups on degenerate features
        iqr = iqr if iqr > 1e-9 else 1.0
        params[c] = (med, iqr)
    return params

def _control_limited_score(row, z):
    return np.nan

def assign_archetypes(features: pd.DataFrame,
                      return_scores: bool = False,
                      min_score: float = 0.0):
    """REPLACEMENT (was boolean-tag version). SECONDARY per-building diagnostic.

    Scores each building individually with the same fair z-score scheme as the
    primary cluster method `assign_archetypes_cluster`. Useful for flagging
    buildings whose standalone archetype disagrees with their cluster's label.
    NOT the method behind Table 2 — that is assign_archetypes_cluster.

    Expects (NaN-tolerant) columns: hs_slope, hs_balance, eui_gas, fmr,
    bi_elec, pli_elec, lf_elec, midday_fraction. pli_bi_ratio derived if absent.
    Control-limited is omitted (features not computed).
    """
    feat = features.copy()
    if 'pli_bi_ratio' not in feat.columns:
        bi  = pd.to_numeric(feat.get('bi_elec'),  errors='coerce')
        pli = pd.to_numeric(feat.get('pli_elec'), errors='coerce')
        feat['pli_bi_ratio'] = pli / bi.replace(0, np.nan)

    z_cols = ['hs_slope', 'hs_balance', 'eui_gas', 'fmr',
              'bi_elec', 'pli_elec', 'lf_elec',
              'pli_bi_ratio', 'midday_fraction']
    zp = _robust_z_params(feat, z_cols)

    def z(col, row):
        med, iqr = zp.get(col, (np.nan, np.nan))
        if pd.isna(med):
            return np.nan
        val = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors='coerce').iloc[0]
        if pd.isna(val):
            return np.nan
        return (val - med) / iqr

    def mean_terms(terms):
        vals = [t for t in terms if not pd.isna(t)]
        return float(np.mean(vals)) if vals else np.nan

    archetypes = ['Fabric-limited', 'Electrification-candidate',
                  'Process-dominated', 'Flexibility candidate', 'Solar-suitable']
    score_rows, labels_out = [], []
    for _, row in feat.iterrows():
        s = {
            'Fabric-limited': mean_terms([
                z('hs_slope', row), z('eui_gas', row), z('hs_balance', row)]),
            'Electrification-candidate': mean_terms([
                -z('hs_slope', row), -z('eui_gas', row), z('fmr', row)]),
            'Process-dominated': mean_terms([
                z('bi_elec', row), -z('fmr', row), -z('lf_elec', row)]),
            'Flexibility candidate': mean_terms([
                z('pli_bi_ratio', row), -z('lf_elec', row), z('midday_fraction', row)]),
            'Solar-suitable': mean_terms([
                z('midday_fraction', row), z('pli_elec', row), -z('bi_elec', row)]),
        }
        score_rows.append(s)
        valid = {k: v for k, v in s.items() if not pd.isna(v)}
        if not valid:
            labels_out.append('Unclassified')
        else:
            best = max(valid, key=valid.get)
            labels_out.append(best if valid[best] >= min_score else 'Unclassified')

    labels = pd.Series(labels_out, index=feat.index, name='archetype')
    if return_scores:
        return labels, pd.DataFrame(score_rows, index=feat.index)[archetypes]
    return labels

def assign_archetypes_cluster(features: pd.DataFrame,
                              cluster_labels: pd.Series,
                              return_scores: bool = False,
                              min_score: float = 0.0):
    """NEW. PRIMARY archetype method. Assigns ONE archetype per CLUSTER by
    scoring the cluster's MEDIAN feature vector, using robust z-scores so no
    archetype is favoured by a feature's native scale or skew.

    This is the method behind Table 2 in the dissertation. The per-building
    `assign_archetypes` is retained only as a secondary diagnostic.

    Parameters
    ----------
    features : DataFrame indexed by 'b<id>'. Must contain (NaN-tolerant; terms
        whose inputs are entirely missing are skipped):
            hs_slope, hs_balance, eui_gas, fmr,
            bi_elec, pli_elec, lf_elec, midday_fraction
        'pli_bi_ratio' is derived internally if absent.
    cluster_labels : Series indexed by 'b<id>', integer cluster IDs (from HAC).
    return_scores  : if True also return per-cluster score DataFrame.
    min_score      : minimum winning score to assign a label, else 'Unclassified'.

    Returns
    -------
    cluster_archetype : Series indexed by cluster ID -> archetype string
    building_archetype: Series indexed by 'b<id>' -> inherited archetype string
    scores            : DataFrame (cluster ID x archetype), only if return_scores
    """
    # Align features to the buildings that were actually clustered
    common = features.index.intersection(cluster_labels.index)
    feat = features.loc[common].copy()
    labs = cluster_labels.loc[common]

    # Derive PLI/BI ratio if not supplied
    if 'pli_bi_ratio' not in feat.columns:
        bi  = pd.to_numeric(feat.get('bi_elec'),  errors='coerce')
        pli = pd.to_numeric(feat.get('pli_elec'), errors='coerce')
        feat['pli_bi_ratio'] = pli / bi.replace(0, np.nan)

    # Per-cluster MEDIAN feature vectors
    feat = feat.assign(_cluster=labs)
    z_cols = ['hs_slope', 'hs_balance', 'eui_gas', 'fmr',
              'bi_elec', 'pli_elec', 'lf_elec',
              'pli_bi_ratio', 'midday_fraction']
    present = [c for c in z_cols if c in feat.columns]
    cluster_medians = (feat.groupby('_cluster')[present]
                           .median(numeric_only=True))

    # Standardise on the distribution of CLUSTER MEDIANS (so z=0 is the typical
    # cluster, and a cluster is scored relative to its peers). This is the
    # correct reference set for cluster-level assignment.
    zp = _robust_z_params(cluster_medians, present)

    def z(col, row):
        med, iqr = zp.get(col, (np.nan, np.nan))
        if pd.isna(med):
            return np.nan
        val = row.get(col, np.nan)
        if pd.isna(val):
            return np.nan
        return (val - med) / iqr

    def mean_terms(terms):
        vals = [t for t in terms if not pd.isna(t)]
        return float(np.mean(vals)) if vals else np.nan

    archetypes = ['Fabric-limited', 'Electrification-candidate',
                  'Process-dominated', 'Flexibility candidate', 'Solar-suitable']

    score_rows, labels_out = [], []
    for cid, row in cluster_medians.iterrows():
        s = {
            'Fabric-limited': mean_terms([
                z('hs_slope', row), z('eui_gas', row), z('hs_balance', row)]),

            'Electrification-candidate': mean_terms([
                -z('hs_slope', row), -z('eui_gas', row), z('fmr', row)]),

            'Process-dominated': mean_terms([
                z('bi_elec', row), -z('fmr', row), -z('lf_elec', row)]),

            'Flexibility candidate': mean_terms([
                z('pli_bi_ratio', row), -z('lf_elec', row), z('midday_fraction', row)]),

            'Solar-suitable': mean_terms([
                z('midday_fraction', row), z('pli_elec', row), -z('bi_elec', row)]),
        }
        # NOTE: Control-limited intentionally omitted — its weekend-gas and
        # pre-heating diagnostics are not computed. It remains in Table 1 for
        # completeness but is not scored here.
        score_rows.append(s)

        valid = {k: v for k, v in s.items() if not pd.isna(v)}
        if not valid:
            labels_out.append('Unclassified')
        else:
            best = max(valid, key=valid.get)
            labels_out.append(best if valid[best] >= min_score else 'Unclassified')

    cluster_archetype = pd.Series(labels_out, index=cluster_medians.index,
                                  name='archetype')
    building_archetype = labs.map(cluster_archetype.to_dict()).rename('archetype')

    if return_scores:
        scores = pd.DataFrame(score_rows, index=cluster_medians.index)[archetypes]
        return cluster_archetype, building_archetype, scores
    return cluster_archetype, building_archetype

# =============================================================================
# 2. FEATURE BUILDERS
# =============================================================================

def _build_operational_features(
    temporal_df: pd.DataFrame,
    meta: pd.DataFrame,
    year: int,
    features: list,
) -> pd.DataFrame:
    """
    Slice temporal_df to `year`, normalise building IDs to 'b<int>', keep
    requested features, and intersect with meta.index.

    Returns DataFrame indexed by canonical building_id.
    """
    year_df = temporal_df[temporal_df['year'] == year].copy()
    if len(year_df) == 0:
        return pd.DataFrame()

    # Idempotent ID normalisation — handles both 'b7' and 7
    year_df['building_id'] = year_df['building_id'].apply(_normalise_bid)

    keep = ['building_id'] + [f for f in features if f in year_df.columns]
    year_df = year_df[keep].drop_duplicates(subset='building_id').set_index('building_id')

    # Meta is also normalised so the intersection actually finds overlap
    meta_norm = _normalise_meta_index(meta)
    common = year_df.index.intersection(meta_norm.index)
    return year_df.loc[common]


def _build_gower_features(temporal_df, meta, year):
    """
    Build the joined operational + metadata feature table for Gower.

    Returns (combined_df, cont_cols, cat_cols).
    Defensive joins so that mismatched building-ID conventions don't
    silently produce an empty table.
    """
    meta_norm = _normalise_meta_index(meta)

    # 1) Continuous features that live in temporal_df but not in meta
    op_cols = [f for f in GOWER_CONTINUOUS if f not in meta_norm.columns]
    op = _build_operational_features(temporal_df, meta_norm, year, op_cols)

    if len(op) == 0:
        # Diagnostic: tell the caller why we got nothing
        print(f'    [gower] no operational rows for {year} '
              f'(temporal n={len(temporal_df[temporal_df["year"] == year])}, '
              f'meta n={len(meta_norm)})')
        return pd.DataFrame(), [], []

    # 2) Continuous + categorical features sourced from meta
    meta_cols_cont = [c for c in GOWER_CONTINUOUS  if c in meta_norm.columns]
    meta_cols_cat  = [c for c in GOWER_CATEGORICAL if c in meta_norm.columns]
    meta_keep = meta_cols_cont + meta_cols_cat

    # Only take meta rows whose IDs appear in op — protects against KeyError
    available = meta_norm.index.intersection(op.index)
    meta_sub = meta_norm.loc[available, meta_keep].copy() if meta_keep else \
               pd.DataFrame(index=available)

    combined = op.join(meta_sub, how='inner')

    cont_cols = [c for c in GOWER_CONTINUOUS  if c in combined.columns]
    cat_cols  = [c for c in GOWER_CATEGORICAL if c in combined.columns]

    return combined, cont_cols, cat_cols

def _build_midday_fraction(profiles_df, year, fuel='elec',
                           season='all', daytype='weekday'):
    """NEW. Fraction of daily demand falling in the 10:00-15:00 window, per
    building, for a given year/fuel/season/daytype slice.

    Uses 'profile_sum1' (24-value profile normalised to sum to 1), so the
    returned value is directly the share of the day's demand in hours 10-14
    inclusive (i.e. the 10:00-15:00 window). This is the M_z input defined in
    the dissertation. Returns a Series indexed by 'b<id>'.
    """
    sub = profiles_df[
        (profiles_df['fuel'] == fuel) &
        (profiles_df['year'] == year) &
        (profiles_df['season'] == season) &
        (profiles_df['daytype'] == daytype)
    ]
    out = {}
    for _, row in sub.iterrows():
        prof = row.get('profile_sum1')
        if prof is None or (isinstance(prof, float) and np.isnan(prof)):
            continue
        arr = np.asarray(prof, dtype=float)
        if arr.shape[0] != 24 or np.all(np.isnan(arr)):
            continue
        # hours 10,11,12,13,14 => the 10:00-15:00 window
        midday = float(np.nansum(arr[10:15]))
        out[f"b{int(row['building_id'])}"] = midday
    return pd.Series(out, name='midday_fraction', dtype=float)

def _build_profile_vectors(profiles_df, meta, year, fuel):
    """Build (n_buildings × 96)-shaped profile vectors for `year`/`fuel`."""
    meta_norm = _normalise_meta_index(meta)

    sub = profiles_df[
        (profiles_df['fuel'] == fuel) &
        (profiles_df['year'] == year)
    ].copy()

    rows = {}
    for bid, grp in sub.groupby('building_id'):
        segments = []
        complete = True
        for season, daytype in PROFILE_SLICES:
            match = grp[(grp['season'] == season) & (grp['daytype'] == daytype)]
            if len(match) == 0:
                complete = False
                break
            prof = match.iloc[0]['profile_sum1']
            if prof is None or (isinstance(prof, float) and np.isnan(prof)):
                complete = False
                break
            segments.append(np.array(prof, dtype=float))
        if not complete:
            continue
        key = _normalise_bid(bid)
        if key in meta_norm.index:
            rows[key] = np.concatenate(segments)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(rows, orient='index')


# =============================================================================
# 3. DISTANCE FUNCTIONS
# =============================================================================

def _gower_distance(features, cont_cols, cat_cols):
    n = len(features)
    matrix = np.zeros((n, n), dtype=float)
    if n < 2:
        return matrix

    cont_cols = [c for c in cont_cols
                 if pd.to_numeric(features[c], errors='coerce').notna().any()]

    ranges = {}
    for c in cont_cols:
        col = pd.to_numeric(features[c], errors='coerce')
        rng = float(col.max() - col.min()) if col.notna().sum() > 1 else 1.0
        ranges[c] = rng if rng > 0 else 1.0

    cont_data = (features[cont_cols].apply(pd.to_numeric, errors='coerce').values
                 if cont_cols else np.empty((n, 0)))
    cat_data  = (features[cat_cols].astype(str).values
                 if cat_cols else np.empty((n, 0)))
    range_vec = (np.array([ranges[c] for c in cont_cols])
                 if cont_cols else np.array([]))

    for i in range(n):
        cont_diff = (np.abs(cont_data - cont_data[i]) / range_vec
                     if len(range_vec) > 0 else np.empty((n, 0)))
        cont_valid = ~np.isnan(cont_diff)
        cont_diff  = np.where(cont_valid, cont_diff, 0.0)

        if cat_data.shape[1] > 0:
            cat_diff = (cat_data != cat_data[i]).astype(float)
            cat_valid = np.ones_like(cat_diff, dtype=bool)
            for k, c in enumerate(cat_cols):
                cat_valid[:, k] = (~pd.isna(features[c].values) &
                                   (features[c].astype(str).values != 'None'))
            cat_diff = np.where(cat_valid, cat_diff, 0.0)
        else:
            cat_diff  = np.empty((n, 0))
            cat_valid = np.empty((n, 0), dtype=bool)

        total_diff  = np.concatenate([cont_diff,  cat_diff ], axis=1)
        total_valid = np.concatenate([cont_valid, cat_valid], axis=1)
        n_valid = total_valid.sum(axis=1)
        matrix[i, :] = total_diff.sum(axis=1) / np.maximum(n_valid, 1)

    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0.0)
    return matrix

def _archetype_geo_distance(features, weights=None):
    if weights is None:
        weights = {'archetype': 0.5, 'geo': 0.3, 'gia': 0.1, 'era': 0.1}

    n = len(features)
    if n < 2:
        return np.zeros((n, n))

    def _col(name, default=np.nan, as_str=False):
        """Safely extract a column as 1-D numpy array; defaults if missing."""
        if name in features.columns:
            s = features[name]
            if as_str:
                # Force to plain ndarray of dtype=object — StringArray/ExtensionArray
                # don't support the [:, None] broadcasting trick used downstream.
                return np.asarray(s.astype(str).values, dtype=object)
            return np.asarray(pd.to_numeric(s, errors='coerce').values, dtype=float)
        if as_str:
            return np.asarray([str(default)] * n, dtype=object)
        return np.asarray([default] * n, dtype=float)

    arche = _col('archetype', default='unclassified', as_str=True)
    era   = _col('era',       default='?',            as_str=True)
    gia   = _col('gia_m2')
    lat   = _col('lat')
    lon   = _col('lon')

    # Categorical components: 0 if match, 1 if not
    arche_diff = (arche[:, None] != arche[None, :]).astype(float)
    era_diff   = (era[:, None]   != era[None, :]).astype(float)

    # GIA: log-scale, then range-normalise
    if np.isfinite(gia).sum() > 1:
        log_gia = np.log(np.clip(gia, 1.0, None))
        rng = np.nanmax(log_gia) - np.nanmin(log_gia)
        rng = rng if rng > 0 else 1.0
        gia_diff = np.abs(log_gia[:, None] - log_gia[None, :]) / rng
        gia_diff = np.nan_to_num(gia_diff, nan=0.0)
    else:
        gia_diff = np.zeros((n, n))

    # Geography: haversine km, normalised by 5 km
    if np.isfinite(lat).sum() > 1 and np.isfinite(lon).sum() > 1:
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        dlat = lat_r[:, None] - lat_r[None, :]
        dlon = lon_r[:, None] - lon_r[None, :]
        a = (np.sin(dlat / 2)**2 +
             np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2)**2)
        km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        geo_diff = np.clip(km / 5.0, 0.0, 1.0)
        geo_diff = np.nan_to_num(geo_diff, nan=1.0)
    else:
        geo_diff = np.zeros((n, n))

    D = (weights['archetype'] * arche_diff +
         weights['geo']       * geo_diff   +
         weights['gia']       * gia_diff   +
         weights['era']       * era_diff)

    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


def _euclidean_distance(features):
    X = features.apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all')
    if X.shape[1] == 0:
        n = len(features)
        return np.zeros((n, n))

    X = X.fillna(X.median(numeric_only=True))
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=0)
    sd = sd.replace(0, 1.0)
    Z = ((X - mu) / sd).values
    Z = np.nan_to_num(Z, nan=0.0)

    diff = Z[:, None, :] - Z[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _peak_timing_distance(feature_series):
    vals = pd.to_numeric(feature_series, errors='coerce').values
    n = len(vals)
    if n < 2:
        return np.zeros((n, n))
    diff = np.abs(vals[:, None] - vals[None, :])
    circular = np.minimum(diff, 24.0 - diff)
    circular[np.isnan(circular)] = 12.0
    np.fill_diagonal(circular, 0.0)
    return circular


def _profile_euclidean_distance(profile_features):
    if len(profile_features) == 0:
        return np.zeros((0, 0))
    X = profile_features.values
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _kl_divergence_symmetric(profile_features):
    if len(profile_features) == 0:
        return np.zeros((0, 0))
    X = profile_features.values
    n = len(X)

    n_slices = len(PROFILE_SLICES)
    assert X.shape[1] == 24 * n_slices, 'Unexpected profile vector length'
    chunks = [X[:, i*24:(i+1)*24] for i in range(n_slices)]

    total = np.zeros((n, n), dtype=float)
    for P in chunks:
        P_safe = np.clip(P, KL_EPSILON, None)
        P_safe = P_safe / P_safe.sum(axis=1, keepdims=True)
        log_P  = np.log(P_safe)

        A_log_A = (P_safe * log_P).sum(axis=1)
        A_log_B = P_safe @ log_P.T
        kl_ab   = A_log_A[:, None] - A_log_B
        total  += (kl_ab + kl_ab.T) / 2

    total = np.maximum(total, 0.0)
    np.fill_diagonal(total, 0.0)
    return total


# =============================================================================
# 4. PER-YEAR RUNNER
# =============================================================================

def compute_distances_for_year(temporal_df, profiles_df, meta, year, centroids=None):
    if centroids is None:
        centroids = _load_centroids()

    meta_norm = _normalise_meta_index(meta)

    out = {'distances': {}, 'building_ids': {}, 'feature_info': {}, 'archetypes': {}}

    op_feat = _build_operational_features(temporal_df, meta_norm, year, EUCLIDEAN_FEATURES)
    if len(op_feat) > 1:
        out['distances']['euclidean']    = _euclidean_distance(op_feat)
        out['building_ids']['euclidean'] = op_feat.index.tolist()
        out['feature_info']['euclidean'] = {
            'n_features': op_feat.shape[1],
            'notes': f'{op_feat.shape[1]}-dim feature vector, z-scored',
        }

    gower_feat, cont_cols, cat_cols = _build_gower_features(temporal_df, meta_norm, year)
    if len(gower_feat) > 1 and (len(cont_cols) + len(cat_cols)) > 0:
        out['distances']['gower']    = _gower_distance(gower_feat, cont_cols, cat_cols)
        out['building_ids']['gower'] = gower_feat.index.tolist()
        out['feature_info']['gower'] = {
            'n_continuous':  len(cont_cols),
            'n_categorical': len(cat_cols),
            'n_buildings':   len(gower_feat),
            'notes': f'{len(cont_cols)} continuous + {len(cat_cols)} categorical features',
        }

    # Archetype assignment + archetype-geo distance
    archetype_input_cols = [
        'hs', 'eui_gas', 'balance_point', 'fmr',
        'bi_elec', 'pli_elec', 'lf_elec', 'peak_hr_elec',
    ]
    op_for_arche = _build_operational_features(
        temporal_df, meta_norm, year,
        [c for c in archetype_input_cols if c not in meta_norm.columns],
    )
    extras_from_meta = [c for c in ['date_built', 'gia_m2', 'era']
                        if c in meta_norm.columns]
    arche_input = op_for_arche.copy()
    if extras_from_meta and len(op_for_arche) > 0:
        arche_input = arche_input.join(
            meta_norm.loc[op_for_arche.index, extras_from_meta], how='left'
        )

    if not centroids.empty and len(arche_input) > 0:
        arche_input = arche_input.join(centroids, how='left')

    if len(arche_input) > 1:
        arche_tags = assign_archetypes(arche_input)
        out['archetypes'][year] = arche_tags.to_dict()

        arche_feat = arche_input.copy()
        arche_feat['archetype'] = arche_tags
        out['distances']['archetype_geo']    = _archetype_geo_distance(arche_feat)
        out['building_ids']['archetype_geo'] = arche_feat.index.tolist()
        out['feature_info']['archetype_geo'] = {
            'notes': 'Weighted Gower over archetype tag, haversine geo, log GIA, era',
            'weights': {'archetype': 0.5, 'geo': 0.3, 'gia': 0.1, 'era': 0.1},
            'n_classified': int((arche_tags != 'unclassified').sum()),
            'n_total':      int(len(arche_tags)),
        }

    for fuel in ['elec', 'gas']:
        feature_col = f'peak_hr_{fuel}'
        op = _build_operational_features(temporal_df, meta_norm, year, [feature_col])
        if len(op) < 2 or op[feature_col].dropna().empty:
            continue
        key = f'peak_timing_{fuel}'
        out['distances'][key]    = _peak_timing_distance(op[feature_col])
        out['building_ids'][key] = op.index.tolist()
        out['feature_info'][key] = {
            'n_features': 1,
            'notes': f'Circular distance on {feature_col}, range 0-12 hours',
        }

    for fuel in ['elec', 'gas']:
        prof_feat = _build_profile_vectors(profiles_df, meta_norm, year, fuel)
        if len(prof_feat) > 1:
            key_euc = f'profile_euclidean_{fuel}'
            key_kl  = f'profile_kl_{fuel}'

            out['distances'][key_euc]    = _profile_euclidean_distance(prof_feat)
            out['building_ids'][key_euc] = prof_feat.index.tolist()
            out['feature_info'][key_euc] = {
                'n_features': prof_feat.shape[1],
                'notes': f'{prof_feat.shape[1]}-dim concatenated profile (4 slices × 24h)',
            }

            out['distances'][key_kl]    = _kl_divergence_symmetric(prof_feat)
            out['building_ids'][key_kl] = prof_feat.index.tolist()
            out['feature_info'][key_kl] = {
                'n_features': prof_feat.shape[1],
                'notes': 'Symmetrised KL, summed over 4 daily-profile slices',
            }

    return out


# =============================================================================
# 5. EMBEDDINGS, CLUSTERINGS, DENDROGRAMS
# =============================================================================

def cluster_from_distance(distance_matrix, k=6, method='average'):
    if len(distance_matrix) < 2:
        return np.ones(len(distance_matrix), dtype=int)
    condensed = squareform(distance_matrix, checks=False)
    Z         = linkage(condensed, method=method)
    k         = max(1, min(k, len(distance_matrix) - 1))
    return fcluster(Z, t=k, criterion='maxclust')


def embed_mds(distance_matrix, random_state=42):
    if len(distance_matrix) < 2:
        return {'coords': np.zeros((len(distance_matrix), 2)), 'stress': 0.0}
    mds = MDS(
        n_components=2,
        dissimilarity='precomputed',
        random_state=random_state,
        normalized_stress='auto',
    )
    coords = mds.fit_transform(distance_matrix)
    return {'coords': coords, 'stress': float(mds.stress_)}


def truncated_dendrogram(distance_matrix, truncate_p=DENDROGRAM_TRUNCATE_P, method='average'):
    if len(distance_matrix) < 2:
        return {'segments': [], 'leaf_labels': [], 'leaf_sizes': [], 'max_height': 0}

    condensed = squareform(distance_matrix, checks=False)
    Z         = linkage(condensed, method=method)

    d = dendrogram(Z, truncate_mode='lastp', p=truncate_p,
                   show_leaf_counts=True, no_plot=True)

    segments = []
    for xs, ys in zip(d['icoord'], d['dcoord']):
        segments.append({
            'x': [float(xs[0]), float(xs[1]), float(xs[2]), float(xs[3])],
            'y': [float(ys[0]), float(ys[1]), float(ys[2]), float(ys[3])],
        })

    leaf_labels = [str(lbl) for lbl in d['ivl']]
    leaf_sizes = []
    for lbl in leaf_labels:
        if lbl.startswith('(') and lbl.endswith(')'):
            try:
                leaf_sizes.append(int(lbl[1:-1]))
            except ValueError:
                leaf_sizes.append(1)
        else:
            leaf_sizes.append(1)

    max_height = float(np.max(Z[:, 2])) if len(Z) > 0 else 0.0

    return {
        'segments':    segments,
        'leaf_labels': leaf_labels,
        'leaf_sizes':  leaf_sizes,
        'max_height':  round(max_height, 4),
    }


def compute_cluster_embedding_dendrogram(distance_matrix, k_values=(2, 3, 4, 5, 6, 8, 10)):
    emb = embed_mds(distance_matrix)
    clusters = {int(k): cluster_from_distance(distance_matrix, k=k) for k in k_values}
    dend = truncated_dendrogram(distance_matrix)
    return {
        'embedding':  emb['coords'],
        'mds_stress': emb['stress'],
        'clusters':   clusters,
        'dendrogram': dend,
    }


# =============================================================================
# 6. ALL-YEARS RUNNER
# =============================================================================

def compute_all_years(temporal_df=None, profiles_df=None, meta=None):
    if temporal_df is None:
        temporal_df = compute_all_buildings_temporal(data_dir=DATA_DIR)
    if meta is None:
        meta = _load_metadata_table()
    if profiles_df is None:
        profiles_df = compute_all_profiles(data_dir=DATA_DIR)

    meta = _normalise_meta_index(meta)
    centroids = _load_centroids()
    print(f'  centroids: {len(centroids)} buildings with coordinates')

    years = sorted(temporal_df['year'].dropna().unique().astype(int))
    print(f'Computing distances for {len(years)} years: {years}')

    print('\nBuildings per year (temporal data):')
    for yr in years:
        n = len(temporal_df[temporal_df['year'] == yr])
        print(f'  {yr}: {n} buildings')

    per_year = {}
    all_metrics_seen = set()
    archetypes_by_year = {}

    MIN_BUILDINGS = 20

    for yr in years:
        n_bldgs = len(temporal_df[temporal_df['year'] == yr])
        if n_bldgs < MIN_BUILDINGS:
            print(f'\n=== Year {yr} === SKIPPED ({n_bldgs} buildings < {MIN_BUILDINGS})')
            continue
        print(f'\n=== Year {yr} ===')
        result = compute_distances_for_year(
            temporal_df, profiles_df, meta, int(yr), centroids=centroids,
        )
        print(f'  metrics: {list(result["distances"].keys())}')
        all_metrics_seen.update(result['distances'].keys())

        if result.get('archetypes', {}).get(int(yr)):
            archetypes_by_year[int(yr)] = result['archetypes'][int(yr)]

        embeddings = {}
        for metric, mat in result['distances'].items():
            print(f'    {metric} ({mat.shape[0]} bldgs): MDS + clustering + dendrogram...')
            embeddings[metric] = compute_cluster_embedding_dendrogram(mat)

        per_year[int(yr)] = {
            'distances':    result['distances'],
            'building_ids': result['building_ids'],
            'feature_info': result['feature_info'],
            'embeddings':   embeddings,
        }

    feature_info_static = {}
    for yr in per_year.keys():
        for metric, info in per_year[int(yr)]['feature_info'].items():
            if metric not in feature_info_static:
                feature_info_static[metric] = info

    return {
        'years':               sorted(per_year.keys()),
        'per_year':            per_year,
        'feature_info_static': feature_info_static,
        'archetypes_by_year':  archetypes_by_year,
    }


# =============================================================================
# 7. ENTRY POINT
# =============================================================================

if __name__ == '__main__':

    print('Loading inputs')
    temporal = compute_all_buildings_temporal(data_dir=DATA_DIR)
    meta     = _load_metadata_table()
    if os.path.exists('daily_profiles.pkl'):
        profiles = pd.read_pickle('daily_profiles.pkl')
    else:
        profiles = compute_all_profiles(data_dir=DATA_DIR)
        profiles.to_pickle('daily_profiles.pkl')

    print('\nComputing all distance matrices across all years')
    payload = compute_all_years(temporal, profiles, meta)

    pd.to_pickle(payload, CACHE_PATH)
    print(f'\nSaved similarity cache to {CACHE_PATH}')
    print(f'  years: {payload["years"]}')
    print(f'  metrics: {list(payload["feature_info_static"].keys())}')
    import pickle
    import numpy as np

    with open('similarity_cache.pkl', 'rb') as f:
        cache = pickle.load(f)

    print(f"Years stored:  {cache['years']}")
    print(f"Metrics seen:  {list(cache['feature_info_static'].keys())}")
    print()

    for year in cache['years']:
        yd = cache['per_year'][year]
        print(f"=== {year} ===")
        for metric, dist in yd['distances'].items():
            n = dist.shape[0]
            # Distance matrix sanity
            symmetric = np.allclose(dist, dist.T)
            zero_diag = np.allclose(np.diag(dist), 0)
            nonneg    = (dist >= 0).all()
            finite    = np.isfinite(dist).all()
            rng       = (dist.min(), dist.max())
            print(f"  {metric:<25} n={n:>3}  "
                f"symmetric={symmetric}  zero_diag={zero_diag}  "
                f"nonneg={nonneg}  finite={finite}  range=[{rng[0]:.3f}, {rng[1]:.3f}]")
    
    
    print(f"{'Year':<6} ", end='')
    metrics = sorted(cache['feature_info_static'].keys())
    for m in metrics:
        print(f"{m[:18]:<20}", end='')
    print()

    for year in cache['years']:
        print(f"{year:<6} ", end='')
        yd = cache['per_year'][year]
        for m in metrics:
            if m in yd['distances']:
                print(f"{yd['distances'][m].shape[0]:<20}", end='')
            else:
                print(f"{'—':<20}", end='')
        print()
    
    def check_known_pairs(year=2022, metric='gower'):
        """
        Validate similarity matrices by checking known pairs:
        - Buildings on the same site / function should be CLOSE
        - Buildings of very different function should be FAR
        """
        yd = cache['per_year'][year]
        if metric not in yd['distances']:
            print(f"No {metric} for {year}"); return

        dist = yd['distances'][metric]
        bids = yd['building_ids'][metric]
        bid_idx = {b: i for i, b in enumerate(bids)}

        # KNOWN-SIMILAR pairs (Sidgwick site, both heating-heavy, similar function)
        similar_pairs = [
            ('b74', 'b69'),   # both Sidgwick, both heating-heavy
            ('b74', 'b72'),
            ('b79', 'b77'),   # both Downing, both gas-dependent
            ('b101', 'b104'), # both West Cambridge engineering
        ]
        # KNOWN-DIFFERENT pairs (different function, different site)
        different_pairs = [
            ('b59', 'b50'),   # Engineering process vs small meeting use
            ('b62', 'b101'),  # Biological vs Engineering, different sites
        ]

        def get_dist(a, b):
            if a in bid_idx and b in bid_idx:
                return dist[bid_idx[a], bid_idx[b]]
            return None

        print(f"\n{metric.upper()} — year {year}")
        print(f"Distance range: [{dist[dist > 0].min():.3f}, {dist.max():.3f}]")
        print(f"Median pairwise distance: {np.median(dist[np.triu_indices_from(dist, k=1)]):.3f}")
        print()
        print("Expected SIMILAR (low distance):")
        for a, b in similar_pairs:
            d = get_dist(a, b)
            if d is not None:
                print(f"  {a} ↔ {b}:  {d:.3f}")
            else:
                print(f"  {a} ↔ {b}:  (one or both missing from cohort)")
        print()
        print("Expected DIFFERENT (high distance):")
        for a, b in different_pairs:
            d = get_dist(a, b)
            if d is not None:
                print(f"  {a} ↔ {b}:  {d:.3f}")
            else:
                print(f"  {a} ↔ {b}:  (one or both missing from cohort)")

    check_known_pairs(2022, 'gower')
    check_known_pairs(2022, 'profile_euclidean_elec')

    from archive.Economies_of_scale import run_stage1  # or wherever you put it

    summary, membership = run_stage1(year=2022, n_clusters=6)

    # Now eyeball the clusters
    print("\n=== Cluster validation ===")
    print("Buildings expected to cluster together (same use type, same site):")
    print()
    known_groups = {
        'Sidgwick gas-dependent':    ['b74', 'b53', 'b72', 'b69', 'b120', 'b75', 'b71'],
        'Downing gas-dependent':     ['b79', 'b77', 'b85'],
        'West Cambridge engineering':['b101', 'b102', 'b104', 'b96', 'b31', 'b59'],
    }

    for group_name, expected in known_groups.items():
        clusters_seen = []
        for b in expected:
            if b in membership.index:
                clusters_seen.append((b, int(membership.loc[b, 'cluster']),
                                    membership.loc[b, 'archetype']))
        print(f"{group_name}:")
        for b, c, a in clusters_seen:
            print(f"  {b}  cluster={c}  archetype={a}")
        # How many distinct clusters?
        n_clusters = len(set(c for _, c, _ in clusters_seen))
        print(f"  → {n_clusters} distinct cluster(s) — "
            f"{'tight' if n_clusters == 1 else 'scattered'}")
        print()

    



    def top_neighbours(year, metric, target, top_n=5):
        yd = cache['per_year'][year]
        if metric not in yd['distances']:
            return None
        dist = yd['distances'][metric]
        bids = yd['building_ids'][metric]
        if target not in bids:
            return None
        i = bids.index(target)
        # Sort by distance, excluding self
        order = np.argsort(dist[i])
        return [(bids[j], dist[i, j]) for j in order if bids[j] != target][:top_n]

    for target in ['b101', 'b59', 'b74']:
        print(f"\nTop-5 nearest neighbours for {target} (2022):")
        for metric in ['gower', 'euclidean', 'profile_euclidean_elec', 'profile_kl_elec']:
            neighbours = top_neighbours(2022, metric, target)
            if neighbours:
                names = ', '.join(f"{b} ({d:.2f})" for b, d in neighbours)
                print(f"  {metric:<25} {names}")