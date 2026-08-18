"""
Economies_of_scale.py  (Stage I + Stage II, GAS + ELECTRICITY, GOWER)
----------------------------------------------------------------------
Full-fuel decarbonisation similarity framework, restructured to follow the
methodology of the electricity-only paper (Economies_of_scale_elec.py):

  1. TRUE GOWER CLUSTERING (Option B). Stage I clusters buildings on a
     purpose-built Gower distance computed HERE from the temporal metrics +
     metadata + centroids, rather than the cached 'archetype_geo' distance
     from similarity_cache.pkl. The feature set extends the electricity
     paper's set with the gas-derived terms:

         eui_elec, bi_elec, pli_elec, swing_elec, peak_hr_elec
                                                (continuous, elec operational)
         eui_gas, hs_slope, hs_balance, fmr    (continuous, gas / fuel-mix)
         gia_m2, date_built                    (continuous, metadata)
         function, era                         (categorical, metadata)
         location                              (folded in as ONE normalised
                                                continuous term, see
                                                _location_gower_column)

     No regeneration of similarity_cache.pkl is needed; the matrix is built
     live from metricsV3 + metadata + building_centroids.csv.

  2. SPAN-CAPPED COHORTS (replaces DBSCAN). Stage II groups same-archetype
     buildings with COMPLETE-LINKAGE agglomerative clustering under a maximum
     cohort-span threshold (`span_cap_m`), which directly bounds the largest
     pairwise distance inside any cohort. DBSCAN's eps only bounds each
     nearest-neighbour hop, so density chaining could produce elongated
     cohorts whose endpoints are far apart — too spread out to deliver as one
     coordinated programme. See find_cohorts for the full rationale.

Archetypes (Section 2 of the dissertation). Five are scored; Control-limited
is intentionally OMITTED because its weekend-gas / pre-heating diagnostics
are not computed in the temporal frame (it remains in Table 1 for
completeness only):

    Fabric-limited              high HS, high gas EUI, high balance-point temp
    Electrification-candidate   low HS, low gas EUI, high FMR
    Process-dominated           high BI, low FMR, low LF
    Flexibility candidate       high PLI/BI ratio, low LF, midday peak
    Solar-suitable              midday peak, high PLI relative to BI

Naming note: the label is 'Electrification-candidate' throughout (matching
similarity_metrics.assign_archetypes_cluster and the Stage II lever/demand/
colour dictionaries). The earlier 'Electrification-ready' alias and the dead
local scorer that used it have been removed.

Dataset: Cambridge University Estates building energy usage archive
(2000-2023), Langtry & Choudhary 2024. Metadata (function, era, GIA) and
centroids come from the estate records / aux_data.

VERIFIED COLUMN NAMES (metricsV3.compute_all_buildings_temporal):
    eui_elec, eui_gas, bi_elec, bi_gas, pli_elec, pli_gas, lf_elec,
    fmr, hs_slope, hs_balance, hs_r2, peak_hr_elec
Derived here: swing_elec = pli_elec - bi_elec
              pli_bi_ratio_elec = pli_elec / bi_elec
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance  import squareform
from sklearn.metrics         import silhouette_score
from sklearn.cluster         import AgglomerativeClustering

from metricsV3           import compute_all_buildings_temporal
from non_spatial_metrics import _load_metadata_table
# Reuse the validated NaN-tolerant Gower kernel from similarity_metrics.
from similarity_metrics  import _gower_distance

CENTROIDS_CSV = 'building_centroids.csv'   # adjust path if not in working dir


# =============================================================================
# GAS + ELECTRICITY GOWER FEATURE DEFINITION
# =============================================================================
# Continuous operational metrics (from temporal frame). 'swing_elec' is derived.
# This extends the electricity paper's set {EUI, BI, PLI, Swing, peak timing}
# with the gas-side terms {gas EUI, heating sensitivity, balance point, FMR}.
GOWER_CONT_OPERATIONAL = ['eui_elec', 'bi_elec', 'pli_elec',
                          'swing_elec', 'peak_hr_elec',
                          'eui_gas', 'hs_slope', 'hs_balance', 'fmr']
# Continuous metadata metrics (from metadata table).
GOWER_CONT_META = ['gia_m2', 'date_built']
# Categorical metadata.
GOWER_CAT_META  = ['function', 'era']
# Location is collapsed to one continuous scalar (see _location_gower_column)
# so it participates in the standard unweighted Gower average as ONE feature.

# Diagnostic features used for archetype MATCHING (per-cluster medians).
DIAGNOSTIC_FEATURES = [
    'eui_elec', 'eui_gas',
    'bi_elec',  'pli_elec', 'lf_elec',
    'fmr',
    'hs_slope',            # heating sensitivity (slope of gas vs degree-day)
    'hs_balance',          # balance-point temperature
    'peak_hr_elec',
    'pli_bi_ratio_elec',   # derived in build_features
]


def _normalise_bid(series) -> pd.Series:
    """Coerce a building_id column/Series to 'b<n>' string format."""
    s = pd.Series(series).astype(str).str.strip()
    return s.where(s.str.startswith('b'), 'b' + s)


def _load_centroids() -> pd.DataFrame:
    """Load building_centroids.csv -> DataFrame indexed by 'b<id>' with
    columns ['lat', 'lon']. Robust to column-name variants. Returns an empty
    frame (not an error) if the file is missing, so Gower can still run
    without the location term."""
    if not os.path.exists(CENTROIDS_CSV):
        print(f"  [warn] {CENTROIDS_CSV} not found; Gower will run WITHOUT the "
              f"location term, and Stage II geographic clustering will be empty.")
        return pd.DataFrame(columns=['lat', 'lon'])
    df = pd.read_csv(CENTROIDS_CSV)
    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ('latitude', 'lat'):            rename[c] = 'lat'
        elif lc in ('longitude', 'lon', 'long'): rename[c] = 'lon'
        elif lc in ('building_id', 'bid', 'id'): rename[c] = 'building_id'
    df = df.rename(columns=rename)
    df['building_id'] = _normalise_bid(df['building_id'])
    df = df.dropna(subset=['lat', 'lon'])
    return df.set_index('building_id')[['lat', 'lon']]


# =============================================================================
# 1. BUILD THE GAS+ELEC GOWER FEATURE TABLE + DISTANCE MATRIX
# =============================================================================

def build_features(year: int = 2022) -> pd.DataFrame:
    """Assemble the per-building feature table used for BOTH Gower clustering
    and archetype matching. Indexed by 'b<id>'.

    Columns produced:
        eui_elec, eui_gas, bi_elec, pli_elec, lf_elec,
        peak_hr_elec, fmr, hs_slope, hs_balance          (operational)
        swing_elec        = pli_elec - bi_elec           (derived)
        pli_bi_ratio_elec = pli_elec / bi_elec           (derived)
        gia_m2, date_built, function, era                (metadata)
        lat, lon                                         (centroids)
    """
    temporal = compute_all_buildings_temporal()
    snap = temporal[temporal['year'] == year].copy()
    snap['building_id'] = _normalise_bid(snap['building_id'])
    snap = snap.drop_duplicates(subset='building_id').set_index('building_id')

    # Derived electricity shape metrics
    snap['swing_elec'] = snap['pli_elec'] - snap['bi_elec']
    snap['pli_bi_ratio_elec'] = snap['pli_elec'] / snap['bi_elec'].replace(0, np.nan)

    # Metadata (prefer metadata GIA over any temporal copy)
    meta = _load_metadata_table().reset_index()
    meta['building_id'] = _normalise_bid(meta['building_id'])
    meta = meta.drop_duplicates(subset='building_id').set_index('building_id')
    snap = snap.drop(columns=['gia_m2'], errors='ignore')
    meta_cols = [c for c in (GOWER_CONT_META + GOWER_CAT_META) if c in meta.columns]
    feat = snap.join(meta[meta_cols], how='left')

    # Centroids
    cent = _load_centroids()
    feat = feat.join(cent, how='left')

    return feat


def _location_gower_column(feat: pd.DataFrame) -> pd.Series:
    """Collapse lat/lon into a single pre-normalised scalar so location can be
    passed to the standard Gower kernel as one continuous feature.

    Each building's scalar is its mean great-circle distance (km) to all other
    buildings. Two buildings that are both central (or both peripheral) get
    similar values; the Gower kernel then range-normalises this like any other
    continuous feature, keeping location as ONE feature (weight 1/n_features)
    consistent with the paper's unweighted Gower averaging, rather than
    letting a raw pairwise-distance matrix dominate.
    """
    if 'lat' not in feat.columns or feat['lat'].notna().sum() < 2:
        return pd.Series(np.nan, index=feat.index, name='loc_scalar')
    lat = np.radians(pd.to_numeric(feat['lat'], errors='coerce').values)
    lon = np.radians(pd.to_numeric(feat['lon'], errors='coerce').values)
    latc = lat[:, None]; lonc = lon[:, None]
    dlat = latc - latc.T; dlon = lonc - lonc.T
    a = np.sin(dlat/2)**2 + np.cos(latc)*np.cos(latc.T)*np.sin(dlon/2)**2
    km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    with np.errstate(invalid='ignore'):
        mean_km = np.nanmean(np.where(km == 0, np.nan, km), axis=1)
    return pd.Series(mean_km, index=feat.index, name='loc_scalar')


def compute_gower_distance(feat: pd.DataFrame):
    """Compute the gas+electricity Gower distance matrix on the feature table.

    Returns (dist_matrix, building_ids). Buildings missing ALL operational
    metrics are dropped; per-feature missingness is handled inside the Gower
    kernel (skipped, not penalised) — so all-electric buildings with no gas
    metrics still cluster on their remaining features.
    """
    op_present = feat[[c for c in GOWER_CONT_OPERATIONAL if c in feat.columns]] \
                     .notna().any(axis=1)
    feat = feat[op_present].copy()

    feat['loc_scalar'] = _location_gower_column(feat)

    cont_cols = [c for c in (GOWER_CONT_OPERATIONAL + GOWER_CONT_META
                             + ['loc_scalar'])
                 if c in feat.columns]
    cat_cols  = [c for c in GOWER_CAT_META if c in feat.columns]

    dist = _gower_distance(feat[cont_cols + cat_cols], cont_cols, cat_cols)
    return dist, feat.index.tolist()


# =============================================================================
# 2. OPTIMAL-K DIAGNOSTICS
# =============================================================================

def find_optimal_k(year: int = 2022,
                   k_range=range(4, 20),
                   method: str = 'average',
                   plot: bool = True):
    """Silhouette + dendrogram-gap analysis on the gas+elec Gower matrix."""
    feat = build_features(year)
    dist, bids = compute_gower_distance(feat)
    n = len(dist)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)

    rows = []
    for k in k_range:
        if k >= n:
            break
        labels = fcluster(Z, t=k, criterion='maxclust')
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(dist, labels, metric='precomputed')
        sizes = pd.Series(labels).value_counts()
        rows.append({
            'k':             k,
            'silhouette':    round(sil, 4),
            'max_cluster_%': round(sizes.max() / n * 100, 1),
            'n_singletons':  int((sizes == 1).sum()),
            'mean_size':     round(n / k, 1),
        })
    results = pd.DataFrame(rows)

    heights = Z[:, 2]
    gaps = np.diff(heights)
    gap_info = [(n - i - 1, gap) for i, gap in enumerate(gaps)]
    gap_info.sort(key=lambda x: x[1], reverse=True)
    top_gaps = [(k, g) for k, g in gap_info if k in list(k_range)][:5]

    print(f"\nOptimal-k (gas+elec Gower): year={year}, "
          f"linkage={method}, n_buildings={n}\n")
    if not results.empty:
        print(results.to_string(index=False))
    print(f"\nLargest dendrogram gaps (suggest natural k values):")
    for k, gap in top_gaps:
        print(f"  k = {k:<3}  (gap = {gap:.4f})")

    if not results.empty:
        best_sil_k = int(results.loc[results['silhouette'].idxmax(), 'k'])
        best_sil = results['silhouette'].max()
        print(f"\nSilhouette suggests: k = {best_sil_k}")
        if top_gaps:
            print(f"Dendrogram gap suggests: k = {top_gaps[0][0]}")
        if best_sil < 0.25:
            print(f"  [warning] best silhouette = {best_sil:.3f} is weak. "
                  f"Cluster structure may not be strong.")
        elif best_sil < 0.5:
            print(f"  [note] best silhouette = {best_sil:.3f} is moderate.")
        else:
            print(f"  [ok] best silhouette = {best_sil:.3f} is strong.")
    else:
        best_sil_k = None

    if plot and not results.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax = axes[0]
        ax.plot(results['k'], results['silhouette'], marker='o',
                linewidth=2, color='C0')
        ax.axvline(best_sil_k, color='C0', linestyle='--', alpha=0.5,
                   label=f'silhouette max: k={best_sil_k}')
        if top_gaps:
            ax.axvline(top_gaps[0][0], color='C3', linestyle=':', alpha=0.5,
                       label=f'gap max: k={top_gaps[0][0]}')
        ax.set_xlabel('k'); ax.set_ylabel('Silhouette score')
        ax.set_title(f'Silhouette by k (gas+elec Gower, {year})')
        ax.grid(True, linestyle='--', alpha=0.4); ax.legend(fontsize=9)
        ax = axes[1]
        ax.bar(results['k'], results['max_cluster_%'], color='grey', alpha=0.6)
        ax.axhline(50, color='C3', linestyle='--', linewidth=0.8,
                   label='50% threshold (dominant cluster)')
        ax.set_xlabel('k'); ax.set_ylabel('Largest cluster size (% of estate)')
        ax.set_title('Cluster size imbalance')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4); ax.legend(fontsize=9)
        plt.tight_layout(); plt.show()

    return results, top_gaps


# =============================================================================
# 3. CLUSTERING
# =============================================================================

def cluster_buildings(feat: pd.DataFrame,
                      n_clusters: int = 15,
                      method: str = 'average'):
    """HAC on the gas+elec Gower matrix. Returns (labels, dist, bids, Z)."""
    dist, bids = compute_gower_distance(feat)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    return pd.Series(labels, index=bids, name='cluster'), dist, bids, Z


# =============================================================================
# 4. ARCHETYPE MATCHING (GAS + ELECTRICITY — 5 archetypes)
# =============================================================================

def _assign_archetype(cluster_medians: pd.Series,
                      estate_medians: pd.Series,
                      min_score_threshold: float = 0.15):
    """Rule-based archetype matching for one cluster (gas + electricity).

    Each archetype score is the MEAN of relative deviations of its signature
    features, so scores are comparable across archetypes regardless of how
    many terms they combine. A relative deviation is
        (cluster_median - estate_median) / |estate_median|.
    Terms whose inputs are missing are skipped, not penalised — an all-electric
    cluster with no gas medians simply cannot score on the gas archetypes.

    Control-limited is NOT scored (weekend-gas / pre-heating diagnostics are
    not computed); it is documented in Table 1 only.

    Returns (best_archetype, all_scores_dict). If the best score is below
    `min_score_threshold`, returns ('Unclassified', scores).
    """
    def rel(feature):
        med = estate_medians.get(feature, np.nan)
        val = cluster_medians.get(feature, np.nan)
        if pd.isna(med) or pd.isna(val) or med == 0:
            return np.nan
        return (val - med) / abs(med)

    def score(terms):
        vals = [t for t in terms if not pd.isna(t)]
        return float(np.mean(vals)) if vals else np.nan

    # 'peak midday' indicator: +1 if cluster peak hour is 10-15, else -1.
    peak = cluster_medians.get('peak_hr_elec', np.nan)
    midday = 0.0 if pd.isna(peak) else (1.0 if 10 <= peak <= 15 else -1.0)

    scores = {
        # high HS, high gas EUI, high balance-point temp
        'Fabric-limited': score([
            rel('hs_slope'), rel('eui_gas'), rel('hs_balance')]),

        # low HS, low gas EUI, high FMR
        'Electrification-candidate': score([
            -rel('hs_slope'), -rel('eui_gas'), rel('fmr')]),

        # high BI, low FMR, low LF
        'Process-dominated': score([
            rel('bi_elec'), -rel('fmr'), rel('swing_elec')]),

        # high PLI/BI ratio, low LF, midday peak
        'Flexibility candidate': score([
            rel('pli_bi_ratio_elec'), rel('swing_elec'), midday]),
        

        # midday peak, high PLI relative to BI
        # 'Solar-suitable': score([
        #     midday, rel('pli_elec'), -rel('bi_elec')]),
        'Solar-suitable': score([
            midday, rel('pli_elec')]),
    }

    valid = {k: v for k, v in scores.items() if not pd.isna(v)}
    if not valid:
        return 'Unclassified', scores
    best = max(valid, key=valid.get)
    if valid[best] < min_score_threshold:
        return 'Unclassified', scores
    return best, scores


# =============================================================================
# 5. STAGE I PIPELINE
# =============================================================================

def run_stage1(year: int = 2022,
               n_clusters: int = 15,
               method: str = 'average',
               min_score_threshold: float = 0.15):
    """Stage I: gas+elec Gower clustering + archetype matching.

      1. Build the feature table (temporal metrics + metadata + centroids).
      2. Compute the Gower distance and cluster with HAC (average linkage).
      3. Score each CLUSTER's median diagnostic vector against the estate
         medians via _assign_archetype (single local source of truth).
      4. Report cluster -> archetype, sizes, members, and score vectors.

    Returns (summary, membership, feat).
    """
    feat = build_features(year)
    labels, dist, bids, Z = cluster_buildings(feat, n_clusters, method)

    feat = feat.loc[labels.index].copy()
    feat['cluster'] = labels

    diag = [c for c in DIAGNOSTIC_FEATURES if c in feat.columns]
    estate_medians = feat[diag].median(numeric_only=True)

    rows = []
    building_arch = pd.Series(index=labels.index, dtype=object)
    for cid, grp in feat.groupby('cluster'):
        cluster_medians = grp[diag].median(numeric_only=True)
        arch, scores = _assign_archetype(cluster_medians, estate_medians,
                                         min_score_threshold)
        members = sorted(grp.index.tolist())
        building_arch.loc[members] = arch
        rows.append({
            'cluster_id':  int(cid),
            'n_buildings': len(members),
            'archetype':   arch,
            'members':     members,
            'scores':      {k: (round(v, 3) if pd.notna(v) else None)
                            for k, v in scores.items()},
        })

    summary = (pd.DataFrame(rows)
               .sort_values('n_buildings', ascending=False)
               .reset_index(drop=True))
    membership = pd.DataFrame({'cluster': labels})
    membership['archetype'] = building_arch

    print(f"\n{'='*70}")
    print(f"Stage I (gas+elec Gower) — year {year}, k={n_clusters}, "
          f"linkage={method}")
    print(f"{'='*70}\n")
    for _, r in summary.iterrows():
        print(f"--- Cluster {r['cluster_id']}: {r['archetype']} "
              f"({r['n_buildings']} buildings) ---")
        print(f"  scores: {r['scores']}")
        print(f"  members: {', '.join(r['members'])}\n")

    return summary, membership, feat


# =============================================================================
# 6. STAGE II: ECONOMIES-OF-SCALE COHORT DISCOVERY
# =============================================================================
# Consumes the summary/membership/feat returned by run_stage1, plus
# building_centroids.csv, and finds geographic cohorts of same-archetype
# buildings that can be delivered together.

# Which scale lever each archetype uses. Geography is primary for these three;
# Process/Flexibility are interpreted functionally/by-load and are flagged.
ARCHETYPE_SCALE_LEVER = {
    'Fabric-limited':            'geographic',   # shared contractor / bulk fabric
    'Electrification-candidate': 'geographic',   # heat network / shared array
    'Solar-suitable':            'geographic',   # shared grid connection + PV procurement
    'Flexibility candidate':     'load',         # aggregate shiftable capacity
    'Process-dominated':         'functional',   # shared SOPs / monitoring
}

# Demand column most relevant to each archetype's economy of scale
ARCHETYPE_DEMAND_COL = {
    'Fabric-limited':            'eui_gas',
    'Electrification-candidate': 'eui_gas',
    'Solar-suitable':            'eui_elec',
    'Flexibility candidate':     'eui_elec',
    'Process-dominated':         'eui_elec',
}


def build_archetype_membership(summary: pd.DataFrame,
                               membership: pd.DataFrame,
                               score_threshold: float = 0.0) -> dict:
    """Pivot Stage I output into per-archetype building lists.

    A building is included in an archetype's list if ITS CLUSTER scores at or
    above `score_threshold` for that archetype. This honours the multi-label
    idea in Table 2: a cluster suitable for both EC and SS contributes its
    buildings to both lists. With score_threshold=0.0 this is "any positive
    suitability"; raise it (e.g. 0.5) to be stricter.

    Returns {archetype: [building_id, ...]}.
    """
    cluster_scores = dict(zip(summary['cluster_id'], summary['scores']))
    members_by_cluster = (membership.reset_index()
                          .rename(columns={'index': 'building_id'})
                          .groupby('cluster')['building_id']
                          .apply(list).to_dict())

    archetype_lists = {}
    for cid, scores in cluster_scores.items():
        blds = members_by_cluster.get(cid, [])
        for arch, sc in scores.items():
            if sc is None:
                continue
            if sc >= score_threshold:
                archetype_lists.setdefault(arch, []).extend(blds)

    # de-duplicate while preserving order
    for arch in archetype_lists:
        seen, uniq = set(), []
        for b in archetype_lists[arch]:
            if b not in seen:
                seen.add(b); uniq.append(b)
        archetype_lists[arch] = uniq
    return archetype_lists


def _haversine_distance_matrix(latlon_deg: np.ndarray) -> np.ndarray:
    """Pairwise haversine distance matrix (metres) for an array of [lat, lon]."""
    r = np.radians(latlon_deg)
    lat = r[:, 0][:, None]; lon = r[:, 1][:, None]
    dlat = lat - lat.T; dlon = lon - lon.T
    a = np.sin(dlat / 2)**2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2)**2
    return 6371000.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def find_cohorts(archetype_lists: dict,
                 centroids: pd.DataFrame,
                 span_cap_m: float = 300.0,
                 min_samples: int = 2) -> pd.DataFrame:
    """Group buildings within each geographic-lever archetype into spatial
    cohorts using COMPLETE-LINKAGE agglomerative clustering with a maximum
    cohort-span distance threshold, in place of DBSCAN.

    WHY NOT DBSCAN: DBSCAN links clusters via nearest-neighbour density
    reachability, so a chain of buildings each within eps of the next can
    join two spatially distant groups into one elongated cluster, even when
    the cluster's endpoints are far apart — producing cohorts too large to
    deliver as a single coordinated site. Complete linkage instead bounds
    the MAXIMUM pairwise distance within any resulting cluster at
    `span_cap_m`, directly capping each cohort's physical footprint, while
    still merging buildings separated by modest gaps (e.g. two rows either
    side of a car park) that a strict nearest-neighbour epsilon would
    otherwise split apart.

    Archetypes whose lever is not 'geographic' are passed through as a single
    functional/load group (not spatially split).

    span_cap_m  : maximum pairwise distance (m) allowed within a cohort.
    min_samples : minimum buildings to form a cohort (2 = adjacent pair counts).
                  Enforced post-hoc: undersized groups become noise.

    cohort_id == -1 marks buildings left in a group smaller than
    `min_samples` (isolated / noise).
    """
    rows = []
    for arch, blds in archetype_lists.items():
        lever = ARCHETYPE_SCALE_LEVER.get(arch, 'geographic')
        have = [b for b in blds if b in centroids.index]
        if not have:
            continue
        sub = centroids.loc[have]

        if lever != 'geographic':
            # Non-spatial lever: treat the whole archetype as one functional group
            for b in have:
                rows.append({'archetype': arch, 'lever': lever, 'cohort_id': 0,
                             'building_id': b,
                             'lat': sub.loc[b, 'lat'], 'lon': sub.loc[b, 'lon']})
            continue

        if len(have) < min_samples:
            for b in have:
                rows.append({'archetype': arch, 'lever': lever, 'cohort_id': -1,
                             'building_id': b,
                             'lat': sub.loc[b, 'lat'], 'lon': sub.loc[b, 'lon']})
            continue

        dist_m = _haversine_distance_matrix(sub[['lat', 'lon']].values)
        # NOTE: 'metric=' requires scikit-learn >= 1.2; on older versions the
        # keyword is 'affinity='.
        model = AgglomerativeClustering(
            n_clusters=None, metric='precomputed', linkage='complete',
            distance_threshold=span_cap_m)
        labels = model.fit_predict(dist_m)

        # enforce min_samples post-hoc: undersized groups become noise (-1)
        sizes = pd.Series(labels).value_counts()
        for b, lab in zip(have, labels):
            lab_out = int(lab) if sizes[lab] >= min_samples else -1
            rows.append({'archetype': arch, 'lever': lever, 'cohort_id': lab_out,
                         'building_id': b,
                         'lat': sub.loc[b, 'lat'], 'lon': sub.loc[b, 'lon']})

    return pd.DataFrame(rows)


def _max_pairwise_km(latlon: np.ndarray) -> float:
    """Largest pairwise haversine distance (km) in a cohort."""
    if len(latlon) < 2:
        return 0.0
    r = np.radians(latlon)
    lat = r[:, 0][:, None]; lon = r[:, 1][:, None]
    dlat = lat - lat.T; dlon = lon - lon.T
    a = np.sin(dlat/2)**2 + np.cos(lat)*np.cos(lat.T)*np.sin(dlon/2)**2
    km = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return float(km.max())


def characterise_cohorts(cohorts: pd.DataFrame,
                         feat: pd.DataFrame,
                         min_cohort_size: int = 2) -> pd.DataFrame:
    """Summarise each (archetype, cohort) with its economy-of-scale levers:
    size, total GIA, relevant demand (gas EUI for fabric/electrification
    cohorts, elec EUI for the rest, per ARCHETYPE_DEMAND_COL), estate-share %,
    physical homogeneity (age & GIA spread), and spatial compactness.

    Uses the Stage I `feat` table directly (no recomputation of the temporal
    frame). Spatial noise (cohort_id == -1) is excluded; functional groups
    (cohort_id from non-geographic archetypes) are retained.

    Returns one row per cohort, sorted by estate-share descending.
    """
    gia = pd.to_numeric(feat.get('gia_m2'),     errors='coerce')
    age = pd.to_numeric(feat.get('date_built'), errors='coerce')

    # Estate-wide GIA-weighted demand totals per fuel (for share calculation)
    def _estate_total(col):
        eui = pd.to_numeric(feat.get(col), errors='coerce')
        return float((eui * gia).sum(skipna=True))

    estate_tot = {c: _estate_total(c) for c in set(ARCHETYPE_DEMAND_COL.values())}

    out = []
    for (arch, cid), grp in cohorts.groupby(['archetype', 'cohort_id']):
        if cid == -1:                      # isolated noise: skip
            continue
        blds = grp['building_id'].tolist()
        if len(blds) < min_cohort_size:
            continue

        gia_c = gia.reindex(blds)
        age_c = age.reindex(blds)
        demand_col = ARCHETYPE_DEMAND_COL.get(arch, 'eui_elec')
        eui_c = pd.to_numeric(feat.reindex(blds).get(demand_col), errors='coerce')
        cohort_demand = float((eui_c * gia_c).sum(skipna=True))
        share = (100.0 * cohort_demand / estate_tot[demand_col]
                 if estate_tot.get(demand_col) else np.nan)

        compact_km = _max_pairwise_km(grp[['lat', 'lon']].values)

        out.append({
            'archetype':      arch,
            'lever':          grp['lever'].iloc[0],
            'cohort_id':      cid,
            'n_buildings':    len(blds),
            'members':        sorted(blds),
            'total_gia_m2':   round(float(gia_c.sum(skipna=True)), 0),
            'demand_basis':   demand_col,
            'cohort_demand':  round(cohort_demand, 0),
            'estate_share_%': round(share, 1) if not pd.isna(share) else None,
            'gia_iqr':        round(float(gia_c.quantile(.75) - gia_c.quantile(.25)), 0)
                              if gia_c.notna().sum() > 1 else None,
            'age_span_yr':    int(age_c.max() - age_c.min())
                              if age_c.notna().sum() > 1 else None,
            'span_km':        round(compact_km, 2),
        })

    summary = pd.DataFrame(out)
    if not summary.empty:
        summary = summary.sort_values('estate_share_%', ascending=False,
                                      na_position='last').reset_index(drop=True)
    return summary


def cohort_impact_summary(cohort_summary: pd.DataFrame) -> pd.DataFrame:
    """Estate-wide headline: per archetype, how many coordinated cohorts
    exist, how many buildings they cover, and what cumulative share of the
    relevant estate demand they address. This is the "X% addressable through
    Y coordinated programmes" result.
    """
    if cohort_summary.empty:
        return pd.DataFrame()
    rows = []
    for arch, grp in cohort_summary.groupby('archetype'):
        rows.append({
            'archetype':          arch,
            'n_cohorts':          len(grp),
            'n_buildings':        int(grp['n_buildings'].sum()),
            'cumulative_share_%': round(grp['estate_share_%'].sum(skipna=True), 1),
        })
    return (pd.DataFrame(rows)
            .sort_values('cumulative_share_%', ascending=False)
            .reset_index(drop=True))


def run_stage2(year: int = 2022,
               n_clusters: int = 15,
               span_cap_m: float = 300.0,
               min_samples: int = 2,
               score_threshold: float = 0.0,
               min_cohort_size: int = 2):
    """Full Stage II pipeline (gas+elec Gower), driven by live Stage I output.

      1. Run Stage I -> cluster suitability scores + membership + features.
      2. Pivot to per-archetype building lists (multi-label via scores).
      3. Span-capped complete-linkage cohorts within geographic-lever
         archetypes.
      4. Characterise + rank cohorts by estate-share (fuel per archetype).
      5. Print the estate-wide impact summary.

    Returns (cohorts_long, cohort_summary, impact, feat, archetype_lists).
    """
    summary, membership, feat = run_stage1(year=year, n_clusters=n_clusters)
    centroids = _load_centroids()

    print(f"\n  centroids available for {len(centroids)} buildings")
    archetype_lists = build_archetype_membership(summary, membership,
                                                 score_threshold=score_threshold)
    for arch, blds in archetype_lists.items():
        print(f"    {arch}: {len(blds)} candidate buildings")

    cohorts = find_cohorts(archetype_lists, centroids,
                           span_cap_m=span_cap_m, min_samples=min_samples)
    cohort_summary = characterise_cohorts(cohorts, feat,
                                          min_cohort_size=min_cohort_size)
    impact = cohort_impact_summary(cohort_summary)

    print(f"\n{'='*70}")
    print(f"Stage II — economies-of-scale cohorts "
          f"(span cap={span_cap_m:.0f} m, min_samples={min_samples})")
    print(f"{'='*70}\n")
    if cohort_summary.empty:
        print("  No cohorts met the size threshold.")
    else:
        for _, r in cohort_summary.iterrows():
            print(f"  [{r['archetype']}] cohort {r['cohort_id']} "
                  f"({r['n_buildings']} bldgs, {r['estate_share_%']}% of estate "
                  f"{r['demand_basis']}, span {r['span_km']} km)")
            print(f"     {', '.join(r['members'])}")
        print(f"\n  --- Estate-wide impact ---")
        print(impact.to_string(index=False))

    return cohorts, cohort_summary, impact, feat


# =============================================================================
# 7. STAGE II: COHORT GIS MAP
# =============================================================================
# Renders a single self-contained Leaflet HTML map of the geographic
# decarbonisation cohorts found by run_stage2():
#   - one circle per building, positioned at its centroid
#   - circle colour = intervention archetype
#   - each geographic cohort drawn as a shaded convex hull around its members
#   - a layer toggle switches archetypes on/off (handles multi-archetype overlap)
#   - functional/load-based groups (Process-dominated, Flexibility) are excluded
#     because they are not spatial cohorts

# Colour per archetype (colour-blind-aware, distinct hues)
ARCHETYPE_COLOURS = {
    'Fabric-limited':            '#e31a1c',   # red
    'Electrification-candidate': '#1f78b4',   # blue
    'Solar-suitable':            '#33a02c',   # green
    'Flexibility candidate':     '#6a3d9a',   # purple (functional; not drawn)
    'Process-dominated':         '#ff7f00',   # orange (functional; not drawn)
}

# Only these archetypes are drawn as spatial cohorts
GEOGRAPHIC_ARCHETYPES = [a for a, lever in ARCHETYPE_SCALE_LEVER.items()
                         if lever == 'geographic']


def _convex_hull(points):
    """Andrew's monotone-chain convex hull.
    points: list of (lon, lat). Returns hull vertices as [(lon, lat), ...]
    in order, or the input itself if < 3 points (a hull needs 3+).
    """
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _build_cohort_map_payload(cohorts: pd.DataFrame,
                              cohort_summary: pd.DataFrame,
                              centroids: pd.DataFrame) -> dict:
    """Assemble the JSON payload the map JS consumes.

    cohorts        : long DataFrame from find_cohorts (archetype, lever,
                     cohort_id, building_id, lat, lon)
    cohort_summary : ranked DataFrame from characterise_cohorts (used for
                     cohort labels: estate share, span, demand basis)
    centroids      : DataFrame indexed by 'b<id>' with lat/lon (context dots)
    """
    summ = {}
    for _, r in cohort_summary.iterrows():
        summ[(r['archetype'], r['cohort_id'])] = {
            'n':     int(r['n_buildings']),
            'share': r['estate_share_%'],
            'span':  r['span_km'],
            'basis': r['demand_basis'],
        }

    layers = []
    for arch in GEOGRAPHIC_ARCHETYPES:
        sub = cohorts[(cohorts['archetype'] == arch) &
                      (cohorts['cohort_id'] >= 0)]   # drop noise (-1)
        if sub.empty:
            continue

        cohort_blocks = []
        for cid, grp in sub.groupby('cohort_id'):
            members = grp['building_id'].tolist()
            info = summ.get((arch, cid))
            # only label/hull cohorts that survived the size filter (in summary)
            if info is None:
                continue
            pts_lonlat = list(zip(grp['lon'].tolist(), grp['lat'].tolist()))
            hull = _convex_hull(pts_lonlat)
            cohort_blocks.append({
                'cohort_id': int(cid),
                'members':   members,
                'points':    [{'bid': b,
                               'lat': float(la),
                               'lon': float(lo)}
                              for b, la, lo in zip(members,
                                                   grp['lat'].tolist(),
                                                   grp['lon'].tolist())],
                'hull':      [[la, lo] for lo, la in hull],  # leaflet wants [lat,lon]
                'label':     (f"{arch} cohort {cid}: {info['n']} bldgs, "
                              f"{info['share']}% of estate {info['basis']}, "
                              f"span {info['span']} km"),
            })

        if cohort_blocks:
            layers.append({
                'archetype': arch,
                'colour':    ARCHETYPE_COLOURS.get(arch, '#666'),
                'cohorts':   cohort_blocks,
            })

    # Context layer: all estate buildings as faint grey dots
    context = [{'bid': b, 'lat': float(r['lat']), 'lon': float(r['lon'])}
               for b, r in centroids.iterrows()]

    centre_lat = float(centroids['lat'].mean())
    centre_lon = float(centroids['lon'].mean())

    return {'layers': layers, 'context': context,
            'centre': [centre_lat, centre_lon]}


def render_cohort_map(cohorts: pd.DataFrame,
                      cohort_summary: pd.DataFrame,
                      centroids: pd.DataFrame,
                      output_path: str = 'stage2_cohort_map.html') -> None:
    """Write the standalone Leaflet cohort map to `output_path`."""
    payload = _build_cohort_map_payload(cohorts, cohort_summary, centroids)
    payload_json = json.dumps(payload)

    css = """
      html, body { margin:0; padding:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif; }
      #map { position:absolute; top:64px; bottom:0; left:0; right:0; }
      #header { height:64px; box-sizing:border-box; padding:10px 18px;
                background:#1b5e20; color:#fff; }
      #header h1 { margin:0; font-size:18px; font-weight:600; }
      #header p  { margin:2px 0 0; font-size:12px; opacity:0.85; }
      .ctrl { position:absolute; top:78px; right:14px; z-index:1000;
              background:#fff; border-radius:8px; padding:10px 12px;
              box-shadow:0 1px 6px rgba(0,0,0,0.3); font-size:18px; max-width:260px; }
      .ctrl h4 { margin:0 0 6px; font-size:13px; }
      .ctrl label { display:block; margin:3px 0; cursor:pointer; }
      .swatch { display:inline-block; width:12px; height:12px; border-radius:2px;
                margin-right:6px; vertical-align:middle; }
      .leaflet-tooltip { font-size:12px; }
    """

    js = """
    var PAYLOAD = __PAYLOAD_JSON__;
    var map = L.map('map').setView(PAYLOAD.centre, 14);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap &copy; CARTO', subdomains:'abcd', maxZoom:20
    }).addTo(map);

    // Context: faint grey dot per estate building
    PAYLOAD.context.forEach(function(p) {
      L.circleMarker([p.lat, p.lon], {
        radius:3, color:'#bbb', weight:1, opacity:0.5,
        fillColor:'#ddd', fillOpacity:0.4
      }).addTo(map).bindTooltip(p.bid, {sticky:true});
    });

    // One Leaflet layerGroup per archetype, so they can be toggled
    var archLayers = {};
    PAYLOAD.layers.forEach(function(layer) {
      var grp = L.layerGroup();
      layer.cohorts.forEach(function(c) {
        // shaded hull
        if (c.hull.length >= 3) {
          L.polygon(c.hull, {
            color: layer.colour, weight:2, opacity:0.7,
            fillColor: layer.colour, fillOpacity:0.12
          }).addTo(grp).bindTooltip(c.label, {sticky:true});
        }
        // member circles
        c.points.forEach(function(pt) {
          L.circleMarker([pt.lat, pt.lon], {
            radius:7, color:'#333', weight:1, opacity:0.9,
            fillColor: layer.colour, fillOpacity:0.8
          }).addTo(grp).bindTooltip(
            '<b>' + pt.bid + '</b><br>' + c.label, {sticky:true, maxWidth:260});
        });
      });
      grp.addTo(map);
      archLayers[layer.archetype] = {grp: grp, colour: layer.colour};
    });

    // Build the toggle control
    var ctrl = document.getElementById('layer-toggles');
    Object.keys(archLayers).forEach(function(arch) {
      var info = archLayers[arch];
      var lab = document.createElement('label');
      var cb = document.createElement('input');
      cb.type = 'checkbox'; cb.checked = true;
      cb.addEventListener('change', function() {
        if (this.checked) info.grp.addTo(map);
        else map.removeLayer(info.grp);
      });
      var sw = document.createElement('span');
      sw.className = 'swatch'; sw.style.background = info.colour;
      lab.appendChild(cb); lab.appendChild(sw);
      lab.appendChild(document.createTextNode(' ' + arch));
      ctrl.appendChild(lab);
    });
    """

    js_final = js.replace('__PAYLOAD_JSON__', payload_json)

    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"/>'
        '<title>Stage II — Decarbonisation Cohorts</title>'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>'
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>'
        '<style>' + css + '</style></head><body>'
        '<div id="header"><h1>Stage II — Decarbonisation Cohorts</h1>'
        '<p>Buildings coloured by intervention archetype; shaded areas are geographic '
        'cohorts deliverable as one coordinated programme. Functional groups '
        '(Process-dominated, Flexibility) are not spatial and are excluded.</p></div>'
        '<div class="ctrl"><h4>Archetypes</h4><div id="layer-toggles"></div></div>'
        '<div id="map"></div>'
        '<script>' + js_final + '</script>'
        '</body></html>'
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nCohort map saved to: {output_path}")
    print(f"  archetypes drawn: {[l['archetype'] for l in payload['layers']]}")
    print(f"  context buildings: {len(payload['context'])}")
    print("  open the HTML file in your browser.")

# =============================================================================
# REPORT FIGURE: ARCHETYPE MEMBERSHIP OVERLAP (UpSet plot)
# =============================================================================
# Perfectly overlapping cohorts (the same buildings claimed by two archetypes)
# are invisible on a single map: one hull hides under the other and a marker
# can only show one fill colour. For the report the overlap is instead shown
# as an UpSet plot of EXCLUSIVE archetype-membership intersections, which
# partitions the estate into archetype-combination groups and answers "how many
# buildings sit in which combination of archetypes" unambiguously, staying
# readable for the full six-archetype set where a Venn diagram would not.

def plot_archetype_upset(archetype_lists: dict,
                         min_intersection: int = 1,
                         save_path: str = 'fig_archetype_upset.png',
                         show: bool = True):
    """UpSet plot of exclusive archetype-membership intersections.

    Each building is assigned to exactly one column: the full combination of
    archetypes whose candidate lists contain it (from
    build_archetype_membership, the pivot of Stage I output). Column bars
    therefore partition the estate — their heights sum to the number of
    buildings assigned to at least one archetype.

    Layout (standard UpSet):
        top-right    : intersection-size bars (multi-archetype combinations
                       highlighted, single-archetype in grey)
        bottom-right : dot matrix showing which archetypes form each column
        bottom-left  : total set-size bars per archetype (archetype colours)

    Parameters
    ----------
    archetype_lists  : {archetype: [building_id, ...]} as returned by
                       build_archetype_membership.
    min_intersection : hide combinations smaller than this many buildings.
    save_path        : PNG path (300 dpi). Set None to skip saving.
    """
    sets = {a: set(b) for a, b in archetype_lists.items() if b}
    if not sets:
        print("  [warn] plot_archetype_upset: no archetype membership to plot.")
        return None

    # Rows: archetypes ordered by total set size (largest on top)
    archs = sorted(sets, key=lambda a: len(sets[a]), reverse=True)
    all_b = set().union(*sets.values())

    # Columns: exclusive combinations (each building in exactly one)
    combo_members = {}
    for b in all_b:
        combo = tuple(a for a in archs if b in sets[a])
        combo_members.setdefault(combo, set()).add(b)
    combos = [(c, m) for c, m in combo_members.items()
              if len(m) >= min_intersection]
    combos.sort(key=lambda cm: (len(cm[1]), len(cm[0])), reverse=True)
    if not combos:
        print("  [warn] plot_archetype_upset: nothing above min_intersection.")
        return None

    n_r, n_c = len(archs), len(combos)
    fig = plt.figure(figsize=(max(8.0, 2.6 + 0.55 * n_c),
                              2.6 + 0.45 * n_r + 2.4))
    gs = fig.add_gridspec(2, 2,
                          width_ratios=[1.5, max(3.0, 0.55 * n_c)],
                          height_ratios=[2.4, max(1.0, 0.45 * n_r)],
                          hspace=0.08, wspace=0.05)
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_mat = fig.add_subplot(gs[1, 1], sharex=ax_bar)
    ax_set = fig.add_subplot(gs[1, 0], sharey=ax_mat)

    # --- top-right: intersection sizes ---
    xs = np.arange(n_c)
    sizes = [len(m) for _, m in combos]
    cols = ['#d95f02' if len(c) > 1 else '#4d4d4d' for c, _ in combos]
    ax_bar.bar(xs, sizes, color=cols, width=0.6)
    for x, s in zip(xs, sizes):
        ax_bar.text(x, s + max(sizes) * 0.02, str(s),
                    ha='center', va='bottom', fontsize=8)
    ax_bar.set_ylabel('Buildings in intersection')
    ax_bar.spines[['top', 'right']].set_visible(False)
    ax_bar.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_bar.set_ylim(0, max(sizes) * 1.12)
    from matplotlib.patches import Patch
    ax_bar.legend(handles=[
        Patch(color='#4d4d4d', label='single archetype'),
        Patch(color='#d95f02', label='multiple archetypes (overlap)')],
        fontsize=8, frameon=False, loc='upper right')

    # --- bottom-right: membership dot matrix ---
    for j, (combo, _) in enumerate(combos):
        rows_in = [archs.index(a) for a in combo]
        rows_out = [i for i in range(n_r) if i not in rows_in]
        ax_mat.scatter([j] * len(rows_out), rows_out, s=45,
                       color='#dddddd', zorder=1)
        ax_mat.scatter([j] * len(rows_in), rows_in, s=55,
                       color='#333333', zorder=3)
        if len(rows_in) > 1:
            ax_mat.plot([j, j], [min(rows_in), max(rows_in)],
                        color='#333333', linewidth=1.6, zorder=2)
    ax_mat.set_xlim(-0.6, n_c - 0.4)
    ax_mat.set_ylim(n_r - 0.5, -0.5)          # largest set on top
    ax_mat.tick_params(left=False, labelleft=False,
                       bottom=False, labelbottom=False)
    for s in ax_mat.spines.values():
        s.set_visible(False)
    for i in range(n_r):
        if i % 2 == 0:
            ax_mat.axhspan(i - 0.5, i + 0.5, color='#f2f2f2', zorder=0)

    # --- bottom-left: total set sizes ---
    set_sizes = [len(sets[a]) for a in archs]
    ax_set.barh(np.arange(n_r), set_sizes,
                color=[ARCHETYPE_COLOURS.get(a, '#666') for a in archs],
                height=0.55)
    for i, s in enumerate(set_sizes):
        ax_set.text(s + max(set_sizes) * 0.03, i, str(s),
                    va='center', ha='left', fontsize=8)
    ax_set.invert_xaxis()                      # bars grow leftwards
    ax_set.set_yticks(np.arange(n_r))
    _abbrev = {
        'Fabric-limited':            'FL',
        'Electrification-candidate': 'EC',
        'Process-dominated':         'PD',
        'Flexibility candidate':     'FC',
        'Solar-suitable':            'SS',
    }
    ax_set.set_yticklabels([_abbrev.get(a, a) for a in archs], fontsize=9)
    ax_set.yaxis.tick_right()                  # names sit next to the matrix
    ax_set.set_xlabel('Set size', fontsize=9)
    ax_set.spines[['top', 'left']].set_visible(False)
    ax_set.set_xlim(max(set_sizes) * 1.35, 0)

    fig.suptitle('Archetype membership overlap (exclusive intersections)',
                 y=0.98, fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  UpSet figure saved to: {save_path}")
    if show:
        plt.show()
    return fig






# =============================================================================
# 8. ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # # Step 1 (optional): pick k empirically on the gas+elec Gower matrix.
    results, gaps = find_optimal_k(year=2022, method='average')

    # # Stage I + II at the theory-driven k = 15.
    cohorts, cohort_summary, impact, feat = run_stage2(
        year=2022, n_clusters=15, span_cap_m=300.0, min_samples=2)

    centroids = _load_centroids()
    render_cohort_map(cohorts, cohort_summary, centroids,
                      output_path='stage2_cohort_map.html')
    
    # summary, membership, feat = run_stage1(year=2022, n_clusters=15)
    # archetype_lists = build_archetype_membership(summary, membership)
    # plot_archetype_upset(archetype_lists, save_path='fig_archetype_upset.png')