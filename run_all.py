"""
run_all.py
----------
Run every stage of the pipeline in order. Stops on the first error.

Usage:
    pixi run python run_all.py

Stages:
    1. metricsV3          — per-building per-year scalar metrics
    2. gis_mapping        — builds map_temporal.html
    3. non_spatial_metrics— builds cached CSVs and pickles
    4. non_spatial_plots  — builds non_spatial.html
    5. similarity_metrics — builds similarity_cache.pkl
    6. similarity_viewing — builds similarity.html
"""

import subprocess
import sys
import time

STAGES = [
    ('Building metrics',      'metricsV3.py'),
    ('Building map',          'gis_mapping.py'),
    ('Non-spatial metrics',   'non_spatial_metrics.py'),
    ('Non-spatial dashboard', 'non_spatial_plots.py'),
    ('Similarity metrics',    'similarity_metrics.py'),
    ('Similarity viewer',     'similarity_viewing.py'),
]


def run_stage(label, script):
    print(f'\n{"=" * 70}')
    print(f'  [{label}]  running {script}')
    print(f'{"=" * 70}')
    t0 = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f'\n>>> {script} FAILED after {elapsed:.1f}s (exit code {result.returncode})')
        return False
    print(f'\n>>> {script} OK ({elapsed:.1f}s)')
    return True


if __name__ == '__main__':
    t_start = time.time()

    for label, script in STAGES:
        ok = run_stage(label, script)
        if not ok:
            print(f'\nPipeline stopped. Fix the error above and re-run.')
            sys.exit(1)

    total = time.time() - t_start
    print(f'\n{"=" * 70}')
    print(f'  All stages complete in {total:.1f}s ({total/60:.1f} min)')
    print(f'{"=" * 70}')
    print('\nOutputs:')
    print('  map_temporal.html       — GIS map')
    print('  non_spatial.html        — distributions, CLDC, profiles, signatures')
    print('  similarity.html         — building similarity view')