"""
run_all.py
----------
Run the main dissertation analysis pipeline in order.

The source data are not distributed with this repository.
See DATA.md for the required local data structure.

Usage:
    python run_all.py

Stages:
    1. metricsV3            — building-level and annual energy metrics
    2. non_spatial_metrics  — profiles, distributions and energy signatures
    3. gis_mapping          — interactive spatial metric map
    4. non_spatial_plots    — interactive non-spatial dashboard
    5. similarity_metrics   — similarity matrices and clustering cache
    6. similarity_viewing   — interactive similarity dashboard
    7. economies_of_scale   — decarbonisation archetypes and delivery cohorts
"""

import subprocess
import sys
import time


STAGES = [
    ('Building metrics',          'metricsV3.py'),
    ('Non-spatial metrics',       'non_spatial_metrics.py'),
    ('GIS map',                   'gis_mapping.py'),
    ('Non-spatial dashboard',     'non_spatial_plots.py'),
    ('Similarity metrics',        'similarity_metrics.py'),
    ('Similarity viewer',         'similarity_viewing.py'),
    ('Delivery cohort analysis',  'Economies_of_scaleV2.py'),
]


def run_stage(label, script):
    print(f'\n{"=" * 70}')
    print(f'  [{label}]  running {script}')
    print(f'{"=" * 70}')

    t0 = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(
            f'\n>>> {script} FAILED after {elapsed:.1f}s '
            f'(exit code {result.returncode})'
        )
        return False

    print(f'\n>>> {script} OK ({elapsed:.1f}s)')
    return True


if __name__ == '__main__':

    t_start = time.time()

    for label, script in STAGES:
        if not run_stage(label, script):
            print('\nPipeline stopped because a stage failed.')
            sys.exit(1)

    total = time.time() - t_start

    print(f'\n{"=" * 70}')
    print(f'  All stages complete in {total:.1f}s ({total / 60:.1f} min)')
    print(f'{"=" * 70}')

    print('\nPrincipal generated outputs:')
    print('  metrics_results.csv            — mean building metrics')
    print('  metrics_results_temporal.csv   — annual building metrics')
    print('  gis_map.html                   — spatial metrics dashboard')
    print('  non_spatial.html               — non-spatial analysis dashboard')
    print('  similarity_cache.pkl           — pairwise similarity results')
    print('  similarity.html                — similarity dashboard')
    print('  stage2_cohort_map.html         — decarbonisation delivery cohorts')

    print('\nNote: generated outputs may contain restricted building-level')
    print('information and should not be committed to the public repository.')
