# Estate-scale Operational Energy and Decarbonisation Analysis

This repository contains the Python code developed for an anonymised dissertation investigating estate-scale operational energy performance and decarbonisation using measured building energy data.

The workflow includes:

- Electricity and gas performance metrics;
- Temporal and load-shape analysis;
- Heating sensitivity analysis;
- Spatial visualisation;
- Mixed-variable building similarity using Gower distance;
- Hierarchical clustering and decarbonisation archetypes;
- Geographic delivery cohort identification; and
- Solar PV and peak-shaving analysis.

## Data availability

The building-level energy data and estate metadata used in this analysis are not included in this repository.

This includes building meter readings, building metadata, geographic coordinates and other estate-specific source files.

The analytical code is provided to document the methodology and enable reproduction where appropriate source data and permissions are available.

## Repository structure

`metricsV3.py`
Calculates the principal building-level and annual metrics, including EUI, baseload intensity, peak-load intensity, load factor, fuel-mix ratio, peak timing and heating sensitivity.

`non_spatial_metrics.py`
Calculates load profiles, load-duration metrics, distributions, and daily energy signatures.

`non_spatial_plots.py`
Produces the interactive non-spatial analysis dashboard.

`gis_mapping.py`
Produces the interactive spatial representation of building metrics.

`similarity_metrics.py`
Calculates pairwise building similarity using Gower, Euclidean, load-profile and peak-timing distances.

`similarity_viewing.py`
Produces the interactive similarity dashboard.

`economies_of_scale.py`
Implements the two-stage similarity and delivery-cohort framework.

`geocode_postcodes.py`
Utility used to derive building centroid coordinates from the restricted estate mapping data.

`run_all.py`
Runs the principal analysis pipeline.

## Requirements

Python 3.10 or later.

Dependencies are listed in `requirements.txt`.

## Running the analysis

Install dependencies:

    pip install -r requirements.txt

The source data must first be placed in the local directory structure described in `DATA.md`.

The main analysis can then be run using:

    python run_all.py

Individual components can also be run separately.

## Data confidentiality

No raw building-level energy data, estate metadata, postcodes, coordinates, meter identifiers or other restricted source data are distributed with this repository.

Generated files containing building-level results are also excluded from version control.

## Reproducibility

The repository contains the analytical procedures used in the dissertation.
Exact numerical reproduction requires access to the original source data.

## Anonymous submission

Author-identifying information has intentionally been omitted from this repository for anonymous assessment.
