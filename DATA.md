# Data

## Data source and attribution

The building energy data structure used in this analysis follows the Cambridge Estates Building Energy Archive developed by Max Langtry and made available through the EECi GitHub repository:

`EECi/Cambridge-Estates-Building-Energy-Archive`

The original archive can be cloned using:

```bash
gh repo clone EECi/Cambridge-Estates-Building-Energy-Archive
```

## Data availability

The building-level datasets used in this dissertation are not distributed with this repository.

The analysis uses measured electricity and gas consumption, building metadata, floor-area information, weather data and geographic information. Some of these data contain building-level or estate-specific information and are therefore excluded from the public repository.

The code is provided to document the analytical methodology. Exact reproduction of the dissertation results requires authorised access to the original source datasets.

## Building energy data

The principal building energy dataset should be arranged locally as:

```text
repo/
└── building_data/
    └── processed_data/
        ├── building_floor_roof_areas.csv
        │
        ├── UCam_Building_b0/
        │   ├── electricity/
        │   │   ├── 2018.csv
        │   │   ├── 2019.csv
        │   │   └── ...
        │   └── gas/
        │       ├── 2018.csv
        │       ├── 2019.csv
        │       └── ...
        │
        ├── UCam_Building_b1/
        │   ├── electricity/
        │   └── gas/
        │
        └── ...
```

Electricity files are expected to contain:

```text
datetime
equipment load [kWh]
```

Gas files are expected to contain:

```text
datetime
heating load [kWh]
```

The `building_floor_roof_areas.csv` file provides building floor-area information used to normalise energy consumption.

## Building metadata

The analysis also requires a local metadata workbook:

```text
Building_Mapping.xlsx
```

The analysis reads the `Conversion` worksheet.

This file contains estate-specific building information and is not included in the repository.

Fields used by the analysis include building ID, gross internal area, construction date, function, tenure and other building characteristics.

## Geographic data

Building centroid coordinates are stored locally as:

```text
building_centroids.csv
```

with the fields:

```text
building_id
postcode
latitude
longitude
```

This file is not distributed because it contains building-level geographic information.

`geocode_postcodes.py` documents the procedure used to generate the centroid file from the restricted building metadata.

## Weather data

Hourly outdoor temperature data used for the heating-sensitivity analysis should be arranged as:

```text
repo/
└── aux_data/
    └── MetOffice Weather Data/
        └── processed_data/
            └── bedford/
                ├── 2018.csv
                ├── 2019.csv
                └── ...
```

Each weather file is expected to contain:

```text
datetime
air_temperature [degC]
```

## Solar generation data

Where the solar analysis is used, hourly modelled solar-generation profiles are stored locally under:

```text
repo/
└── aux_data/
    └── RenewablesNinja Generation Data/
        └── processed_data/
            ├── 2018.csv
            ├── 2019.csv
            └── ...
```

Solar source and validation data containing site-specific identifiers are not distributed with the repository.

## Configuration

All local data locations are defined in:

```text
config.py
```

Paths are relative to the repository root so that no user-specific absolute file paths are required.

## Generated files

Running the analysis generates intermediate and final files including CSV, pickle and HTML outputs.

These may contain building-level metrics, identifiers, metadata or geographic information and are therefore excluded from version control using `.gitignore`.

Examples include:

```text
metrics_results.csv
metrics_results_temporal.csv
metadata_table.csv
distribution_summary.csv
cldc_summary.csv
daily_profiles.pkl
daily_signatures.pkl
similarity_cache.pkl
gis_map.html
non_spatial.html
similarity.html
stage2_cohort_map.html
```
