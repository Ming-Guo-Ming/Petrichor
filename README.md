# Petrichor

**Petrichor** is a Python tool for processing **Cosmic-Ray Neutron Sensor (CRNS)** observations into corrected neutron counts and field-scale soil-moisture estimates.

*Authors: Ming Guo, Rafael Rosolem, Miguel Rico-Ramirez and Shams Rahman*  
*Institution: University of Bristol, UK*

## Overview

Petrichor provides a consistent, site-based workflow for CRNS data processing and visualisation. It is designed to make CRNS processing easier to configure, inspect, and reproduce while retaining the scientific correction and calibration steps required for soil-moisture estimation.

Petrichor can:

- standardise and aggregate CRNS time-series data to hourly intervals;
- correct neutron counts for atmospheric pressure, atmospheric water vapour, incoming cosmic-ray intensity, and above-ground biomass;
- calibrate or reuse the site-specific `N0` parameter;
- convert corrected neutron counts into volumetric soil moisture;
- apply quality control and estimate uncertainty;
- fill eligible meteorological gaps using ERA5-Land;
- calculate effective sensing depth;
- generate hourly, daily, monthly, and yearly data products;
- export diagnostic figures, run metadata, and a site summary.

For the scientific background, configuration reference, input formats, equations, ERA5-Land setup, and complete output descriptions, see the **[Petrichor Wiki](../../wiki)**.

> **Development status:** Petrichor is under active development. Users should inspect configuration values, warnings, quality-control flags, and diagnostic outputs before using results in research or operational decisions.

## Installation

Python 3.10 or later is recommended. Download or clone this repository, open a terminal in the project directory, and create an environment:

```bash
conda create -n petrichor python=3.10 -y
conda activate petrichor
python -m pip install numpy pandas matplotlib scipy beautifulsoup4
```

To use ERA5-Land meteorological gap filling, also install:

```bash
python -m pip install "cdsapi>=0.7.7" xarray netcdf4
```

ERA5-Land downloads require each user to create their own Copernicus Climate Data Store account, accept the relevant dataset terms, and configure a personal `.cdsapirc` file. Never upload API credentials to GitHub. See **[ERA5-Land configuration](../../wiki/ERA5-Land-Configuration)** in the Wiki for the complete setup procedure.

## Running Petrichor

Petrichor requires:

1. a site time-series CSV containing CRNS observations;
2. a site JSON file containing metadata and processing settings.

An optional calibration CSV is required when `N0` must be calibrated, and an optional AGB CSV can be supplied for dynamic above-ground biomass correction.

Place the files in a site folder such as:

```text
input/
└── USA011/
    ├── USA011.json
    ├── USA011.csv
    └── USA011_calibration.csv
```

Run one site from the project root:

```bash
python -m src.main --site input/USA011/USA011.json
```

To process every site folder under `input/`:

```bash
python -m src.main --site-dir input
```

See the Wiki pages for **[input time-series data](../../wiki/Input-Timeseries-Data)**, **[metadata configuration](../../wiki/Metadata-Configuration)**, and **[N0 calibration](../../wiki/N0-Calibration)** before preparing a new site.

## Outputs

Results are written to a directory named after the site:

```text
output/USA011/
├── data/
├── figures/
└── logs/
```

- `data/` contains the full processed time series, daily/monthly/yearly summaries, run metadata, statistics, and NMDB diagnostics.
- `figures/` contains correction, soil-moisture, uncertainty, and quick-look plots.
- `logs/` contains a human-readable site summary.

For a description of every file and important output column, see **[Final Outputs](../../wiki/Final-Outputs)**.

## Example

The repository includes **USA011 — Santa Rita Creosote** as an example site. Its input data, metadata, calibration file, and example outputs show the expected Petrichor directory structure and provide a reference for setting up a new station.

## Documentation

The **[Petrichor Wiki](../../wiki)** contains the complete documentation, including:

- CRNS theory and applications;
- metadata and input-file requirements;
- neutron correction methods;
- `N0` calibration;
- ERA5-Land gap filling;
- dynamic AGB correction;
- neutron-to-soil-moisture conversion;
- output files and quality-control guidance;
- step-by-step usage instructions.

## Authors

Petrichor is developed by **Ming Guo**, **Rafael Rosolem**, **Miguel Rico-Ramirez** and **Shams Rahmanat** the **University of Bristol, UK**.

Petrichor builds on methods and software developed by the wider CRNS community, including the open-source **[crspy](https://github.com/danpower101/crspy)** and **[crspy-lite](https://github.com/Joe-Wagstaff/crspy-lite)** projects. Please consult the Wiki for scientific references and acknowledgements.

