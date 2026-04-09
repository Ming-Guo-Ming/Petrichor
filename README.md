# Petrichor

Petrichor is a Python-based processing toolkit for **Cosmic-Ray Neutron Sensing (CRNS)** data. It converts raw neutron counts into corrected neutron intensity, calibrated site parameters, and finally **soil moisture** estimates. The current version is designed around a **single-field / multi-station** workflow: one JSON configuration file controls one station run, and batch mode can process many station JSON files under the same input directory.

---

## 1. Main features

Petrichor currently supports the following core functions:

### 1.1 Data input and preprocessing
- Read local CSV input files
- Support flexible CSV headers (`header_rows = 0 / 1 / 2`)
- Replace missing values with `NaN`
- Round timestamps to hourly resolution and aggregate repeated records
- Fill missing hours to create a continuous hourly time series
- Normalize neutron counts by `NET` to produce hourly neutron counts (`N_CPH`)

### 1.2 Correction chain
- Pressure correction
- Humidity correction
- Incoming neutron intensity correction using NMDB data
- Above-ground biomass correction (if corresponding input is provided)
- Unified corrected neutron count output (`N_corr`)

### 1.3 Site parameter handling
- Read `Rc`, `beta_B`, `p0_ref`, `N0`, `sm_max` and other parameters from JSON
- Automatically calculate missing parameters when possible
- Write back calculated values such as `Rc`, `beta_B`, `p0_ref`, `N0`, and `sm_max` to the JSON config when they are missing

### 1.4 Calibration and soil moisture estimation
- N0 calibration using calibration samples
- Soil moisture estimation using the CRNS transfer equation
- Soil moisture smoothing (`SM_SMOOTH`)
- Effective sensing depth calculation (`D86_*` series)
- Quality control flags and final QC-filtered outputs

### 1.5 Outputs and visualization
- Full time-series export
- Daily, monthly, and yearly aggregated outputs
- Descriptive statistics export
- Run metadata export
- Detailed site summary text report
- Quicklook figure and additional diagnostic figures

---

## 2. Recommended project structure

Run Petrichor from the **project root directory**. A typical folder layout is:

```text
petrichor/
├── src/
│   ├── main.py
│   ├── processing.py
│   ├── analysis.py
│   └── visualization.py
├── input/
│   └── 001/
│       ├── site001.json
│       ├── input.csv
│       └── calibration.csv
├── cache/
└── output/
```

You can also organize multiple stations under `input/` and run them in batch mode.

---

## 3. Environment setup

### 3.1 Python version

Because the code uses syntax such as `str | None`, it is recommended to use:

- **Python 3.10 or above**

### 3.2 Create an environment

You can use either `conda` or `venv`.

#### Option A: conda

```bash
conda create -n petrichor python=3.10 -y
conda activate petrichor
pip install numpy pandas matplotlib scipy beautifulsoup4
```

#### Option B: venv

```bash
python3.10 -m venv petrichor-env
source petrichor-env/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib scipy beautifulsoup4
```

### 3.3 Main third-party packages

Petrichor currently relies mainly on:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `beautifulsoup4`

The following modules are from the Python standard library and do **not** need separate installation:

- `argparse`
- `json`
- `hashlib`
- `pathlib`
- `urllib`
- `datetime`
- `zoneinfo`

---

## 4. Input data and configuration

### 4.1 Input CSV

The input data file is usually a station time series such as `input.csv`.

Typical required columns include:

- `TIMESTAMP`
- `N`
- `NET`
- `PA_1`
- `TA_1`
- `RH_1`
- `BATT`

Optional columns may include:

- `PA_2`
- `TA_2`
- `RH_2`
- `PREC`
- `SWC_1`
- `SWC_2`
- `TA_i`
- `RH_i`
- biomass-related columns if AGB correction is used

### 4.2 Calibration CSV

If `N0` is **not already provided** in the JSON file, Petrichor needs a calibration CSV.

The calibration file is expected to contain at least the following fields:

- `DATE`
- `SWV`
- `DIST`
- `DEPTH_AVG`
- `LOC`

If `PROFILE` is not present, Petrichor can derive it from `LOC`.

### 4.3 JSON configuration file

Each station run is controlled by one JSON file, for example:

```text
input/001/site001.json
```

The JSON is organized into sections such as:

- `project`: output and cache paths
- `site` or `field`: site metadata
- `config`: processing parameters

A simplified example:

```json
{
  "project": {
    "output_dir": "output",
    "cache_dir": "cache"
  },
  "site": {
    "site_id": "001",
    "country": "UK"
  },
  "config": {
    "input_data_dir": ".",
    "calib_data_dir": ".",
    "data_source": "Local",
    "intensity_method": "mcjannet2023",
    "header_rows": 2,
    "net_min_seconds": 1800,
    "nmdb_station": "JUNG",
    "site_latitude": 53.2734,
    "site_longitude": 352.5113,
    "site_elevation": 73,
    "time_col": "TIMESTAMP",
    "N0": 2866,
    "bulk_density": 1.2,
    "lattice_water": 0.03,
    "soil_organic_carbon": 0.05
  }
}
```

### 4.4 Notes on configuration

- If `N0` is missing, Petrichor will try to calibrate it from the calibration CSV.
- If `Rc`, `beta_B`, `p0_ref`, or `sm_max` are missing, Petrichor may calculate them automatically and write them back into the JSON.
- If `data_source = "url"`, the input CSV can be downloaded automatically and cached locally.
- NMDB data are fetched and cached under the cache directory. The first run may require an internet connection if no cache is available.

---

## 5. How to run Petrichor

All commands below should be run from the **project root**.

### 5.1 Run a single site

```bash
python -m src.main --site input/001/site001.json
```

This command processes one station JSON file only.

### 5.2 Run all sites in batch mode

```bash
python -m src.main --site-dir input
```

This command recursively searches for all JSON files under `input/` and runs them one by one.

### 5.3 When no JSON exists yet

If the target JSON file does not exist, Petrichor can launch an **interactive config generator** and create a new config file step by step.

---

## 6. Output files

For each site, Petrichor writes results under:

```text
output/<site_id>/
```

Typical output structure:

```text
output/<site_id>/
├── data/
│   ├── timeseries_full.csv
│   ├── timeseries_daily_mean.csv
│   ├── timeseries_monthly_mean.csv
│   ├── timeseries_yearly_mean.csv
│   ├── stats_describe.csv
│   ├── run_metadata.json
│   ├── n0_debug_inputs.csv
│   └── nmdb_station_counts.txt
├── figures/
│   ├── quicklook.png
│   ├── pressure_series.png
│   ├── rhov_series.png
│   └── other diagnostic plots
└── logs/
    └── site_summary.txt
```

### Important outputs

- `timeseries_full.csv`: full processed time series
- `timeseries_daily_mean.csv`: daily aggregated results
- `timeseries_monthly_mean.csv`: monthly aggregated results
- `timeseries_yearly_mean.csv`: yearly aggregated results
- `stats_describe.csv`: descriptive statistics
- `run_metadata.json`: metadata and key parameters used in the run
- `site_summary.txt`: detailed textual summary of the site run
- `quicklook.png`: quick overview figure for counts, correction factors, soil moisture, and sensing depth

---

## 7. Typical workflow

A practical workflow is usually:

1. Prepare one site folder under `input/`
2. Put `input.csv` and, if needed, `calibration.csv` into that folder
3. Create or edit the site JSON file
4. Activate the Python environment
5. Run Petrichor with `python -m src.main --site ...`
6. Check outputs in `output/<site_id>/`
7. Review figures and `site_summary.txt`

---

## 8. Example commands

### Single station example

```bash
cd /path/to/petrichor
conda activate petrichor
python -m src.main --site input/001/site001.json
```

### Batch example

```bash
cd /path/to/petrichor
conda activate petrichor
python -m src.main --site-dir input
```

---

## 9. Notes

- Please make sure the JSON paths are correct.
- Please make sure the input CSV contains the required columns.
- If you want to skip calibration, provide a valid `N0` directly in the JSON file.
- If NMDB data cannot be downloaded, check your internet connection or reuse existing cached NMDB files.
- It is recommended to keep `input`, `cache`, and `output` separated clearly for reproducibility.

---

## 10. One-sentence summary

**Petrichor is a CRNS processing toolkit that takes raw neutron count data, applies correction and calibration steps, and exports soil moisture products, summary tables, and diagnostic figures in a reproducible workflow.**
