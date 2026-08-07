# _*_ coding:UTF-8 _*_
"""
main.py - command-line entry point and run orchestrator for Petrichor

This module drives the end-to-end Petrichor workflow for one site JSON file
or for a directory of site JSON files.

Current responsibilities include:
- loading and validating site/field configuration JSON
- resolving local input paths or downloading remote input data
- reading and preprocessing raw station data
- loading or fetching NMDB reference counts with caching
- computing or reusing Rc, beta_B, p0_ref, rhov0_ref, jung_ref, and N0
- applying the full neutron-count correction chain
- running soil-moisture conversion, QC, uncertainty, exports, and figures
- writing updated configuration values back to JSON when required

Expected JSON sections:
- project: output, cache, and figure paths
- site / field: site metadata such as ID, country, timezone
- config: processing options, column names, calibration settings, and physics parameters

This file is intentionally orchestration-focused.
Core physics and processing routines live in processing.py,
exports live in analysis.py, and plotting lives in visualization.py.
"""

import os
import sys
import json
import argparse
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path

from src.processing import *
from src.analysis import run_analysis_exports
from src.visualization import *
import src.processing as processing


# ==============================================================================
# [0] HELPERS
# ==============================================================================

def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(p: Path, obj: dict) -> None:
    # Strip transient runtime-only keys so they are never persisted to the
    # site JSON.  "_agb_daily_lookup" is rebuilt from the external AGB file
    # on every run and must not be baked into the config.
    if isinstance(obj, dict) and isinstance(obj.get("config"), dict):
        cfg_copy = {k: v for k, v in obj["config"].items()
                    if k != "_agb_daily_lookup"}
        obj = {**obj, "config": cfg_copy}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _cfg_get(cfg: dict, key: str, default=None):
    """
    Unified config getter for Petrichor JSON.

    Priority:
    1) cfg[key]
    2) cfg["config"][key]
    3) cfg["field"][key]
    4) cfg["project"][key]
    """
    if not isinstance(cfg, dict):
        return default

    if key in cfg:
        return cfg.get(key, default)

    for section in ("config", "site", "field", "project"):
        sec = cfg.get(section, None)
        if isinstance(sec, dict) and key in sec:
            return sec.get(key, default)

    return default

def _resolve_config_path(field_json_path: Path, raw_path: str | None) -> Path | None:
    """
    Resolve a file path from config JSON.

    Rules:
    - empty / null -> None
    - absolute path -> use as is
    - relative path -> resolve relative to the JSON file directory
    """
    if raw_path in (None, "", "null"):
        return None

    p = Path(str(raw_path)).expanduser()
    if p.is_absolute():
        return p.resolve()

    return (field_json_path.parent / p).resolve()

def _download_file(url: str, dest_path: Path) -> Path:
    """
    Download a file from URL to dest_path.
    If already exists, reuse it.
    """
    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"[CACHE] Reusing downloaded file: {dest_path.name}")
        return dest_path

    print(f"[IO] Downloading: {url}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[IO] Saved downloaded file to: {dest_path}")
    return dest_path

def _read_url_input_data(cfg: dict) -> tuple[pd.DataFrame, dict, Path]:
    """
    Download remote CSV into cache_dir/url_downloads/ and read it
    with the same reader used for local input.
    Returns: df, units_dict, downloaded_path
    """
    data_url = str(_cfg_get(cfg, "data_url", "")).strip()
    if not data_url:
        raise ValueError("[IO] data_source='url' but data_url is empty.")

    cache_dir = Path(_cfg_get(cfg, "cache_dir", "cache")).expanduser().resolve()
    url_cache_dir = cache_dir / "url_downloads"

    # use hash so repeated URLs do not collide
    suffix = Path(data_url.split("?")[0]).suffix or ".csv"
    fname = hashlib.sha256(data_url.encode("utf-8")).hexdigest()[:16] + suffix
    local_path = url_cache_dir / fname

    downloaded = _download_file(data_url, local_path)
    df, units_dict = read_local_input_data(str(downloaded))
    return df, units_dict, downloaded

def _load_or_fetch_nmdb_cached(cfg: dict, station: str, startdate: str, enddate: str, site_out: Path) -> pd.DataFrame | None:
    """
    Load NMDB counts from cache if available; otherwise fetch and cache them.
    Also writes a copy to site_out/data/nmdb_station_counts.txt for station outputs.
    """
    cache_dir = Path(_cfg_get(cfg, "cache_dir", "cache")).expanduser().resolve()
    nmdb_cache_dir = cache_dir / "nmdb"
    nmdb_cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = nmdb_cache_dir / f"{station}_{startdate}_{enddate}.csv"

    if cache_file.exists():
        print(f"[CACHE] Using cached NMDB file: {cache_file.name}")
        nmdb_df = pd.read_csv(cache_file)
    else:
        print(f"[NMDB] No cache found. Fetching NMDB for {station} {startdate} -> {enddate}")
        nmdb_dict = nmdb_get(
            startdate=startdate,
            enddate=enddate,
            station=station,
            default_dir=str(site_out)
        )
        nmdb_df = pd.DataFrame({
            "DT": list(nmdb_dict.keys()),
            "N_COUNT": list(nmdb_dict.values())
        })
        nmdb_df.to_csv(cache_file, index=False)
        print(f"[CACHE] NMDB cached to: {cache_file}")

    # always write a station-local copy for reproducibility
    local_nmdb_path = site_out / "data" / "nmdb_station_counts.txt"
    local_nmdb_path.parent.mkdir(parents=True, exist_ok=True)
    nmdb_df.to_csv(local_nmdb_path, sep=";", index=False, header=False)

    nmdb_df["DT"] = pd.to_datetime(nmdb_df["DT"], errors="coerce")
    nmdb_df["N_COUNT"] = pd.to_numeric(nmdb_df["N_COUNT"], errors="coerce")
    nmdb_df = nmdb_df.dropna(subset=["DT"])

    return nmdb_df

def _resolve_config_dir(field_json_path: Path, raw_path: str | None) -> Path | None:
    """Resolve a directory path from config JSON."""
    if raw_path in (None, "", "null"):
        return None

    p = Path(str(raw_path)).expanduser()
    if p.is_absolute():
        return p.resolve()

    return (field_json_path.parent / p).resolve()


def _discover_site_files(field_json_path: Path, cfg: dict) -> tuple[Path, Path | None]:
    """
    Discover one input CSV and one calibration CSV for a site.

    Priority:
    1) config.input_data_filepath / config.calib_data_filepath
    2) config.input_data_dir / config.calib_data_dir with auto-discovery
    """
    # ---------- first try explicit filepaths ----------
    input_fp = _resolve_config_path(field_json_path, _cfg_get(cfg, "input_data_filepath", None))
    calib_fp = _resolve_config_path(field_json_path, _cfg_get(cfg, "calib_data_filepath", None))

    if input_fp is not None:
        if not input_fp.exists():
            raise FileNotFoundError(f"Input file not found: {input_fp}")
        if calib_fp is not None and not calib_fp.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_fp}")
        return input_fp, calib_fp

    # ---------- fallback to directory discovery ----------
    input_dir = _resolve_config_dir(field_json_path, _cfg_get(cfg, "input_data_dir", "."))
    calib_dir = _resolve_config_dir(field_json_path, _cfg_get(cfg, "calib_data_dir", "."))

    if input_dir is None or not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_candidates = sorted(input_dir.glob("*.csv"))

    if calib_dir is not None and calib_dir.exists():
        calib_candidates = sorted(
            p for p in calib_dir.glob("*.csv")
            if ("calib" in p.name.lower() or "calibration" in p.name.lower())
        )
    else:
        calib_candidates = []

    # remove calibration-like and AGB files from input candidates.
    # The dedicated AGB series lives in "<input_stem>_AGB.csv" and must not
    # be mistaken for the main input CSV.
    input_candidates = [
        p for p in input_candidates
        if ("calib" not in p.name.lower()
            and "calibration" not in p.name.lower()
            and not p.name.lower().endswith("_agb.csv"))
    ]

    if len(input_candidates) == 0:
        raise FileNotFoundError(f"No input CSV found in: {input_dir}")
    if len(input_candidates) > 1:
        raise ValueError(
            f"Multiple input CSV candidates found in {input_dir}: "
            + ", ".join(p.name for p in input_candidates)
        )

    input_path = input_candidates[0]

    if len(calib_candidates) == 0:
        calib_path = None
    elif len(calib_candidates) == 1:
        calib_path = calib_candidates[0]
    else:
        raise ValueError(
            f"Multiple calibration CSV candidates found in {calib_dir}: "
            + ", ".join(p.name for p in calib_candidates)
        )

    return input_path, calib_path

# ------------------get config helper-----------------------------------------------------------------

def _prompt_with_default(prompt: str, default=None, allowed: list[str] | None = None):
    while True:
        if default is None:
            raw = input(f"{prompt}: ").strip()
            if raw == "":
                print("[INPUT] This field is required.")
                continue
        else:
            raw = input(f"{prompt} [{default}]: ").strip()
            if raw == "":
                raw = default

        if allowed is not None:
            allowed_lower = [a.lower() for a in allowed]
            if str(raw).lower() not in allowed_lower:
                print(f"[INPUT] Please choose one of: {', '.join(allowed)}")
                continue

        return raw


def _create_field_json_interactive(field_json_path: Path) -> dict:
    print(f"[INFO] Field config not found: {field_json_path}")
    print("[INFO] Launching interactive field-config generator...")

    input_data_filepath = _prompt_with_default("config.input_data_filepath", "input.csv")
    output_dir = _prompt_with_default("project.output_dir", "output")
    calib_data_filepath = _prompt_with_default("config.calib_data_filepath", "calibration.csv")
    cache_dir = _prompt_with_default("project.cache_dir", "cache")

    field_id = _prompt_with_default("field.field_id", field_json_path.stem)
    country = _prompt_with_default("field.country", "UK")
    timezone = _prompt_with_default("field.timezone", "")

    data_source = _prompt_with_default(
        "config.data_source (local/url)",
        "local",
        allowed=["local", "url"]
    ).lower()

    data_url = _prompt_with_default("config.data_url", "")

    intensity_method = _prompt_with_default(
        "config.intensity_method (mcjannet2023/hawdon2014)",
        "mcjannet2023",
        allowed=["mcjannet2023", "hawdon2014"]
    ).lower()

    site_latitude = float(_prompt_with_default("config.site_latitude", None))
    site_longitude = float(_prompt_with_default("config.site_longitude", None))
    site_elevation = float(_prompt_with_default("config.site_elevation", 0.0))

    net_min_seconds = int(_prompt_with_default("config.net_min_seconds", 1800))
    nmdb_station = str(_prompt_with_default("config.nmdb_station", "JUNG")).upper()

    rhov0_ref = float(_prompt_with_default("config.rhov0_ref", 0))
    bulk_density = float(_prompt_with_default("config.bulk_density", 1.43))
    lattice_water = float(_prompt_with_default("config.lattice_water", 0.0))
    soil_organic_carbon = float(_prompt_with_default("config.soil_organic_carbon", 0.0))

    config = {
        "project": {
            "output_dir": output_dir,
            "cache_dir": cache_dir
        },
        "site": {
            "site_id": field_id,
            "country": country,
            "timezone": timezone if str(timezone).strip() else None
        },
        "config": {
            "input_data_filepath": input_data_filepath,
            "calib_data_filepath": calib_data_filepath,
            "data_source": data_source,
            "data_url": data_url,
            "intensity_method": intensity_method,
            "missing_value": [-999, -99, "NA", "N/A", ""],
            "net_min_seconds": net_min_seconds,
            "nmdb_station": nmdb_station,

            "a0": 0.0808,
            "a1": 0.372,
            "a2": 0.115,

            "site_latitude": site_latitude,
            "site_longitude": site_longitude,
            "site_elevation": site_elevation,
            "rc": None,
            "beta_B": None,
            "p0_ref": None,
            "jung_ref": None,
            "rhov0_ref": rhov0_ref,
            "N0": None,
            "bulk_density": bulk_density,
            "sm_max": None,
            "lattice_water": lattice_water,
            "soil_organic_carbon": soil_organic_carbon,

            "smwindow": 12,

            "accuracy": 0.01,
            "calib_start_time": "09:00:00",
            "calib_end_time": "17:00:00",

            "required_vars": [
                "TIMESTAMP", "N", "NET", "PA_1", "TA_1", "RH_1", "BATT"
            ],
            "additional_vars": [
                "PA_2", "TA_2", "RH_2", "PREC", "SWC_1", "SWC_2", "TA_i", "RH_i"
            ]
        }
    }

    field_json_path.parent.mkdir(parents=True, exist_ok=True)
    _save_json(field_json_path, config)

    print(f"[INFO] New field config created: {field_json_path}")
    return config


def _discover_site_jsons(root_dir: Path) -> list[Path]:
    """
    Find all site JSON files under a directory.

    Rule:
    - recursively search for *.json
    - skip hidden files/folders
    - return sorted paths for stable batch execution
    """
    root_dir = root_dir.resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Site directory not found: {root_dir}")

    json_files = []
    for p in root_dir.rglob("*.json"):
        parts = p.relative_to(root_dir).parts
        if any(part.startswith(".") for part in parts):
            continue
        json_files.append(p.resolve())

    json_files = sorted(json_files)

    if not json_files:
        raise FileNotFoundError(f"No site JSON files found under: {root_dir}")

    return json_files

def _has_valid_n0(cfg: dict) -> bool:
    """Return True if N0 exists and can be interpreted as a valid number."""
    value = _cfg_get(cfg, "N0", None)
    if value in (None, "", "null"):
        return False
    try:
        num = float(value)
        return np.isfinite(num)
    except Exception:
        return False


# -------- core runner ----------------------------------------------------------

def run_one_site(project_root: Path, field_json_path: Path):
    print(r"""
            _        _      _                
           | |      (_)    | |               
 _ __   ___| |_ _ __ _  ___| |__   ___  _ __ 
| '_ \ / _ \ __| '__| |/ __| '_ \ / _ \| '__|
| |_) |  __/ |_| |  | | (__| | | | (_) | |   
| .__/ \___|\__|_|  |_|\___|_| |_|\___/|_|   
| |                                         
|_|                                         

                   Ming Guo; Rafael Rosolem
                   University of Bristol, UK
                   MMXXV --> date
    """)

# ==============================================================================
# [1] SETUP
# ==============================================================================

    workdir = project_root.resolve()
    print(f"[INFO] Project root: {workdir}")

    if not field_json_path.exists():
        config = _create_field_json_interactive(field_json_path)
    else:
        config = _load_json(field_json_path)

    validate_processing_config(config)

    output_dir = Path(_cfg_get(config, "output_dir", workdir / "output")).expanduser()
    cache_dir = Path(_cfg_get(config, "cache_dir", workdir / "cache")).expanduser()

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    input_path, calib_path = _discover_site_files(field_json_path, config)

    site_id = (
        _cfg_get(config, "site_id", None)
        or _cfg_get(config, "field_id", None)
        or field_json_path.stem
    )

    print(f"[INFO] Using config: {field_json_path}")
    print(f"[INFO] Site ID: {site_id}")
    print(f"[INFO] Input file: {input_path}")
    print(f"[INFO] Calibration file: {calib_path}")
    print(f"[INFO] Output dir: {output_dir}")

    if input_path is None:
        raise ValueError("config.input_data_filepath is missing in JSON.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

# ==============================================================================
# [2] LOOP OVER STATIONS
# ==============================================================================

    site_out = output_dir / str(site_id)
    (site_out / "figures").mkdir(parents=True, exist_ok=True)
    (site_out / "logs").mkdir(parents=True, exist_ok=True)
    (site_out / "data").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"[INFO] Processing site: {site_id}")
    print("=" * 80)

# ==============================================================================
# [3] RAW DATA PRE-PROCESSING
# ==============================================================================

    data_source = str(_cfg_get(config, "data_source", "local")).lower()

    if data_source == "url":
        df, units_dict, downloaded_path = _read_url_input_data(config)
        input_path = downloaded_path
        print(f"[INFO] URL input cached at: {input_path}")
    else:
        print(f"[INFO] Local input: {input_path}")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df, units_dict = read_local_input_data(str(input_path))

    print("[INFO] Data loading complete.")

    # Replace missing values first
    time_col = TIME_COL
    timestamp_format = DEFAULT_TIMESTAMP_FORMAT
    df = replace_missing_with_nan(df, _cfg_get(config, "missing_value", -999))
    df = mask_invalid_relative_humidity(df, config)

    # Load AGB from a dedicated per-site file located next to the input CSV.
    # AGB is no longer stored inline in the input file.  The expected file is
    # "<input_stem>_AGB.csv" (e.g. CSP1_AGB.csv) with columns DATE and AGB_KGM2;
    # it holds the (possibly sparse) measured AGB points.  These points are
    # carried forward and held constant until the next measurement appears
    # (step interpolation) by processing.resolve_daily_agb.
    #
    # The lookup is stored under config["config"]["_agb_daily_lookup"] only as
    # a transient runtime value; _save_json strips it so it is never written
    # back into the site JSON.
    agb_path = input_path.parent / f"{input_path.stem}_AGB.csv"
    # Optional explicit override via config.agb_file (resolved next to input).
    _agb_cfg = _cfg_get(config, "agb_file", None)
    if _agb_cfg:
        _p = Path(_agb_cfg).expanduser()
        agb_path = _p if _p.is_absolute() else (input_path.parent / _p)

    if agb_path.exists():
        _agb_df = pd.read_csv(agb_path)
        _date_col = next(
            (c for c in _agb_df.columns
             if str(c).upper() in ("DATE", "DATE_AGB", "YYYY-MM-DD")),
            None,
        )
        _val_col = next(
            (c for c in _agb_df.columns if str(c).upper() == "AGB_KGM2"),
            None,
        )
        if _date_col is not None and _val_col is not None:
            # AGB file dates are written as DD/MM/YYYY (or ISO YYYY-MM-DD,
            # which dayfirst does not affect), so parse with dayfirst=True
            # to avoid misreading e.g. "03/06/2016" as 6 March 2016.
            _raw_dates = pd.to_datetime(_agb_df[_date_col], errors="coerce", dayfirst=True)
            _raw_agb = pd.to_numeric(_agb_df[_val_col], errors="coerce")
            _invalid_agb = _agb_df[_val_col].notna() & _raw_agb.isna()
            if _invalid_agb.any():
                raise ValueError(
                    f"[AGB] {_invalid_agb.sum()} non-numeric value(s) found in "
                    f"{agb_path.name}:AGB_KGM2."
                )
            if (_raw_agb.dropna() < 0).any():
                raise ValueError(
                    f"[AGB] Negative AGB is not allowed in {agb_path.name}:AGB_KGM2."
                )
            _lookup_raw = (
                pd.DataFrame({"date": _raw_dates, "agb": _raw_agb})
                .dropna(subset=["date", "agb"])
                .groupby("date")["agb"]
                .max()
            )
            if "config" not in config:
                config["config"] = {}
            config["config"]["_agb_daily_lookup"] = {
                d.strftime("%Y-%m-%d"): float(v) for d, v in _lookup_raw.items()
            }
            print(f"[AGB] Loaded {len(_lookup_raw)} AGB points from {agb_path.name}.")
        else:
            raise ValueError(
                f"[AGB] {agb_path.name} must contain a date column and the exact "
                "column AGB_KGM2. AGB values must be supplied in kg/m²."
            )
    else:
        print(f"[AGB][INFO] No AGB file found ({agb_path.name}); "
              f"AGB correction will be skipped.")

    # Before ERA5-Land gap filling, only hard-check non-meteorological variables.
    # Meteorological columns may be absent and created by ERA5-Land later.
    if _cfg_get(config, "required_vars", None) is not None:
        meteo_fillable = {
            _cfg_get(config, "pressure_col", "PA_1"),
            _cfg_get(config, "temp_col", "TA_1"),
            _cfg_get(config, "rh_col", "RH_1"),
            _cfg_get(config, "ah_col", "AH_1"),
        }

        core_required = [
            c for c in list(_cfg_get(config, "required_vars", []))
            if c not in meteo_fillable
        ]

        core_check_cfg = json.loads(json.dumps(config))

        if "config" not in core_check_cfg:
            core_check_cfg["config"] = {}

        core_check_cfg["config"]["required_vars"] = core_required

        check_variables_in_data(df, core_check_cfg)

    # Round timestamps and aggregate
    df = round_to_hour(
        df,
        time_col,
        timestamp_format=timestamp_format,
    )

    # Ensure continuous hourly timestamps
    df = continuous_hourly_timestamps(
        df,
        time_col,
        timestamp_format=timestamp_format,
    )

    # Normalize neutron counts by NET
    df, neutron_col_by_net, net_meta = normalize_by_net(
        df,
        config,
        n_col="N",
        net_col="NET",
        out_col="N_CPH",
    )

    # Fill missing meteorological data using ERA5-Land,
    # but only where neutron counts exist.
    df = gapfill_meteorology_from_era5_land(
        df=df,
        cfg=config,
        site_out=site_out,
        neutron_col=neutron_col_by_net,
    )
    df = mask_invalid_relative_humidity(df, config)

    # After ERA5-Land filling, run the full required-variable check.
    if _cfg_get(config, "required_vars", None) is not None:
        check_variables_in_data(df, config)

# ==============================================================================
# [4] DATA IMPORTS AND INITIAL CALCULATION
# ==============================================================================

    nmdb_station = _cfg_get(config, "nmdb_station", "JUNG")
    df[time_col] = parse_timestamp_series(df[time_col], timestamp_format=timestamp_format)
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    if len(df) == 0:
        raise ValueError(f"No valid timestamps after parsing for site {site_id}.")

    startdate = df[time_col].iloc[0]
    enddate = df[time_col].iloc[-1]

    nmdb_df = _load_or_fetch_nmdb_cached(
        cfg=config,
        station=nmdb_station,
        startdate=startdate.strftime("%Y-%m-%d"),
        enddate=enddate.strftime("%Y-%m-%d"),
        site_out=site_out
    )

    if nmdb_df is not None:
        print("\n[INFO] NMDB head:")
        print(nmdb_df.head())
    else:
        print("[WARN] NMDB dataframe is None.")

    # 5.2 Calculate cutoff rigidity if not in config
    Rc = None
    if _cfg_get(config, "rc", None) in (None, "", "null"):
        print("[INFO] No rc in config. Calculating from latitude/longitude...")
        lat = _cfg_get(config, "site_latitude", None)
        lon = _cfg_get(config, "site_longitude", None)
        if lat is None or lon is None:
            print("[ERROR] site_latitude/longitude missing. Cannot compute rc.")
            Rc = None
        else:
            Rc = rc_retrieval(latitude=float(lat), longitude=float(lon))

            # write rc back to field JSON
            try:
                if "config" not in config:
                    config["config"] = {}
                config["config"]["rc"] = round(float(Rc), 2)
                _save_json(field_json_path, config)
                print(f"[INFO] Updated {field_json_path.name} with rc={Rc:.3f}")
            except Exception as e:
                print(f"[WARN] Could not write rc to field JSON: {e}")

    else:
        try:
            Rc = float(_cfg_get(config, "rc"))
        except Exception:
            Rc = _cfg_get(config, "rc")
        print(f"[INFO] Using rc from config: {Rc}")

# 5.3 Decide neutron column.
    if neutron_col_by_net in df.columns:
        neutron_col = neutron_col_by_net
    elif "N_CPH" in df.columns:
        neutron_col = "N_CPH"
    elif "MOD" in df.columns:
        neutron_col = "MOD"
    elif NEUTRON_COL in df.columns:
        neutron_col = NEUTRON_COL
    else:
        raise ValueError(
            "[NEUTRON] No valid neutron count column found. Expected one of: "
            f"{neutron_col_by_net}, N_CPH, MOD, {NEUTRON_COL}. "
            f"Available columns: {list(df.columns)}"
        )

    # Create a per-run copy for calculated/transient values. The raw neutron
    # input convention is fixed as N; when NET is available, the resolved N_CPH
    # column is passed explicitly and is never written into the site JSON.
    run_cfg = json.loads(json.dumps(config))
    if "config" not in run_cfg:
        run_cfg["config"] = {}
    # Store config file path so processing functions can resolve relative paths
    run_cfg["config"]["_config_path"] = str(field_json_path)

    print(f"[INFO] Neutron column: {neutron_col}")

    # 5.4 Calculate beta coefficient / p0_ref if not in config
    beta_B = _cfg_get(run_cfg, "beta_B", None)
    p0_ref = _cfg_get(run_cfg, "p0_ref", None)

    try:
        if beta_B is not None:
            beta_B = float(beta_B)
        if p0_ref is not None:
            p0_ref = float(p0_ref)
    except Exception:
        beta_B = None
        p0_ref = None

    try:
        elev = float(_cfg_get(run_cfg, "site_elevation", 0.0))
        lat = _cfg_get(run_cfg, "site_latitude", None)

        if (beta_B is None or p0_ref is None) and Rc is not None and lat is not None:
            beta_B, p0_ref = betacoeff(
                latitude=float(lat),
                elevation=elev,
                Rc=float(Rc),
            )
            beta_B = float(beta_B)
            p0_ref = round(float(p0_ref), 3)
            # write back to field JSON
            if "config" not in config:
                config["config"] = {}
            config["config"]["beta_B"] = float(beta_B)
            config["config"]["p0_ref"] = float(p0_ref)

            # sync to runtime config
            if "config" not in run_cfg:
                run_cfg["config"] = {}
            run_cfg["config"]["beta_B"] = float(beta_B)
            run_cfg["config"]["p0_ref"] = float(p0_ref)

            try:
                _save_json(field_json_path, config)
                print(f"[INFO] Updated {field_json_path.name} with beta_B={beta_B:.5f}, p0_ref={p0_ref:.3f}")
            except Exception as e:
                print(f"[WARN] Could not write beta_B/p0_ref to field JSON: {e}")

    except Exception as e:
        print(f"[WARN] betacoeff failed: {e}")

# ==============================================================================
# [5] CORRECTION FACTOR CHAIN
# ==============================================================================

    # Build NMDB reference (median) if available, unless config already provides one
    jung_ref = _cfg_get(run_cfg, "jung_ref", None)
    if jung_ref in (None, "", "null"):
        jung_ref = None
        if nmdb_df is not None and "N_COUNT" in nmdb_df.columns:
            try:
                jung_ref = float(pd.to_numeric(nmdb_df["N_COUNT"], errors="coerce").median())
            except Exception:
                jung_ref = None

    # Read absolute humidity baseline from config (do not auto-compute here)
    rhov0_ref = _cfg_get(run_cfg, "rhov0_ref", None)
    if rhov0_ref is None or str(rhov0_ref).strip().lower() in ("", "none", "null", "nan"):
        print("[HUM] rhov0_ref missing in field JSON. Humidity correction will be skipped.")
        rhov0_ref = None
    else:
        rhov0_ref = float(rhov0_ref)

    # Single-call unified corrections: Ncorr = N * f_p * f_h * f_i * f_v
    df_corr = apply_all_corrections_combined(
        df=df,
        cfg=run_cfg,
        nmdb_df=nmdb_df,
        jung_ref=jung_ref,
        beta_B=beta_B,
        p0_ref=p0_ref,
        rhov0_ref=rhov0_ref,
        neutron_col=neutron_col,
        out_col_int="N_corr_int",
        out_col_agb="N_corr"
    )

    # crspy-lite style corrected neutron count used for calibration, QC, and final SM
    df_corr["MOD_CORR"] = pd.to_numeric(df_corr["N_corr"], errors="coerce").round()

    # keep a raw-count alias for uncertainty logic and easier comparison with crspy-lite outputs
    if "MOD" not in df_corr.columns:
        if "N" in df_corr.columns:
            df_corr["MOD"] = pd.to_numeric(df_corr["N"], errors="coerce")
        else:
            df_corr["MOD"] = pd.to_numeric(df_corr[neutron_col], errors="coerce")
# ==============================================================================
# [6] CALIBRATION
# ==============================================================================


    N0_cfg = _cfg_get(config, "N0", None)

    if N0_cfg in (None, "", "null"):
        print("[CAL] N0 not found in config.")

        if calib_path is None:
            print("[CAL][WARN] config.calib_data_filepath is missing. Cannot calibrate N0.")
        elif not calib_path.exists():
            print(f"[CAL][WARN] Calibration file not found: {calib_path}")
        else:
            try:
                print(f"[CAL] Calibrating site-specific N0 using: {calib_path.name}")

                N0 = n0_calibration(
                    corrected_data=df_corr.copy(),
                    country=str(_cfg_get(config, "country", "site")),
                    site_id=str(site_id),
                    defineaccuracy=float(_cfg_get(config, "accuracy", 0.01)),
                    calib_start_time=str(_cfg_get(config, "calib_start_time", "09:00:00")),
                    calib_end_time=str(_cfg_get(config, "calib_end_time", "17:00:00")),
                    config_data=config,
                    calib_data_filepath=str(calib_path),
                    default_dir=str(site_out),
                    sm_calc_method=str(_cfg_get(config, "theta_method", "desilets")).lower()
                )

                if "config" not in config:
                    config["config"] = {}
                config["config"]["N0"] = int(round(float(N0)))

                if "config" not in run_cfg:
                    run_cfg["config"] = {}
                run_cfg["config"]["N0"] = int(round(float(N0)))

                try:
                    _save_json(field_json_path, config)
                    print(f"[CAL] Updated {field_json_path.name} with N0={int(round(float(N0)))}")
                except Exception as e:
                    print(f"[CAL][WARN] Could not write calibrated N0 to config JSON: {e}")

                print(f"[CAL] Site-specific N0 = {int(round(float(N0)))}")

            except Exception as e:
                print(f"[CAL][WARN] Site-specific calibration failed: {e}")

    else:
        try:
            N0_cfg = float(N0_cfg)
            print(f"[CAL] Using N0 from config: {N0_cfg:.3f}")
        except Exception:
            print(f"[CAL] Using N0 from config: {N0_cfg}")

        if "config" not in run_cfg:
            run_cfg["config"] = {}
        run_cfg["config"]["N0"] = N0_cfg

# ==============================================================================
# [7] SOIL MOISTURE BOUNDS
# ==============================================================================

    bd_raw = _cfg_get(run_cfg, "bulk_density", None)
    bd_available = False
    bd_for_sm = None

    if bd_raw not in (None, "", "null"):
        try:
            bd_for_sm = float(bd_raw)
            bd_available = np.isfinite(bd_for_sm)
        except Exception:
            bd_available = False

    sm_max_cfg = _cfg_get(run_cfg, "sm_max", None)

    if not bd_available:
        print(
            "[SWC][WARN] bulk_density is missing. "
            "sm_max will not be calculated, and SM/D86 will be skipped."
        )

        if "config" not in run_cfg:
            run_cfg["config"] = {}

        # Keep sm_max as None unless the user explicitly provided a value.
        if sm_max_cfg in (None, "", "null"):
            run_cfg["config"]["sm_max"] = None

    elif sm_max_cfg in (None, "", "null"):
        print("[SWC] No sm_max found in field JSON. Calculating from bulk density...")
        sm_max = round(sm_max_calc(bd=bd_for_sm), 4)

        if "config" not in config:
            config["config"] = {}
        config["config"]["sm_max"] = sm_max

        if "config" not in run_cfg:
            run_cfg["config"] = {}
        run_cfg["config"]["sm_max"] = sm_max

        try:
            _save_json(field_json_path, config)
            print(f"[INFO] Updated {field_json_path.name} with sm_max={sm_max:.4f}")
        except Exception as e:
            print(f"[WARN] Could not write sm_max to field JSON: {e}")

    else:
        sm_max = float(sm_max_cfg)

        if "config" not in run_cfg:
            run_cfg["config"] = {}
        run_cfg["config"]["sm_max"] = sm_max

        print(f"[INFO] Using sm_max from config: {sm_max:.4f}")

# ==============================================================================
# [8] QC AND THETA PIPELINE
# ==============================================================================

    swc_crns_col = str(_cfg_get(run_cfg, "swc_crns_col", "SWC_CRNS"))

    df_corr = run_crns_qc_and_theta_pipeline(
        df=df_corr,
        cfg=run_cfg,
        neutron_col=neutron_col,
        corr_n_col="MOD_CORR",
        swc_crns_col=swc_crns_col
    )
     
# ==============================================================================
# [9] ANALYSIS AND EXPORT
# ==============================================================================

    run_analysis_exports(
        df=df_corr,
        time_col=time_col,
        site_out=site_out,
        site_id=site_id,
        run_cfg=run_cfg,
        project_root=project_root,
        input_path=input_path,
        field_json_path=field_json_path,
        Rc=Rc,
        beta_B=beta_B,
        p0_ref=p0_ref,
        rhov0_ref=rhov0_ref,
        jung_ref=jung_ref,
        nmdb_station=nmdb_station,
        neutron_col=neutron_col,
        decimals=3,
    )

# ==============================================================================
# [10] VISUALIZATION
# ==============================================================================

    run_all_visualizations(
        df=df_corr,
        cfg=run_cfg,
        site_out=site_out,
        time_col=time_col,
        neutron_col=neutron_col
    )

    print(f"[INFO] Done. Outputs at: {site_out}")

    return {
        "site_id": str(site_id),
        "folder_name": field_json_path.parent.name,
        "n0_available": _has_valid_n0(run_cfg),
    }
        
# -------- CLI -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="petrichor runner (single-site or batch-site config)")
    ap.add_argument("--site", help="single site json path, e.g. input/siteA/siteA.json")
    ap.add_argument("--site-dir", help="directory containing many site folders, e.g. input")
    args = ap.parse_args()

    project_root = Path.cwd().resolve()

    if args.site and args.site_dir:
        raise ValueError("Please use either --site or --site-dir, not both.")

    if args.site:
        site_arg = args.site.strip()
        site_json_path = project_root / site_arg if not Path(site_arg).is_absolute() else Path(site_arg)
        run_one_site(project_root, site_json_path.resolve())
        return

    if args.site_dir:
        site_dir_arg = args.site_dir.strip()
        site_dir_path = project_root / site_dir_arg if not Path(site_dir_arg).is_absolute() else Path(site_dir_arg)

        site_jsons = _discover_site_jsons(site_dir_path)

        print(f"[BATCH] Found {len(site_jsons)} site JSON file(s).")

        ok_count = 0
        fail_count = 0
        failed_folders = []
        n0_missing_folders = []

        for site_json_path in site_jsons:
            print("\n" + "#" * 90)
            print(f"[BATCH] Running site config: {site_json_path}")
            print("#" * 90)

            try:
                result = run_one_site(project_root, site_json_path.resolve())
                ok_count += 1

                if not result["n0_available"]:
                    n0_missing_folders.append(result["folder_name"])

            except Exception as e:
                fail_count += 1
                failed_folders.append(site_json_path.parent.name)
                print(f"[BATCH][ERROR] Failed for {site_json_path.name}: {e}")

        print("\n" + "=" * 90)
        print(f"[BATCH] Finished. Success: {ok_count}, Failed: {fail_count}")

        if failed_folders:
            print(f"[BATCH] Failed folders: {', '.join(failed_folders)}")
        else:
            print("[BATCH] Failed folders: none")

        if n0_missing_folders:
            print(f"[BATCH] N0 missing: {', '.join(n0_missing_folders)}")
        else:
            print("[BATCH] N0 missing: none")

        print("=" * 90)
        return

    print("Usage:")
    print("  python -m src.main --site input/001/001.json")
    print("  python -m src.main --site-dir input")


if __name__ == "__main__":
    main()
