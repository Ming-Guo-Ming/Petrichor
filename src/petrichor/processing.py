# _*_ coding: UTF-8 _*_
"""
processing.py - core data processing, correction, calibration, QC, and summary tools for Petrichor

This module contains the main scientific and preprocessing routines used by Petrichor.

Main function groups:
- input parsing and preprocessing
- timestamp handling and hourly resampling
- missing-value replacement and NET-based neutron normalization
- NMDB retrieval and cutoff-rigidity lookup
- pressure, humidity, intensity, and vegetation corrections
- N0 calibration utilities and weighting functions
- soil-moisture conversion and uncertainty propagation
- effective sensing depth calculation
- quality-control flagging
- site summary text generation

The module is written so that most scientific calculations are implemented here,
while main.py only coordinates workflow and file management.

Important implementation note:
some routines preserve compatibility with legacy crspy-lite style logic,
but the surrounding workflow, configuration structure, output naming,
and helper functions are Petrichor-specific.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import urllib.request
from scipy.interpolate import griddata

# Fixed Petrichor input and physics conventions. Input files are intentionally
# required to follow this format; these are not site-configurable settings.
HEADER_ROWS = 1
TIME_COL = "TIMESTAMP"
DEFAULT_TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M"
NEUTRON_COL = "N"
PARTICLE_DENSITY = 2.65
ERA5_FILL_ABSOLUTE_HUMIDITY = False
ERA5_AREA_BUFFER_DEG = 0.1
APPLY_SM_SMOOTHING = True

def _cfg_get(cfg: dict, key: str, default=None):
    """
    Unified config getter for Petrichor JSON:

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


def _cfg_section(cfg: dict, section: str) -> dict:
    """Safe section getter."""
    val = cfg.get(section, {})
    return val if isinstance(val, dict) else {}


# Reference-monitor metadata used by the incoming-neutron correction. Keeping
# this small registry in processing.py avoids a separate station-data module.
# JUNG values follow the current NMDB Jungfraujoch IGY station specification.
NMDB_STATION_METADATA = {
    "JUNG": {
        "latitude": 46.55,
        "elevation": 3570.0,
        "rc": 4.5,
    },
}


def _resolve_nmdb_station_metadata(cfg: dict) -> dict[str, float | str]:
    """Resolve reference-monitor metadata without using site values as defaults.

    Explicit JSON values override the built-in entry. An unlisted NMDB station
    remains usable when all three reference values are supplied explicitly.
    """
    station = str(_cfg_get(cfg, "nmdb_station", "")).strip().upper()
    built_in = NMDB_STATION_METADATA.get(station, {})
    resolved = {
        "latitude": _cfg_get(cfg, "nmdb_latitude", built_in.get("latitude")),
        "elevation": _cfg_get(cfg, "nmdb_elevation", built_in.get("elevation")),
        "rc": _cfg_get(cfg, "nmdb_rc", built_in.get("rc")),
    }

    missing = [name for name, value in resolved.items()
               if _is_missing_config_value(value)]
    if missing:
        raise ValueError(
            f"[INT] NMDB station '{station or '<empty>'}' has no complete reference "
            f"metadata. Supply {', '.join('nmdb_' + name for name in missing)}. "
            "Site latitude, elevation and Rc are intentionally not used as fallbacks."
        )

    try:
        latitude = float(resolved["latitude"])
        elevation = float(resolved["elevation"])
        rc = float(resolved["rc"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[INT] Invalid reference metadata for NMDB station '{station}'."
        ) from exc

    if not np.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError(f"[INT] Invalid NMDB latitude for '{station}': {latitude}")
    if not np.isfinite(elevation):
        raise ValueError(f"[INT] Invalid NMDB elevation for '{station}': {elevation}")
    if not np.isfinite(rc) or rc < 0.0:
        raise ValueError(f"[INT] Invalid NMDB cutoff rigidity for '{station}': {rc}")

    return {
        "station": station,
        "latitude": latitude,
        "elevation": elevation,
        "rc": rc,
    }


def _is_missing_config_value(value) -> bool:
    """Return True for missing config values such as None, null, nan, or empty string."""
    if value is None:
        return True

    if isinstance(value, str):
        text = value.strip().lower()
        return text in ("", "none", "null", "nan", "na", "n/a")

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    return False


def _bulk_density_or_none(cfg: dict) -> float | None:
    """
    Return bulk density as a finite float, or None if it is missing/invalid.

    This allows Petrichor to continue processing neutron corrections even when
    soil-moisture-related calculations cannot be performed.
    """
    value = _cfg_get(cfg, "bulk_density", _cfg_get(cfg, "bd", None))

    if _is_missing_config_value(value):
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if not np.isfinite(value):
        return None

    return value


def validate_processing_config(cfg: dict) -> None:
    """Validate processing parameters before any site data are processed.

    Optional values may be absent, but a value that is present must be
    physically and numerically valid. Invalid configuration is fatal because
    silently continuing could produce plausible-looking scientific results.
    """
    def _optional_float(key: str) -> float | None:
        value = _cfg_get(cfg, key, None)
        if _is_missing_config_value(value):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"[CONFIG] '{key}' must be numeric; got {value!r}.") from exc
        if not np.isfinite(number):
            raise ValueError(f"[CONFIG] '{key}' must be finite; got {value!r}.")
        return number

    n0 = _optional_float("N0")
    if n0 is not None:
        if n0 <= 0:
            raise ValueError(f"[CONFIG] 'N0' must be greater than 0; got {n0}.")
        if not n0.is_integer():
            raise ValueError(f"[CONFIG] 'N0' must be an integer; got {n0}.")

    bulk_density = _optional_float("bulk_density")
    if bulk_density is not None and not 0.0 < bulk_density < PARTICLE_DENSITY:
        raise ValueError(
            "[CONFIG] 'bulk_density' must satisfy "
            f"0 < bulk_density < {PARTICLE_DENSITY}; got {bulk_density}."
        )

    smwindow_raw = _cfg_get(cfg, "smwindow", 12)
    try:
        smwindow = float(smwindow_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[CONFIG] 'smwindow' must be a positive integer; got {smwindow_raw!r}."
        ) from exc
    if not np.isfinite(smwindow) or smwindow <= 0 or not smwindow.is_integer():
        raise ValueError(
            f"[CONFIG] 'smwindow' must be a positive integer; got {smwindow_raw!r}."
        )

    latitude = _optional_float("site_latitude")
    if latitude is not None and not -90.0 <= latitude <= 90.0:
        raise ValueError(
            f"[CONFIG] 'site_latitude' must be between -90 and 90; got {latitude}."
        )

    longitude = _optional_float("site_longitude")
    if longitude is not None and not -180.0 <= longitude <= 180.0:
        raise ValueError(
            f"[CONFIG] 'site_longitude' must be between -180 and 180; got {longitude}."
        )

    rc = _optional_float("rc")
    if rc is not None and rc < 0.0:
        raise ValueError(f"[CONFIG] 'rc' must be non-negative; got {rc}.")

    sm_min = _optional_float("qc_theta_min")
    if sm_min is None:
        sm_min = 0.0
    if sm_min < 0.0:
        raise ValueError(f"[CONFIG] 'qc_theta_min' must be non-negative; got {sm_min}.")

    sm_max = _optional_float("sm_max")
    if sm_max is not None:
        if sm_max < 0.0:
            raise ValueError(f"[CONFIG] 'sm_max' must be non-negative; got {sm_max}.")
        if sm_max < sm_min:
            raise ValueError(
                f"[CONFIG] 'sm_max' ({sm_max}) must be >= 'qc_theta_min' ({sm_min})."
            )

    for key, default in (("a0", 0.0808), ("a1", 0.372), ("a2", 0.115)):
        value = _optional_float(key)
        value = default if value is None else value
        if value <= 0.0:
            raise ValueError(f"[CONFIG] '{key}' must be greater than 0; got {value}.")


def mask_invalid_relative_humidity(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Replace RH observations outside 0--100 percent with missing values."""
    out = df.copy()
    configured = str(_cfg_get(cfg, "rh_col", "RH_1"))
    rh_columns = [
        col for col in dict.fromkeys((configured, "RH_1", "RH_2", "RH_i", "RH"))
        if col in out.columns
    ]
    for col in rh_columns:
        numeric = pd.to_numeric(out[col], errors="coerce")
        invalid = numeric.notna() & ~numeric.between(0.0, 100.0, inclusive="both")
        count = int(invalid.sum())
        if count:
            print(
                f"[QC][RH] {count} value(s) outside 0-100% in '{col}' "
                "were replaced with NaN."
            )
        out[col] = numeric.mask(invalid)
    return out

# ==============================================================================
# IO HELPERS
# ==============================================================================

def read_local_input_data(
    input_path: str,
    header_rows: int = HEADER_ROWS,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Read a CSV using Petrichor's one-row-header convention.

    ``header_rows`` remains as a low-level compatibility parameter for older
    callers, but normal Petrichor runs always use the code default of one row:
    the first row contains column names and data starts on the second row.
    """
    path = Path(input_path)
    print(f"[IO] Reading local input: {path} (header_rows={header_rows})")
    units_dict: Dict[str, str] = {}

    if header_rows >= 2:
        headers = pd.read_csv(path, nrows=2, header=None)
        col_names = headers.iloc[0].tolist()
        units = headers.iloc[1].tolist()
        units_dict = dict(zip(col_names, units))
        df = pd.read_csv(path, skiprows=2, names=col_names)
    elif header_rows == 1:
        headers = pd.read_csv(path, nrows=1, header=None)
        col_names = headers.iloc[0].tolist()
        df = pd.read_csv(path, skiprows=1, names=col_names)
    else:
        df = pd.read_csv(path)

    print(f"[IO] Local CSV loaded: rows={len(df)}, cols={len(df.columns)}")
    return df, units_dict

def parse_timestamp_series(
    series: pd.Series,
    timestamp_format: str | list[str] | tuple[str, ...] | None = DEFAULT_TIMESTAMP_FORMAT,
) -> pd.Series:
    """
    Parse timestamps using Petrichor's preferred format with robust fallbacks.

    Supported behavior
    ------------------
    - Normal runs first try ``%d/%m/%Y %H:%M``.
    - Direct callers may supply one format or an ordered list of formats.
    - Always fall back to a small set of common formats used by Petrichor.
    - Finally use pandas automatic parsing with day-first interpretation for
      only the values that remain unresolved.

    This allows a column to contain the preferred format alongside values such
    as:
    - "%Y-%m-%d %H:%M:%S"
    - "%d/%m/%Y %H:%M"
    """

    s = pd.Series(series, copy=True)

    # If already datetime-like, keep it.
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    raw = s.astype("string").str.strip()

    raw = raw.replace({
        "": pd.NA,
        "nan": pd.NA,
        "NaN": pd.NA,
        "none": pd.NA,
        "None": pd.NA,
        "null": pd.NA,
        "NULL": pd.NA,
        "NaT": pd.NA,
        "noData": pd.NA,
        "NODATA": pd.NA,
    })

    # Build candidate format list.
    candidate_formats: list[str] = []

    if isinstance(timestamp_format, (list, tuple)):
        for fmt in timestamp_format:
            if fmt not in (None, "", "auto", "AUTO", "Auto") and fmt not in candidate_formats:
                candidate_formats.append(fmt)
    elif timestamp_format not in (None, "", "auto", "AUTO", "Auto"):
        candidate_formats.append(str(timestamp_format))

    # Common Petrichor formats to try as fallbacks.
    common_formats = [
        DEFAULT_TIMESTAMP_FORMAT,
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
    ]

    for fmt in common_formats:
        if fmt not in candidate_formats:
            candidate_formats.append(fmt)

    # Parse incrementally: fill only the still-missing rows each time.
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    for fmt in candidate_formats:
        mask = raw.notna() & parsed.isna()
        if not mask.any():
            break
        parsed_try = pd.to_datetime(raw.loc[mask], format=fmt, errors="coerce")
        parsed.loc[mask] = parsed_try

    # Final fallback for anything still not parsed.
    mask = raw.notna() & parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(raw.loc[mask], errors="coerce", dayfirst=True)

    return parsed

def _sum_with_nan(series: pd.Series):
    """Sum values while keeping all-NaN groups as NaN instead of 0."""
    s = pd.to_numeric(series, errors="coerce")
    return s.sum(min_count=1)


def _mean_with_nan(series: pd.Series):
    """Mean of numeric values with non-numeric entries coerced to NaN."""
    s = pd.to_numeric(series, errors="coerce")
    return s.mean()


def round_to_hour(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    timestamp_format: str | None = DEFAULT_TIMESTAMP_FORMAT,
) -> pd.DataFrame:
    """
    Parse timestamps, align records to hourly bins, and aggregate columns
    with variable-specific rules.

    Rules
    -----
    - Neutron-count-like variables are summed within each hour.
    - NET is summed within each hour so N_CPH is computed from hourly totals.
    - Pressure / temperature / relative humidity / battery are averaged.
    - Other numeric columns default to hourly mean.
    - Hours with no records are not created here; they are filled later by
      continuous_hourly_timestamps().
    """
    out = df.copy()

    out[time_col] = parse_timestamp_series(
        out[time_col],
        timestamp_format=timestamp_format
    )

    bad = out[time_col].isna().sum()
    if bad:
        print(f"[WARN] {bad} timestamps could not be parsed and will be dropped.")

    out = out.dropna(subset=[time_col]).copy()

    if out.empty:
        raise ValueError(f"[TIME] No valid timestamps left after parsing column '{time_col}'.")

    out[time_col] = out[time_col].dt.floor("h")

    # Save only genuinely non-numeric string columns before numeric conversion
    # so fields such as DATE_AGB survive aggregation. Some pandas versions keep
    # numeric CSV columns as object dtype after missing-value replacement; those
    # columns must not be preserved here or the later merge would create
    # duplicate names such as N_x/N_y and PA_1_x/PA_1_y.
    preserved_str_cols: dict[str, pd.Series] = {}
    for col in out.columns:
        if col == time_col:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            numeric_candidate = pd.to_numeric(out[col], errors="coerce")
            if numeric_candidate.notna().any():
                continue
            # Keep first non-NaN value per hourly bin.
            preserved_str_cols[col] = (
                out.groupby(time_col, sort=False)[col]
                .first()
            )

    # Convert non-time columns to numeric where possible.
    for col in out.columns:
        if col == time_col:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Columns that must be summed within each hour.
    sum_cols = {
        "N",
        "NET",
        "MOD",
    }

    # Columns that must be averaged within each hour.
    mean_cols = {
        "PA_1", "PA_2",
        "PRESS",
        "TA_1", "TA_2", "TA_i", "TEMP",
        "RH_1", "RH_2", "RH_i", "RH",
        "BATT",
        "GroundWaterLevel",
    }

    agg_map = {}
    for col in out.columns:
        if col == time_col:
            continue

        if not pd.api.types.is_numeric_dtype(out[col]):
            continue

        if out[col].notna().sum() == 0:
            continue

        if col in sum_cols:
            agg_map[col] = _sum_with_nan
        elif col in mean_cols:
            agg_map[col] = _mean_with_nan
        else:
            agg_map[col] = _mean_with_nan

    if not agg_map:
        raise ValueError(
            "[AGG] No numeric columns available for hourly aggregation. "
            "Please check missing-value markers, timestamp parsing, and input headers."
        )

    hourly = out.groupby(time_col, as_index=False).agg(agg_map)

    # Re-attach preserved string columns (e.g. DATE_AGB).
    for col, series in preserved_str_cols.items():
        series.name = col
        hourly = hourly.merge(series.reset_index(), on=time_col, how="left")

    return hourly

def replace_missing_with_nan(df: pd.DataFrame, missing_value) -> pd.DataFrame:
    """Replace blanks and placeholder values (scalar or list) with NA."""
    out = df.replace(r'^\s*$', pd.NA, regex=True)
    out = out.replace(missing_value, pd.NA)
    return out


def continuous_hourly_timestamps(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    timestamp_format: str | None = DEFAULT_TIMESTAMP_FORMAT,
) -> pd.DataFrame:
    """Fill hourly index to ensure continuity (gaps become NA rows)."""
    out = df.copy()
    out[time_col] = parse_timestamp_series(
        out[time_col],
        timestamp_format=timestamp_format
    )
    out = out.dropna(subset=[time_col]).sort_values(time_col)

    if out.empty:
        return out

    idx = pd.date_range(out[time_col].iloc[0], out[time_col].iloc[-1], freq="h")
    out = out.set_index(time_col).reindex(idx).reset_index().rename(columns={"index": time_col})
    return out


# ==============================================================================
# ERA5-LAND METEOROLOGICAL GAP FILLING
# ==============================================================================

def _parse_site_timezone(cfg: dict):
    """
    Return the site timezone used to convert ERA5-Land UTC timestamps to
    Petrichor's local, timezone-naive timestamps.

    Accepted config values
    ----------------------
    - site.timezone / config.timezone as an IANA name, e.g. "Europe/Berlin"
    - numeric UTC offset, e.g. 0, 1, 10, -5
    - empty / missing -> UTC
    """
    tz_value = _cfg_get(cfg, "timezone", None)

    if tz_value in (None, "", "null"):
        return timezone.utc

    if isinstance(tz_value, (int, float, np.integer, np.floating)):
        return timezone(timedelta(hours=float(tz_value)))

    tz_text = str(tz_value).strip()

    try:
        return timezone(timedelta(hours=float(tz_text)))
    except Exception:
        pass

    try:
        return ZoneInfo(tz_text)
    except Exception:
        print(f"[ERA5][WARN] Invalid timezone '{tz_text}'. Falling back to UTC.")
        return timezone.utc


def _days_in_month(year: int, month: int) -> list[str]:
    """Return all valid day strings for a given month."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    return [f"{d:02d}" for d in range(1, int(end.day) + 1)]


def _month_starts_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return month-start timestamps covering start -> end inclusive."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_convert("UTC").tz_localize(None)

    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_convert("UTC").tz_localize(None)

    start_m = start_ts.to_period("M").to_timestamp()
    end_m = end_ts.to_period("M").to_timestamp()

    return list(pd.date_range(start_m, end_m, freq="MS"))


def _download_era5_land_month(
    *,
    cfg: dict,
    site_id: str,
    year: int,
    month: int,
    variables: list[str],
    cache_dir: Path,
) -> Path:
    """
    Download one month of ERA5-Land data for a small box around the site.
    The file is cached and reused on later runs.
    """
    lat = float(_cfg_get(cfg, "site_latitude"))
    lon = float(_cfg_get(cfg, "site_longitude"))
    buffer_deg = ERA5_AREA_BUFFER_DEG

    var_key = "_".join(sorted(variables)).replace("/", "-")
    era5_dir = Path(cache_dir) / "era5_land"
    era5_dir.mkdir(parents=True, exist_ok=True)

    nc_path = era5_dir / f"{site_id}_{year}_{month:02d}_{var_key}.nc"

    if nc_path.exists():
        print(f"[ERA5][CACHE] Using cached ERA5-Land file: {nc_path.name}")
        return nc_path

    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "[ERA5] cdsapi is not installed. Install it with: pip install cdsapi"
        ) from exc

    area = [
        min(90.0, lat + buffer_deg),
        max(-180.0, lon - buffer_deg),
        max(-90.0, lat - buffer_deg),
        min(180.0, lon + buffer_deg),
    ]

    request = {
        "variable": variables,
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": _days_in_month(year, month),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    print(f"[ERA5] Downloading ERA5-Land {year}-{month:02d} for {site_id}")
    client = cdsapi.Client()

    try:
        client.retrieve("reanalysis-era5-land", request, str(nc_path))
    except Exception as modern_error:
        # Backward-compatible request style used by older CDS/crspy examples.
        legacy_request = dict(request)
        legacy_request.pop("data_format", None)
        legacy_request.pop("download_format", None)
        legacy_request["format"] = "netcdf"

        print("[ERA5][WARN] Modern CDS request failed. Retrying legacy format=netcdf request.")

        try:
            client.retrieve("reanalysis-era5-land", legacy_request, str(nc_path))
        except Exception as legacy_error:
            raise RuntimeError(
                f"[ERA5] Download failed. Modern error: {modern_error}; "
                f"legacy error: {legacy_error}"
            ) from legacy_error

    return nc_path


def _read_era5_land_point(nc_path: Path, cfg: dict) -> pd.DataFrame:
    """Read one cached ERA5-Land NetCDF and extract the nearest grid point."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "[ERA5] xarray is not installed. Install it with: pip install xarray netcdf4"
        ) from exc

    lat = float(_cfg_get(cfg, "site_latitude"))
    lon = float(_cfg_get(cfg, "site_longitude"))
    site_tz = _parse_site_timezone(cfg)

    with xr.open_dataset(nc_path) as ds:
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        lat_name = "latitude"
        lon_name = "longitude"

        lon_for_sel = lon
        try:
            lon_values = ds[lon_name].values
            if np.nanmin(lon_values) >= 0 and lon_for_sel < 0:
                lon_for_sel = lon_for_sel % 360.0
        except Exception:
            pass

        point = ds.sel(
            {lat_name: lat, lon_name: lon_for_sel},
            method="nearest",
        )

        era = pd.DataFrame()
        era["ERA5_UTC"] = pd.to_datetime(point[time_name].values, utc=True)
        era["TIMESTAMP_LOCAL"] = (
            era["ERA5_UTC"]
            .dt.tz_convert(site_tz)
            .dt.tz_localize(None)
            .dt.floor("h")
        )

        if "t2m" in point:
            era["TA_ERA5_C"] = np.asarray(point["t2m"].values).reshape(-1) - 273.15

        if "d2m" in point:
            era["TD_ERA5_C"] = np.asarray(point["d2m"].values).reshape(-1) - 273.15

        if "sp" in point:
            era["PA_ERA5_HPA"] = np.asarray(point["sp"].values).reshape(-1) / 100.0

    if {"TA_ERA5_C", "TD_ERA5_C"}.issubset(era.columns):
        es_t = _saturation_vapor_pressure_hpa(era["TA_ERA5_C"])
        es_td = _saturation_vapor_pressure_hpa(era["TD_ERA5_C"])

        rh = 100.0 * es_td / es_t
        era["RH_ERA5_PERCENT"] = pd.to_numeric(rh, errors="coerce").clip(
            lower=0.0,
            upper=100.0,
        )

        # Absolute humidity in g m^-3, available if later needed.
        era["AH_ERA5_GM3"] = _absolute_humidity(es_td, era["TA_ERA5_C"])

    return era


def _download_era5_land_timeseries(
    *,
    cfg: dict,
    site_id: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    variables: list[str],
    cache_dir: Path,
) -> Path:
    """
    Download ERA5-Land hourly time-series data for one site.

    This uses the CDS time-series catalogue:
        reanalysis-era5-land-timeseries

    It is much faster for one site over a long period than downloading
    monthly spatial NetCDF files from reanalysis-era5-land.
    """
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "[ERA5] cdsapi is not installed. Install it with: pip install cdsapi"
        ) from exc

    lat = float(_cfg_get(cfg, "site_latitude"))
    lon = float(_cfg_get(cfg, "site_longitude"))

    start_utc = pd.Timestamp(start_utc)
    end_utc = pd.Timestamp(end_utc)

    if start_utc.tzinfo is not None:
        start_utc = start_utc.tz_convert("UTC").tz_localize(None)

    if end_utc.tzinfo is not None:
        end_utc = end_utc.tz_convert("UTC").tz_localize(None)

    start_date = start_utc.strftime("%Y-%m-%d")
    end_date = end_utc.strftime("%Y-%m-%d")

    var_key = "_".join(sorted(variables)).replace("/", "-")
    era5_dir = Path(cache_dir) / "era5_land_timeseries"
    era5_dir.mkdir(parents=True, exist_ok=True)

    csv_path = era5_dir / f"{site_id}_{start_date}_{end_date}_{var_key}.csv"

    if csv_path.exists():
        print(f"[ERA5][CACHE] Using cached ERA5-Land time-series file: {csv_path.name}")
        return csv_path

    request = {
        "variable": variables,
        "location": {
            "latitude": lat,
            "longitude": lon,
        },
        "date": f"{start_date}/{end_date}",
        "data_format": "csv",
    }

    print(
        "[ERA5] Downloading ERA5-Land time-series "
        f"{start_date} -> {end_date} for {site_id}"
    )

    client = cdsapi.Client()

    try:
        client.retrieve(
            "reanalysis-era5-land-timeseries",
            request,
            str(csv_path),
        )
    except Exception as first_error:
        # Compatibility fallback in case the CDS API expects `format` instead of `data_format`.
        request_legacy = dict(request)
        request_legacy.pop("data_format", None)
        request_legacy["format"] = "csv"

        print("[ERA5][WARN] Time-series data_format request failed. Retrying with format=csv.")

        try:
            client.retrieve(
                "reanalysis-era5-land-timeseries",
                request_legacy,
                str(csv_path),
            )
        except Exception as second_error:
            raise RuntimeError(
                "[ERA5] ERA5-Land time-series download failed. "
                f"First error: {first_error}; second error: {second_error}"
            ) from second_error

    return csv_path


def _standardize_era5_timeseries_table(era_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Convert a raw ERA5-Land time-series table into Petrichor-ready columns.
    """
    site_tz = _parse_site_timezone(cfg)

    rename_map = {}
    for col in era_raw.columns:
        key = str(col).strip().lower()

        if key in ("time", "valid_time", "date", "datetime"):
            rename_map[col] = "time"
        elif key in ("2m_temperature", "t2m"):
            rename_map[col] = "t2m"
        elif key in ("2m_dewpoint_temperature", "d2m"):
            rename_map[col] = "d2m"
        elif key in ("surface_pressure", "sp"):
            rename_map[col] = "sp"

    era_raw = era_raw.rename(columns=rename_map)

    if "time" not in era_raw.columns:
        raise KeyError(
            "[ERA5] Could not find a time column in ERA5-Land time-series file. "
            f"Columns found: {list(era_raw.columns)}"
        )

    era = pd.DataFrame()
    era["ERA5_UTC"] = pd.to_datetime(era_raw["time"], errors="coerce", utc=True)
    era["TIMESTAMP_LOCAL"] = (
        era["ERA5_UTC"]
        .dt.tz_convert(site_tz)
        .dt.tz_localize(None)
        .dt.floor("h")
    )

    if "t2m" in era_raw.columns:
        era["TA_ERA5_C"] = pd.to_numeric(era_raw["t2m"], errors="coerce") - 273.15

    if "d2m" in era_raw.columns:
        era["TD_ERA5_C"] = pd.to_numeric(era_raw["d2m"], errors="coerce") - 273.15

    if "sp" in era_raw.columns:
        era["PA_ERA5_HPA"] = pd.to_numeric(era_raw["sp"], errors="coerce") / 100.0

    if {"TA_ERA5_C", "TD_ERA5_C"}.issubset(era.columns):
        es_t = _saturation_vapor_pressure_hpa(era["TA_ERA5_C"])
        es_td = _saturation_vapor_pressure_hpa(era["TD_ERA5_C"])

        era["RH_ERA5_PERCENT"] = (100.0 * es_td / es_t).clip(
            lower=0.0,
            upper=100.0,
        )

        era["AH_ERA5_GM3"] = _absolute_humidity(es_td, era["TA_ERA5_C"])

    era = era.dropna(subset=["TIMESTAMP_LOCAL"])
    era = era.drop_duplicates(subset=["TIMESTAMP_LOCAL"])
    era = era.sort_values("TIMESTAMP_LOCAL")

    return era


def _read_era5_timeseries_netcdf(path: Path, cfg: dict) -> pd.DataFrame:
    """
    Read one ERA5-Land time-series NetCDF file and convert it to Petrichor-ready columns.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "[ERA5] xarray is not installed. Install it with: pip install xarray netcdf4"
        ) from exc

    with xr.open_dataset(path) as ds:
        raw = ds.to_dataframe().reset_index()

    return _standardize_era5_timeseries_table(raw, cfg)


def _combine_era5_timeseries_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine ERA5-Land time-series frames from multiple parameter groups.
    """
    if not frames:
        return pd.DataFrame()

    out = frames[0].copy()

    for frame in frames[1:]:
        out = out.merge(
            frame,
            on=["ERA5_UTC", "TIMESTAMP_LOCAL"],
            how="outer",
            suffixes=("", "_DUP"),
        )

        dup_cols = [c for c in out.columns if c.endswith("_DUP")]
        for dup_col in dup_cols:
            base_col = dup_col.replace("_DUP", "")
            if base_col in out.columns:
                out[base_col] = out[base_col].combine_first(out[dup_col])
            else:
                out[base_col] = out[dup_col]

        out = out.drop(columns=dup_cols, errors="ignore")

    out = out.drop_duplicates(subset=["TIMESTAMP_LOCAL"])
    out = out.sort_values("TIMESTAMP_LOCAL")

    return out


def _read_era5_land_timeseries_csv(csv_path: Path, cfg: dict) -> pd.DataFrame:
    """
    Read ERA5-Land time-series output.

    The CDS time-series backend may return:
    - a plain CSV,
    - a ZIP containing multiple CSV files,
    - a NetCDF file,
    - a ZIP containing multiple NetCDF files.

    This reader handles all of them and returns one merged dataframe.
    """
    import zipfile
    import tempfile

    csv_path = Path(csv_path)

    frames = []

    # Case 1: CDS returned a ZIP archive, usually because variables belong
    # to more than one ERA5-Land parameter group.
    if zipfile.is_zipfile(csv_path):
        print(f"[ERA5] Reading ERA5-Land time-series ZIP archive: {csv_path.name}")

        with zipfile.ZipFile(csv_path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                for name in names:
                    lower = name.lower()

                    if lower.endswith(".csv"):
                        with zf.open(name) as f:
                            raw = pd.read_csv(f)

                        frames.append(_standardize_era5_timeseries_table(raw, cfg))

                    elif lower.endswith(".nc") or lower.endswith(".netcdf"):
                        extracted = tmpdir / Path(name).name
                        with zf.open(name) as src, open(extracted, "wb") as dst:
                            dst.write(src.read())

                        frames.append(_read_era5_timeseries_netcdf(extracted, cfg))

        return _combine_era5_timeseries_frames(frames)

    # Case 2: plain NetCDF file, even if the filename extension is misleading.
    first_bytes = csv_path.read_bytes()[:16]

    if first_bytes.startswith(b"CDF") or first_bytes.startswith(b"\x89HDF"):
        print(f"[ERA5] Reading ERA5-Land time-series NetCDF file: {csv_path.name}")
        return _read_era5_timeseries_netcdf(csv_path, cfg)

    # Case 3: plain CSV file.
    try:
        raw = pd.read_csv(csv_path)
        return _standardize_era5_timeseries_table(raw, cfg)

    except UnicodeDecodeError:
        # Last-resort text encoding fallback.
        raw = pd.read_csv(csv_path, encoding="latin1")
        return _standardize_era5_timeseries_table(raw, cfg)

def gapfill_meteorology_from_era5_land(
    df: pd.DataFrame,
    cfg: dict,
    site_out: Path | str,
    neutron_col: str | None = None,
) -> pd.DataFrame:
    """
    Fill missing pressure, temperature, and humidity data using ERA5-Land.

    Rules
    -----
    - Only runs when config['era5_land_gapfill'] is True or missing.
    - Only fills rows where neutron-count data exist.
    - Only fills missing local meteorological values; observed values are not overwritten.
    - ERA5-Land time is UTC and is converted to local site time before merging.
    - Units are converted to Petrichor conventions:
        surface_pressure: Pa -> hPa
        2m_temperature: K -> degC
        2m_dewpoint_temperature + 2m_temperature -> RH (%) and AH (g m^-3)
    """
    if not bool(_cfg_get(cfg, "era5_land_gapfill", True)):
        return df

    out = df.copy()

    time_col = TIME_COL
    pressure_col = _cfg_get(cfg, "pressure_col", "PA_1")
    temp_col = _cfg_get(cfg, "temp_col", "TA_1")
    rh_col = _cfg_get(cfg, "rh_col", "RH_1")
    ah_col = _cfg_get(cfg, "ah_col", "AH_1")

    if time_col not in out.columns:
        raise KeyError(f"[ERA5] Missing time column '{time_col}'.")

    if _cfg_get(cfg, "site_latitude", None) in (None, "", "null"):
        raise ValueError("[ERA5] site_latitude is required for ERA5-Land gap filling.")

    if _cfg_get(cfg, "site_longitude", None) in (None, "", "null"):
        raise ValueError("[ERA5] site_longitude is required for ERA5-Land gap filling.")

    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")

    if neutron_col is None or neutron_col not in out.columns:
        for cand in ("N_CPH", "N", "MOD"):
            if cand in out.columns:
                neutron_col = cand
                break

    if neutron_col is None or neutron_col not in out.columns:
        print("[ERA5][WARN] No neutron count column found. ERA5-Land gap filling skipped.")
        return out

    has_neutron = pd.to_numeric(out[neutron_col], errors="coerce").notna()

    for col in [pressure_col, temp_col, rh_col]:
        if col not in out.columns:
            out[col] = np.nan

    missing_pressure = out[pressure_col].isna()
    missing_temp = out[temp_col].isna()
    missing_rh = out[rh_col].isna()

    fill_needed = has_neutron & (missing_pressure | missing_temp | missing_rh)

    if not fill_needed.any():
        print("[ERA5] No meteorological gaps with valid neutron counts. ERA5-Land download skipped.")
        return out

    variables = []

    if (has_neutron & missing_pressure).any():
        variables.append("surface_pressure")

    if (has_neutron & (missing_temp | missing_rh)).any():
        variables.append("2m_temperature")

    if (has_neutron & missing_rh).any():
        variables.append("2m_dewpoint_temperature")

    variables = sorted(set(variables))

    site_id = str(_cfg_get(cfg, "site_id", _cfg_get(cfg, "field_id", "site")))
    cache_dir = Path(_cfg_get(cfg, "cache_dir", "cache")).expanduser().resolve()

    site_tz = _parse_site_timezone(cfg)

    local_times = pd.to_datetime(out.loc[fill_needed, time_col], errors="coerce").dropna()
    local_aware = local_times.dt.tz_localize(
        site_tz,
        ambiguous="NaT",
        nonexistent="shift_forward",
    ).dropna()

    utc_times = local_aware.dt.tz_convert(timezone.utc)

    if utc_times.empty:
        print("[ERA5][WARN] No valid gap timestamps after timezone conversion. ERA5-Land skipped.")
        return out

    era_frames = []

    era5_backend = str(_cfg_get(cfg, "era5_land_backend", "timeseries")).lower()

    if era5_backend == "timeseries":
        try:
            csv_path = _download_era5_land_timeseries(
                cfg=cfg,
                site_id=site_id,
                start_utc=utc_times.min(),
                end_utc=utc_times.max(),
                variables=variables,
                cache_dir=cache_dir,
            )

            era_frames.append(_read_era5_land_timeseries_csv(csv_path, cfg))

        except Exception as e:
            print(f"[ERA5][WARN] ERA5-Land time-series backend failed: {e}")

            if bool(_cfg_get(cfg, "era5_land_strict", False)):
                raise

            print("[ERA5][WARN] Falling back to monthly ERA5-Land NetCDF backend.")

            for month_start in _month_starts_between(utc_times.min(), utc_times.max()):
                try:
                    nc_path = _download_era5_land_month(
                        cfg=cfg,
                        site_id=site_id,
                        year=int(month_start.year),
                        month=int(month_start.month),
                        variables=variables,
                        cache_dir=cache_dir,
                    )

                    era_frames.append(_read_era5_land_point(nc_path, cfg))

                except Exception as month_error:
                    print(
                        "[ERA5][WARN] ERA5-Land monthly gap filling failed for "
                        f"{site_id} {int(month_start.year)}-{int(month_start.month):02d}: "
                        f"{month_error}"
                    )

                    if bool(_cfg_get(cfg, "era5_land_strict", False)):
                        raise

                    continue

    else:
        for month_start in _month_starts_between(utc_times.min(), utc_times.max()):
            try:
                nc_path = _download_era5_land_month(
                    cfg=cfg,
                    site_id=site_id,
                    year=int(month_start.year),
                    month=int(month_start.month),
                    variables=variables,
                    cache_dir=cache_dir,
                )

                era_frames.append(_read_era5_land_point(nc_path, cfg))

            except Exception as e:
                print(
                    "[ERA5][WARN] ERA5-Land monthly gap filling failed for "
                    f"{site_id} {int(month_start.year)}-{int(month_start.month):02d}: {e}"
                )

                if bool(_cfg_get(cfg, "era5_land_strict", False)):
                    raise

                continue

    if not era_frames:
        return out

    era = pd.concat(era_frames, ignore_index=True)
    era = era.drop_duplicates(subset=["TIMESTAMP_LOCAL"]).sort_values("TIMESTAMP_LOCAL")

    merged = out[[time_col]].merge(
        era,
        left_on=time_col,
        right_on="TIMESTAMP_LOCAL",
        how="left",
    )

    fill_counts = {}

    def _fill_one(target_col: str, source_col: str) -> None:
        if source_col not in merged.columns:
            fill_counts[target_col] = 0
            return

        source = pd.to_numeric(merged[source_col], errors="coerce")
        mask = has_neutron & out[target_col].isna() & source.notna()

        out.loc[mask, target_col] = source.loc[mask].to_numpy()

        flag_col = f"ERA5_FILLED_{target_col}"
        if flag_col not in out.columns:
            out[flag_col] = False

        out.loc[mask, flag_col] = True
        fill_counts[target_col] = int(mask.sum())

    _fill_one(pressure_col, "PA_ERA5_HPA")
    _fill_one(temp_col, "TA_ERA5_C")
    _fill_one(rh_col, "RH_ERA5_PERCENT")

    required_vars = list(_cfg_get(cfg, "required_vars", []))
    fill_ah = ERA5_FILL_ABSOLUTE_HUMIDITY or (ah_col in required_vars)

    if fill_ah and "AH_ERA5_GM3" in merged.columns:
        if ah_col not in out.columns:
            out[ah_col] = np.nan

        _fill_one(ah_col, "AH_ERA5_GM3")

    msg = ", ".join(f"{k}={v}" for k, v in fill_counts.items())
    print(f"[ERA5] Gap-filled meteorological data from ERA5-Land: {msg}")

    return out


def check_variables_in_data(df: pd.DataFrame, config: dict) -> None:
    """Validate required/additional columns.

    RH_1 and AH_1 are treated as humidity-interchangeable: if one of them is
    listed in required_vars but absent, and the other is present in the data,
    it satisfies the humidity requirement (Petrichor can compute humidity
    correction and N0 calibration from AH_1 alone).
    """
    required = list(_cfg_get(config, "required_vars", []))

    humidity_pair = {"RH_1", "AH_1"}
    humidity_required = [c for c in required if c in humidity_pair]
    humidity_present = [c for c in humidity_pair if c in df.columns]

    missing = []
    for c in required:
        if c in df.columns:
            continue
        # Allow AH_1 <-> RH_1 substitution for the humidity requirement.
        if c in humidity_pair and humidity_present:
            continue
        missing.append(c)

    if missing:
        print("[ERROR] Missing required columns:", ", ".join(missing))
        raise ValueError("Required variables missing.")

    if humidity_required and not humidity_present:
        print("[ERROR] Humidity input missing: neither RH_1 nor AH_1 found in data.")
        raise ValueError("Required variables missing.")

    if humidity_required and ("RH_1" in humidity_required) and ("RH_1" not in df.columns) and ("AH_1" in df.columns):
        print("[INFO] RH_1 not found; using AH_1 as humidity input.")

    addl = _cfg_get(config, "additional_vars", [])
    found_addl = [c for c in addl if c in df.columns]
    if found_addl:
        print("[INFO] Found additional vars:", ", ".join(found_addl))


# ==============================================================================
# NET NORMALIZATION
# ==============================================================================

def normalize_by_net(df: pd.DataFrame, cfg: dict,
                     n_col: str = "N", net_col: str = "NET",
                     out_col: str = "N_CPH") -> tuple[pd.DataFrame, str, dict]:
    """
    Convert counts to counts-per-hour using NET (live time).
    Adds audit flags and returns the chosen neutron column.
    """
    out = df.copy()
    net_min = int(_cfg_get(cfg, "net_min_seconds", 1800))

    if (n_col in out.columns) and (net_col in out.columns):
        NET = pd.to_numeric(out[net_col], errors="coerce")
        N   = pd.to_numeric(out[n_col], errors="coerce")
        valid = (NET >= net_min) & (NET > 0)
        scale = np.where(valid, 3600.0 / NET, np.nan)
        out[out_col] = N * scale
        out["NET_MIN_SECONDS"] = net_min
        out["NET_SCALED_FLAG"] = pd.Series(np.where(valid, 1, 0), index=out.index, dtype="Int64")
        return out, out_col, {"net_min_seconds": net_min, "net_scaling_applied": True}

    out["NET_MIN_SECONDS"] = net_min
    out["NET_SCALED_FLAG"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    return out, n_col, {"net_min_seconds": net_min, "net_scaling_applied": False}


# ==============================================================================
# NMDB & CUTOFF RIGIDITY
# ==============================================================================

def nmdb_get(startdate: str, enddate: str, station: str, default_dir: str) -> Dict[pd.Timestamp, float]:
    """Fetch hourly NMDB counts via NEST ASCII."""
    sy, sm, sd = str(startdate).split("-")
    ey, em, ed = str(enddate).split("-")
    url = (
        "http://www.nmdb.eu/nest/draw_graph.php?formchk=1&stations[]={station}"
        "&tabchoice=1h&dtype=corr_for_efficiency&tresolution=60&force=1&yunits=0"
        "&date_choice=bydate&start_day={sd}&start_month={sm}&start_year={sy}"
        "&start_hour=0&start_min=0&end_day={ed}&end_month={em}&end_year={ey}"
        "&end_hour=23&end_min=59&output=ascii"
    ).format(station=station, sd=sd, sm=sm, sy=sy, ed=ed, em=em, ey=ey)
    print(f"[NMDB] {url}")
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    pre = soup.find_all('pre')[0].text
    pre = pre[pre.find('start_date_time'):]
    pre = pre.replace("start_date_time   1HCOR_E", "")
    lines = pre.strip().split("\n")[1:]
    df = pd.DataFrame(lines)
    df = df[0].str.split(";", n=2, expand=True)
    df.columns = ['DT', 'N_COUNT']
    df['N_COUNT'] = pd.to_numeric(df['N_COUNT'], errors='coerce').replace(0, np.nan)

    out_dir = Path(default_dir) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nmdb_station_counts.txt").write_text(
        df.to_csv(sep=";", index=False, header=False)
    )
    dts = pd.to_datetime(df['DT'])
    return dict(zip(dts, df['N_COUNT']))


def _load_rc_table() -> np.ndarray:
    """Lazy-load the cutoff rigidity grid from src.rc_table.

    Raises
    ------
    RuntimeError
        If src.rc_table cannot be imported. We never silently fall back to a
        constant grid because that would hide a real configuration problem and
        produce a wrong Rc for every site.
    """
    try:
        from .rc_table import cutoff_rigidity as rc_table
    except Exception as exc:
        raise RuntimeError(
            "[Rc] Could not import src.rc_table.cutoff_rigidity. "
            "Petrichor refuses to use a constant fallback grid because that "
            "silently corrupts the intensity correction. "
            "Please make sure src/rc_table.py is present and importable."
        ) from exc
    return np.asarray(rc_table)


def rc_retrieval(latitude: float, longitude: float) -> float:
    """Interpolate cutoff rigidity from the global rc_table grid."""
    print(f"[Rc] Computing cutoff rigidity at lat={latitude}, lon={longitude}")
    xq = float(longitude)
    yq = float(latitude)
    if xq < 0:
        xq = -xq + 180.0
    Z = _load_rc_table()
    x = np.linspace(0, 360, Z.shape[1])
    y = np.linspace(90, -90, Z.shape[0])
    X, Y = np.meshgrid(x, y)
    zq = griddata(np.c_[X.ravel(), Y.ravel()], Z.ravel(), (xq, yq))
    zq = float(np.round(zq, 2))
    print(f"[Rc] Result: {zq} GV (±0.3 GV)")
    return zq


# ==============================================================================
# CORRECTIONS (each block groups its own helpers)
# ==============================================================================

# --- PRESSURE CORRECTION ---------------------------------------------------

def betacoeff(latitude: float, elevation: float, Rc: float) -> Tuple[str, float]:
    """Compute barometric coefficient B and reference pressure x0 (hPa)."""
    rho_rck = 2670.0
    x0 = (101325 * (1 - 2.25577e-5 * elevation) ** 5.25588) / 100.0  # hPa
    z = -0.00000448211 * x0**3 + 0.0160234 * x0**2 - 27.0977 * x0 + 15666.1
    g_lat = 978032.7 * (1 + 0.0053024 * (np.sin(np.radians(latitude)) ** 2)
                        - 0.0000058 * (np.sin(np.radians(2 * latitude))) ** 2)
    del_free_air = -0.3087691 * z
    del_boug = rho_rck * z * 0.00004193
    g_corr = (g_lat + del_free_air + del_boug) / 100000.0
    g = g_corr / 10.0
    x = x0 / g

    n_1 = 0.01231386
    alpha_1 = 0.0554611
    k_1 = 0.6012159
    b0 = 4.74235E-06; b1 = -9.66624E-07; b2 = 1.42783E-09
    b3 = -3.70478E-09; b4 = 1.27739E-09; b5 = 3.58814E-11
    b6 = -3.146E-15;   b7 = -3.5528E-13; b8 = -4.29191E-14

    term1 = n_1 * (1 + np.exp(-alpha_1 * Rc ** k_1)) ** -1 * (x - x0)
    term2 = 0.5 * (b0 + b1 * Rc + b2 * Rc ** 2) * (x ** 2 - x0 ** 2)
    term3 = 0.3333 * (b3 + b4 * Rc + b5 * Rc ** 2) * (x ** 3 - x0 ** 3)
    term4 = 0.25   * (b6 + b7 * Rc + b8 * Rc ** 2) * (x ** 4 - x0 ** 4)

    beta_coeff = round(float(abs((term1 + term2 + term3 + term4) / (x0 - x))), 5)
    print(f"B = {beta_coeff}, x0 = {x0:.2f} hPa")
    return beta_coeff, float(x0)

def _pressfact_B(press_hpa: pd.Series, B: float, p0_hpa: float) -> pd.Series:
    """Np = N * exp(B * (P - p0))"""

    """
    Pressure correction factor f_p for neutron counts.

    Purpose
    -------
    Adjusts observed neutron counts from ambient pressure P to a reference pressure p0.

    Parameters
    ----------
    press : array-like or pandas.Series
        Ambient air pressure P (hPa, a.k.a. mb).
    B : float
        Barometric (pressure) coefficient, typically in hPa^-1.
        Can be obtained via `betacoeff()`.
    p0 : float
        Reference pressure (hPa) to which counts are normalized.

    Returns
    -------
    f_p : numpy.ndarray or pandas.Series
        Pressure correction factor. Apply as: N_corr = N * f_p

    Notes
    -----
    Formula:
        f_p = exp(B * (P - p0))
    
    """
    return np.exp(B * (press_hpa - p0_hpa))

def apply_pressure_correction(df: pd.DataFrame, cfg: dict,
                              B: float, p0: float,
                              in_col: Optional[str] = None,
                              out_col: str = "N_corr_press") -> pd.DataFrame:
    """Apply barometric correction using local block helper."""

    """
    Apply pressure correction to a neutron-count column.

    Purpose
    -------
    Produces a pressure-corrected count column using:
        N_corr_press = N_in * exp(B * (P - p0))

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing at least the neutron and pressure columns.
    cfg : dict
        Config with column names; uses cfg["pressure_col"] (default "PA_1").
    B : float
        Barometric coefficient (hPa^-1).
    p0 : float
        Reference pressure (hPa).
    in_col : str, optional
        Name of the input neutron column. Direct calls default to Petrichor's
        fixed raw-count column ``N``.
    out_col : str, default "N_PRESS_ONLY"
        Name of the output corrected column.

    Returns
    -------
    pandas.DataFrame
        Copy of df with the new pressure-corrected column added.
    """
    neutron_col  = in_col or NEUTRON_COL
    pressure_col = _cfg_get(cfg, "pressure_col", "PA_1")
    out = df.copy()
    for c in [neutron_col, pressure_col]:
        if c not in out.columns:
            raise KeyError(f"[PRESS] Missing column '{c}'.")
    N  = pd.to_numeric(out[neutron_col], errors="coerce")
    P  = pd.to_numeric(out[pressure_col], errors="coerce")
    out[out_col] = N * _pressfact_B(P, float(B), float(p0))
    return out

# --- HUMIDITY CORRECTION ---------------------------------------------------

def _saturation_vapor_pressure_hpa(Tc: pd.Series) -> pd.Series:
    """Return saturation vapor pressure in hPa (Tetens formula)."""
    return 6.112 * np.exp((17.67 * Tc) / (243.5 + Tc))

def _actual_vapor_pressure_hpa_from_T_RH(es_hpa: pd.Series, RH_percent: pd.Series) -> pd.Series:
    """Return actual vapor pressure in hPa from es (hPa) and RH (%)."""
    return es_hpa * (RH_percent / 100.0)

def _absolute_humidity(ea_hpa: pd.Series, T_c: pd.Series) -> pd.Series:
    """ρv [g m^-3] from ea [Pa] and T [°C]."""

    """
    Compute absolute humidity ρ_v (g m^-3).

    Purpose
    -------
    Converts temperature and RH (or dew point) to absolute humidity for humidity correction.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain T and RH, or dew-point column if configured.
    cfg : dict
        Column mapping: "temp_col" (°C), "rh_col" (%), optional "dewpoint_col" (°C).

    Returns
    -------
    pandas.Series
        Absolute humidity ρ_v in g m^-3 per row.
    """
    ea_pa = ea_hpa * 100.0
    return 1000.0 * ea_pa / (461.5 * (T_c + 273.15))

def _humidity_factor(rhov: pd.Series, rhov0: float) -> pd.Series:
    """Nh = Np * (1 + 0.0054*(ρv - ρv0))."""
    return (1.0 + 0.0054 * (rhov - rhov0))


def apply_humidity_correction(df: pd.DataFrame, cfg: dict,
                              rhov0: float,
                              in_col: Optional[str] = None,
                              out_col: str = "N_corr_hum") -> pd.DataFrame:
    """Apply humidity correction using AH, dew point, or T/RH based on config."""
    out = df.copy()
    Np_col = in_col or "N_corr_press"

    if Np_col not in out.columns:
        raise KeyError(f"[HUM] Missing '{Np_col}'.")

    dew_col = _cfg_get(cfg, "dewpoint_col", None)
    T_col = _cfg_get(cfg, "temp_col", "TA_1")
    RH_col = _cfg_get(cfg, "rh_col", "RH_1")

    # Humidity source priority: AH_1 (if present) > dew point > T/RH.
    # Using AH_1 whenever available lets the correction run even if RH_1 is
    # missing from the input CSV.
    if "AH_1" in out.columns:
        rhov = pd.to_numeric(out["AH_1"], errors="coerce")

    elif dew_col and (dew_col in out.columns):
        Td = pd.to_numeric(out[dew_col], errors="coerce")
        es_hpa = _saturation_vapor_pressure_hpa(Td)
        ea_hpa = es_hpa
        rhov = _absolute_humidity(ea_hpa, Td)

    else:
        for c in [T_col, RH_col]:
            if c not in out.columns:
                raise KeyError(f"[HUM] Missing column '{c}'.")

        T = pd.to_numeric(out[T_col], errors="coerce")
        RH = pd.to_numeric(out[RH_col], errors="coerce")
        es_hpa = _saturation_vapor_pressure_hpa(T)
        ea_hpa = _actual_vapor_pressure_hpa_from_T_RH(es_hpa, RH)
        rhov = _absolute_humidity(ea_hpa, T)

    Fh = _humidity_factor(rhov, float(rhov0))
    out[out_col] = pd.to_numeric(out[Np_col], errors="coerce") * Fh
    return out


# --- INTENSITY CORRECTION --------------------------------------------------

''' 
Group 2 methods: Hawdon (2014), McJannet & Desilets (2023)

Suggesstion: use the McJannet2023 method
'''


def _RcCorr_gamma(Rc_site: float, Rc_ref: float) -> float:
    """Hawdon et al. (2014) geomagnetic factor gamma."""

    """
    Amplitude scaling factor for geomagnetic differences (Hawdon et al., 2014).

    Purpose
    -------
    Adjusts the intensity correction to account for different cutoff rigidity between
    the CRNS site (Rc) and the reference neutron monitor (Rc_ref).

    Parameters
    ----------
    Rc : float
        Site cutoff rigidity (GV).
    Rc_ref : float
        Reference station cutoff rigidity (GV).

    Returns
    -------
    gamma : float
        Amplitude scaling factor: gamma = -0.075 * (Rc - Rc_ref) + 1
    """
    return -0.075 * (Rc_site - Rc_ref) + 1.0

def _intensity_factor_hawdon2014(station_ref: float,
                                 station_count: pd.Series,
                                 Rc_site: float, Rc_ref: float) -> pd.Series:
    """Hawdon 2014: f_i = 1 / (((ratio - 1) * gamma) + 1)."""

    """
    Incoming neutron intensity correction (Hawdon et al., 2014).

    Purpose
    -------
    Produces the intensity correction factor f_i using the ratio of
    reference monitor counts and a geomagnetic amplitude factor.

    Parameters
    ----------
    station_ref : float or array-like
        Reference monitor count (e.g., baseline value).
    station_count : float or array-like
        Monitor count at the target time.
    gamma : float
        Amplitude scaling factor from Rc difference (see RcCorr).

    Returns
    -------
    f_i : numpy.ndarray or pandas.Series
        Intensity correction factor.
        Apply as: N_corr_int = N_in / f_i

    Notes
    -----
    Using the McJannet & Desilets (2023) notation:
        r = station_count / station_ref
        f_i = 1 / ( (r - 1) * gamma + 1 )
    
    References
    ----------
    Hawdon et al. (2014) :  https://doi.org/10.1002/2013WR015138    
    """
    ratio = station_count / station_ref
    gamma = _RcCorr_gamma(Rc_site, Rc_ref)
    return 1.0 / (((ratio - 1.0) * gamma) + 1.0)

def _atmospheric_depth_gcm2(elevation_m: float, latitude_deg: float) -> float:
    """Atmospheric depth (g/cm^2) for McJannet & Desilets (2023)."""

    """
    Estimate atmospheric depth (g cm^-2) at a given location.

    Purpose
    -------
    Required intermediate for McJannet & Desilets (2023) location factor τ.

    Parameters
    ----------
    elevation : float
        Site elevation above sea level (m).
    latitude : float
        Geographic latitude (degrees, -90..90).

    Returns
    -------
    X : float
        Atmospheric depth in g cm^-2.
    """
    rho_rock = 2670.0
    p_sea = 1013.25
    M = 0.0289644
    R = 8.31432
    T0 = 288.15
    lapse = -0.0065
    lat = float(latitude_deg); z = float(elevation_m)
    g0 = 9.780327 * (1 + 0.0053024 * np.sin(np.radians(lat))**2 - 0.0000058 * np.sin(2*np.radians(lat))**2)
    free_air = -3.086e-6 * z
    bouguer  = 4.193e-10 * rho_rock * z
    g = g0 + free_air + bouguer
    p_ref = p_sea * (1 + lapse / T0 * z) ** (-(g * M) / (R * lapse))
    return float((10.0 * p_ref) / g)

def _location_factor_tau(site_depth: float, Rc_site: float,
                         ref_depth: float, Rc_ref: float) -> tuple[float, float]:
    
    """McJannet & Desilets (2023) tau and K (K not used in f_i but returned for completeness)."""

    """
    Location factor τ and scaling K (McJannet & Desilets, 2023).

    Purpose
    -------
    Computes τ for the site and K for reconciling the chosen reference monitor
    to the CLMX baseline used in the paper.

    Parameters
    ----------
    site_atm_depth : float
        Atmospheric depth at the CRNS site (g cm^-2).
    Rc : float
        Cutoff rigidity at the CRNS site (GV).
    ref_atm_depth : float
        Atmospheric depth at the reference monitor (g cm^-2).
    Rc_ref : float
        Cutoff rigidity at the reference monitor (GV).

    Returns
    -------
    tau : float
        Location factor τ for the site.
    K : float
        Scaling factor between the chosen monitor and CLMX reference.

    References
    ----------
    Atmosphere, U. S. (1976). US standard atmosphere. National Oceanic and Atmospheric Administration.

    McJannet, D. L., & Desilets, D. (2023). Incoming Neutron Flux Corrections for Cosmic‐Ray Soil and Snow Sensors Using the 
        Global Neutron Monitor Network. Water Resources Research, 59(4), e2022WR033889.
    """
    c0, c1, c2, c3, c4, c5 = -0.0009, 1.7699, 0.0064, 1.8855, 0.000013, -1.2237
    eps = 1.0
    tau_new = eps * (c0 * ref_depth + c1) * (1 - np.exp(-(c2 * ref_depth + c3) * (Rc_ref ** (c4 * ref_depth + c5))))
    tau = eps * (c0 * site_depth + c1) * (1 - np.exp(-(c2 * site_depth + c3) * (Rc_site ** (c4 * site_depth + c5))))
    K = 1.0 / tau_new
    return float(tau), float(K)

def _intensity_factor_mcjannet2023(station_ref: float,
                                   station_count: pd.Series,
                                   tau: float) -> pd.Series:
    """f_i = 1 / (tau * ratio + 1 - tau)."""

    """
    Incoming neutron intensity correction (McJannet & Desilets, 2023).

    Purpose
    -------
    Produces intensity correction factor f_i using the location factor τ.

    Parameters
    ----------
    station_ref : float or array-like
        Reference monitor count (e.g., baseline).
    station_count : float or array-like
        Monitor count at the target time.
    tau : float
        Location factor for the CRNS site.

    Returns
    -------
    f_i : numpy.ndarray or pandas.Series
        Intensity correction factor.
        Apply as: N_corr_int = N_in / f_i

    Notes
    -----
    Let r = station_count / station_ref, then:
        f_i = 1 / ( τ * r + 1 - τ )
    """
    ratio = station_count / station_ref
    return 1.0 / (tau * ratio + 1.0 - tau)

def apply_intensity_correction(df: pd.DataFrame, cfg: dict,
                               nmdb_df: pd.DataFrame,
                               jung_ref: float,
                               in_col: str = "N_corr_hum",
                               out_col: str = "N_corr_int") -> pd.DataFrame:
    """Choose an intensity method and correct N."""

    """
    Apply incoming-intensity correction to a neutron-count column.

    Purpose
    -------
    Aligns hourly NMDB monitor data to site timestamps and converts an input count
    column to an intensity-corrected column. Internally uses a chosen method
    (Hawdon2014 or McJannet2023) depending on your implementation.

    Parameters
    ----------
    df : pandas.DataFrame
        Site data with a time column and an input neutron column (e.g., "N_corr_hum").
    cfg : dict
        Site configuration for correction parameters such as latitude,
        elevation, and cutoff rigidity. The timestamp column is fixed as
        ``TIMESTAMP`` and is not read from configuration.
    nmdb_df : pandas.DataFrame
        Reference monitor table with columns ["DT", "N_COUNT"] (datetime, counts).
    jung_ref : float
        Reference baseline count at the monitor (e.g., a median or a fixed reference date).
    in_col : str, optional
        Input column to be corrected (default "N_corr_hum").
    out_col : str, default "N_INT_ONLY"
        Output column name.

    Returns
    -------
    pandas.DataFrame
        Copy of df with the intensity-corrected column added.

    Notes
    -----
    Typical workflow:
      1) Resample/merge-asof NMDB to hourly and align by time.
      2) Compute r = N_COUNT / jung_ref.
      3) Choose correction (Hawdon or McJannet) to build f_i.
      4) N_corr_int = N_in / f_i
    """
    out = df.copy()
    tcol = TIME_COL
    if in_col not in out.columns:
        raise KeyError(f"[INT] Missing '{in_col}'.")
    if nmdb_df is None:
        out[out_col] = out[in_col]
        return out

    nmdb = nmdb_df.copy()
    if 'DT' not in nmdb.columns:
        raise KeyError("[INT] NMDB must have 'DT'.")
    if 'N_COUNT' not in nmdb.columns:
        if nmdb.shape[1] >= 2:
            nmdb.columns = ['DT', 'N_COUNT'][:nmdb.shape[1]]
        else:
            raise KeyError("[INT] NMDB must have 'N_COUNT'.")

    # Make both merge keys explicit datetime64[ns] before merge_asof
    out[tcol] = pd.to_datetime(out[tcol], errors="coerce").astype("datetime64[ns]")
    out = out.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)

    nmdb["DT"] = pd.to_datetime(nmdb["DT"], errors="coerce").astype("datetime64[ns]")
    nmdb = nmdb.dropna(subset=["DT"]).sort_values("DT")
    nmdb = (
        nmdb.set_index("DT")
            .resample("h")
            .mean()
            .interpolate()
            .reset_index()
    )

    nmdb["DT"] = pd.to_datetime(nmdb["DT"], errors="coerce").astype("datetime64[ns]")
    nmdb_merge = nmdb.rename(columns={"DT": tcol}).copy()
    nmdb_merge[tcol] = pd.to_datetime(nmdb_merge[tcol], errors="coerce").astype("datetime64[ns]")

    merged = pd.merge_asof(
        out[[tcol, in_col]].sort_values(tcol),
        nmdb_merge.sort_values(tcol),
        on=tcol,
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )

    N_hum  = pd.to_numeric(merged[in_col], errors="coerce")
    counts = pd.to_numeric(merged['N_COUNT'], errors="coerce")
    ref    = float(jung_ref)
    method = str(_cfg_get(cfg, "intensity_method", "hawdon2014")).lower()

    if method == "hawdon2014":
        Rc_site = _cfg_get(cfg, 'rc', None)
        if Rc_site in (None, "", "null"):
            raise KeyError("[INT] Hawdon2014 requires 'rc'.")
        ref_meta = _resolve_nmdb_station_metadata(cfg)
        f_i = _intensity_factor_hawdon2014(
            ref, counts,
            float(Rc_site),
            float(ref_meta["rc"])
        )

    elif method == "mcjannet2023":
        Rc_site = _cfg_get(cfg, 'rc', None)
        if Rc_site in (None, "", "null"):
            raise KeyError("[INT] McJannet2023 requires 'rc'.")

        site_lat  = _cfg_get(cfg, "site_latitude", None)
        site_elev = _cfg_get(cfg, "site_elevation", None)

        if (site_lat in (None, "", "null")) or (site_elev in (None, "", "null")):
            raise KeyError("[INT] McJannet2023 requires site_latitude and site_elevation.")

        site_depth = _atmospheric_depth_gcm2(float(site_elev), float(site_lat))
        ref_meta = _resolve_nmdb_station_metadata(cfg)
        ref_lat = ref_meta["latitude"]
        ref_elev = ref_meta["elevation"]
        ref_depth = _atmospheric_depth_gcm2(float(ref_elev), float(ref_lat))
        Rc_ref = float(ref_meta["rc"])
        tau, _K  = _location_factor_tau(site_depth, float(Rc_site), ref_depth, Rc_ref)
        f_i = _intensity_factor_mcjannet2023(ref, counts, tau)

    else:
        raise ValueError(f"[INT] Unknown intensity_method '{method}'.")

    out[out_col] = N_hum * f_i
    out["F_INTENSITY"] = pd.Series(f_i).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return out


# --- AGB (VEGETATION) CORRECTION ------------------------------------------
AGB_MIN_DENOMINATOR = 0.05


def _agb_factor(agb: pd.Series) -> pd.Series:
    """Return the AGB factor for AGB expressed exclusively in kg/m²."""

    """
    Above-ground biomass correction factor f_v.

    Purpose
    -------
    Accounts for neutron moderation by above-ground biomass (vegetation).
    The factor typically reduces/increases counts depending on biomass load.

    Parameters
    ----------
    agb : pandas.Series
        Above-ground biomass in kg m^-2.

    Returns
    -------
    f_v : float or numpy.ndarray
        Vegetation correction factor.

    Notes
    -----
    Example linearized form used in some implementations:
        f_v = 1 / (1 - c * AGB)
    with c ≈ 0.009 (model-dependent).
    """
    agb = pd.to_numeric(agb, errors="coerce")
    negative = agb.notna() & (agb < 0.0)
    if negative.any():
        raise ValueError(
            f"[AGB] AGB_KGM2 must be non-negative; found {int(negative.sum())} "
            "negative value(s)."
        )

    denom = 1.0 - 0.009 * agb
    unsafe = denom.notna() & (denom <= AGB_MIN_DENOMINATOR)
    if unsafe.any():
        raise ValueError(
            f"[AGB] {int(unsafe.sum())} AGB value(s) make the correction "
            f"denominator <= {AGB_MIN_DENOMINATOR}. Maximum AGB_KGM2 is "
            f"{agb.max():.6g} kg/m²; correction aborted."
        )
    return 1.0 / denom

def resolve_daily_agb(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Map a sparse AGB time series onto every hourly row using step interpolation.

    AGB is supplied as an independent series of (DATE, AGB_KGM2) measurement points
    (see main.py, which loads "<input_stem>_AGB.csv" into the transient
    config["config"]["_agb_daily_lookup"]).  The series is NOT linearly
    interpolated: each hourly row takes the AGB value of the most recent
    measurement on or before its calendar date, and that value is held
    constant until the next measurement appears (last-observation-carried-
    forward). Rows before the first measurement remain unknown (NaN).

    Workflow:
    1. Build a date→AGB lookup (priority: pre-extracted lookup from the AGB
       file → config.agb_file → inline DATE_AGB/AGB columns).
    2. Extract each row's calendar date from TIMESTAMP / DATETIME.
    3. Forward-fill (step) the measured AGB onto every hourly row.
    4. Use AGB_KGM2 directly; no alternative AGB unit is accepted.

    If no AGB source is available the DataFrame is returned unchanged.
    """
    out = df.copy()

    time_col = TIME_COL
    ts = pd.to_datetime(
        out["DATETIME"] if "DATETIME" in out.columns else out[time_col],
        errors="coerce",
    )
    row_dates = ts.dt.date

    lookup = None  # pd.Series indexed by datetime.date, values = AGB (kg/m²)

    # ------------------------------------------------------------------
    # Priority 1: pre-extracted lookup built in main.py from the dedicated
    # external AGB file ("<input_stem>_AGB.csv").  This is the standard
    # workflow now that AGB lives in its own file.
    # ------------------------------------------------------------------
    pre_extracted = _cfg_get(cfg, "_agb_daily_lookup", None)
    if pre_extracted and isinstance(pre_extracted, dict):
        lookup = pd.Series({
            pd.Timestamp(k).date(): float(v)
            for k, v in pre_extracted.items()
        })
        print(f"[AGB] Using AGB lookup from file: {len(lookup)} measured points.")

    # ------------------------------------------------------------------
    # Priority 2: external AGB CSV file specified in config (agb_file).
    # Expected columns: DATE and AGB_KGM2 (kg/m²).
    # ------------------------------------------------------------------
    if lookup is None:
        agb_file = _cfg_get(cfg, "agb_file", None)
        if agb_file:
            agb_path = Path(agb_file)
            if not agb_path.is_absolute():
                cfg_path = _cfg_get(cfg, "_config_path", None)
                if cfg_path:
                    agb_path = Path(cfg_path).parent.parent / agb_file
                else:
                    agb_path = Path(agb_file)
            if agb_path.exists():
                agb_df = pd.read_csv(agb_path)
                date_col_ext = next(
                    (c for c in agb_df.columns if c.upper() in ("DATE", "YYYY-MM-DD")), None
                )
                val_col_ext = next(
                    (c for c in agb_df.columns if c.upper() == "AGB_KGM2"), None
                )
                if date_col_ext and val_col_ext:
                    # AGB file dates are written as DD/MM/YYYY (or ISO
                    # YYYY-MM-DD, unaffected by dayfirst), so use dayfirst=True.
                    ext_dates = pd.to_datetime(
                        agb_df[date_col_ext], errors="coerce", dayfirst=True
                    ).dt.date
                    ext_vals = pd.to_numeric(agb_df[val_col_ext], errors="coerce")
                    lookup = (
                        pd.DataFrame({"date": ext_dates, "AGB": ext_vals})
                        .dropna(subset=["date", "AGB"])
                        .groupby("date")["AGB"]
                        .max()
                    )
                    print(f"[AGB] External file: {agb_path.name} → "
                          f"{len(lookup)} dates loaded.")
                else:
                    raise ValueError(
                        f"[AGB] agb_file '{agb_path}' must contain a date column "
                        "and AGB_KGM2 in kg/m²."
                    )
            else:
                print(f"[AGB][WARN] agb_file '{agb_path}' not found. "
                      f"Falling back to inline AGB column.")

    # ------------------------------------------------------------------
    # Priority 3: inline AGB_KGM2 values, retained only for direct callers.
    # ------------------------------------------------------------------
    if lookup is None:
        if "AGB_KGM2" not in out.columns:
            return out  # no AGB data → skip
        agb_vals = pd.to_numeric(out["AGB_KGM2"], errors="coerce")
        _agb_date_col = (
            "DATE_AGB" if "DATE_AGB" in out.columns
            else "DATE"  if "DATE"     in out.columns
            else None
        )
        if _agb_date_col is not None:
            agb_dates = pd.to_datetime(
                out[_agb_date_col], errors="coerce", dayfirst=True
            ).dt.date
            lookup = (
                pd.DataFrame({"date": agb_dates, "AGB": agb_vals})
                .dropna(subset=["date", "AGB"])
                .groupby("date")["AGB"]
                .max()
            )
        else:
            lookup = (
                pd.DataFrame({"date": row_dates, "AGB": agb_vals})
                .dropna(subset=["AGB"])
                .groupby("date")["AGB"]
                .max()
            )

    if lookup is None or len(lookup) == 0:
        print("[AGB][WARN] No AGB measurements available; skipping AGB correction.")
        return out

    # ------------------------------------------------------------------
    # Step interpolation (last observation carried forward).
    # For each hourly row, take the AGB of the most recent measurement whose
    # date is on or before the row date.  The value is held constant until a
    # newer measurement appears. Rows before the first measurement remain NaN.
    # ------------------------------------------------------------------
    meas = (
        pd.DataFrame({
            "date": pd.to_datetime(list(lookup.index)),
            "AGB": lookup.values,
        })
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    row_df = pd.DataFrame({
        "_orig": np.arange(len(out)),
        "date": pd.to_datetime(pd.Series(row_dates), errors="coerce"),
    })
    valid = row_df.dropna(subset=["date"]).sort_values("date")

    merged = pd.merge_asof(valid, meas, on="date", direction="backward")

    mapped = pd.Series(np.nan, index=np.arange(len(out)), dtype="float64")
    mapped.loc[merged["_orig"].to_numpy()] = merged["AGB"].to_numpy()

    n_matched = int(mapped.notna().sum())
    print(f"[AGB] AGB step-mapped to {n_matched}/{len(mapped)} hourly rows "
          f"(held constant between measurements). Max raw AGB = {mapped.max():.4f}")

    out["AGB_KGM2"] = mapped.values
    out["AGB"] = mapped.values
    return out


def apply_agb_correction(df: pd.DataFrame, cfg: dict,
                         in_col: str = "N_corr_int",
                         out_col: str = "N_corr_agb") -> pd.DataFrame:
    """Apply vegetation correction if AGB column exists; else pass-through."""

    """
    Apply above-ground biomass (AGB) correction to a neutron-count column.

    Purpose
    -------
    Produces N_corr_agb by multiplying the input column with a biomass-dependent
    factor f_v = f(AGB). If AGB column is missing, passes input through.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table; should include the AGB column if available.
    cfg : dict
        Config with column mapping. Uses cfg["agb_col"] if set.
    in_col : str, optional
        Input column to correct (default "N_corr_int").
    out_col : str, default "N_AGB_ONLY"
        Name of output corrected column.

    Returns
    -------
    pandas.DataFrame
        Copy of df with the vegetation-corrected column added.

    Notes
    -----
    If no AGB column is configured/found, the function copies the input column
    to `out_col` (no-op).
    """
    out = df.copy()
    if in_col not in out.columns:
        raise KeyError(f"[AGB] Missing '{in_col}'.")
    input_counts = pd.to_numeric(out[in_col], errors="coerce")
    agb_col = _cfg_get(cfg, "agb_col", None)
    if agb_col and (agb_col in out.columns):
        agb = pd.to_numeric(out[agb_col], errors="coerce")
        correction_applied = agb.notna()
        # Unknown AGB means that vegetation correction is unavailable, not that
        # the already pressure/humidity/intensity-corrected neutron count is
        # unknown. Preserve the count and use the neutral factor for those rows.
        factor = _agb_factor(agb).where(correction_applied, 1.0)
        out[out_col] = input_counts * factor
        out["AGB_CORRECTION_APPLIED"] = correction_applied
    else:
        out[out_col] = input_counts
        out["AGB_CORRECTION_APPLIED"] = False
    return out


# --- ONE-STOP CORRECTION PIPELINE -----------------------------------------
def apply_all_corrections_combined(
    df: pd.DataFrame,
    cfg: dict,
    nmdb_df: Optional[pd.DataFrame],
    jung_ref: Optional[float],
    beta_B: Optional[float],
    p0_ref: Optional[float],
    rhov0_ref: Optional[float],
    neutron_col: str = NEUTRON_COL,
    out_col_int: str = "N_corr_int",
    out_col_agb: str = "N_corr"
) -> pd.DataFrame:
    """
    Apply pressure, humidity, intensity, and AGB corrections in sequence
    using the standalone correction functions as the single source of truth.

    This wrapper only orchestrates the correction chain and organizes
    diagnostic outputs for plotting and debugging.
    """
    tcol = TIME_COL
    ncol = neutron_col

    out = df.copy()

    # Safer timestamp parsing and sorting
    out[tcol] = pd.to_datetime(out[tcol], errors="coerce")
    out = out.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)

    if ncol not in out.columns:
        raise KeyError(f"[CORR] Missing neutron column '{ncol}'.")

    N_raw = pd.to_numeric(out[ncol], errors="coerce")

    # ------------------------------------------------------------------
    # 1) Pressure correction
    # ------------------------------------------------------------------
    if (beta_B in (None, "", "null")) or (p0_ref in (None, "", "null")):
        out["N_corr_press"] = N_raw
    else:
        out = apply_pressure_correction(
            df=out,
            cfg=cfg,
            B=float(beta_B),
            p0=float(p0_ref),
            in_col=ncol,
            out_col="N_corr_press"
        )

    # ------------------------------------------------------------------
    # 2) Humidity correction
    # ------------------------------------------------------------------
    if rhov0_ref in (None, "", "null"):
        out["N_corr_hum"] = pd.to_numeric(out["N_corr_press"], errors="coerce")
    else:
        out = apply_humidity_correction(
            df=out,
            cfg=cfg,
            rhov0=float(rhov0_ref),
            in_col="N_corr_press",
            out_col="N_corr_hum"
        )

        # Optional diagnostic: store absolute humidity if needed for plotting.
        # Same source priority as apply_humidity_correction: AH_1 > dew > T/RH.
        dew_col = _cfg_get(cfg, "dewpoint_col", None)
        T_col = _cfg_get(cfg, "temp_col", "TA_1")
        RH_col = _cfg_get(cfg, "rh_col", "RH_1")

        try:
            if "AH_1" in out.columns:
                out["pv"] = pd.to_numeric(out["AH_1"], errors="coerce")
            elif dew_col and (dew_col in out.columns):
                Td = pd.to_numeric(out[dew_col], errors="coerce")
                es_hpa = _saturation_vapor_pressure_hpa(Td)
                ea_hpa = es_hpa
                out["pv"] = _absolute_humidity(ea_hpa, Td)
            else:
                T = pd.to_numeric(out[T_col], errors="coerce")
                RH = pd.to_numeric(out[RH_col], errors="coerce")
                es_hpa = _saturation_vapor_pressure_hpa(T)
                ea_hpa = _actual_vapor_pressure_hpa_from_T_RH(es_hpa, RH)
                out["pv"] = _absolute_humidity(ea_hpa, T)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 3) Intensity correction
    # ------------------------------------------------------------------
    if (nmdb_df is None) or (jung_ref in (None, "", "null")):
        out[out_col_int] = pd.to_numeric(out["N_corr_hum"], errors="coerce")
        out["F_INTENSITY"] = 1.0
    else:
        out = apply_intensity_correction(
            df=out,
            cfg=cfg,
            nmdb_df=nmdb_df,
            jung_ref=float(jung_ref),
            in_col="N_corr_hum",
            out_col=out_col_int
        )

    # ------------------------------------------------------------------
    # 4) AGB correction
    # Expand daily AGB values (DATE_AGB / AGB columns) to every hourly
    # row before applying the correction factor.
    # ------------------------------------------------------------------
    out = resolve_daily_agb(out, cfg)
    # Ensure agb_col always points to the resolved "AGB" column when
    # DATE_AGB data is present; fall back to whatever is in cfg otherwise.
    if "AGB" in out.columns:
        cfg = dict(cfg)          # shallow copy – do not mutate caller's cfg
        cfg["agb_col"] = "AGB"
    out = apply_agb_correction(
        df=out,
        cfg=cfg,
        in_col=out_col_int,
        out_col=out_col_agb
    )

    # ------------------------------------------------------------------
    # 5) Derive per-factor diagnostics from stepwise outputs
    # ------------------------------------------------------------------
    N_press = pd.to_numeric(out["N_corr_press"], errors="coerce")
    N_hum   = pd.to_numeric(out["N_corr_hum"], errors="coerce")
    N_int   = pd.to_numeric(out[out_col_int], errors="coerce")
    N_agb   = pd.to_numeric(out[out_col_agb], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        Fp = (N_press / N_raw).replace([np.inf, -np.inf], np.nan)
        Fh = (N_hum / N_press).replace([np.inf, -np.inf], np.nan)

    agb_col = _cfg_get(cfg, "agb_col", None)
    if agb_col and (agb_col in out.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            Fv = (N_agb / N_int).replace([np.inf, -np.inf], np.nan)
    else:
        Fv = pd.Series(1.0, index=out.index)

    # Store factors at full precision; CSV rounding is handled by _prepare_export_frame.
    # Rounding here to 3 d.p. created visible staircase artefacts in slow-changing
    # factors like F_AGB (which may change by only ~0.0001/day during the growing season).
    out["F_PRESSURE"] = pd.to_numeric(Fp, errors="coerce")
    out["F_HUMIDITY"] = pd.to_numeric(Fh, errors="coerce")
    out["F_AGB"]      = pd.to_numeric(Fv, errors="coerce")

    return out


# ==============================================================================
# CALIBRATION (N0) & THETA
# ==============================================================================

def es(T):
    """Saturation vapor pressure (hPa) from air temperature in degC."""
    return 6.112 * np.exp((17.67 * T) / (243.5 + T))


def ea(esat, RH):
    """Actual vapor pressure (hPa) from saturation vapor pressure and RH (%)."""
    return esat * (RH / 100.0)


def pv(ea_hpa, T):
    """Absolute humidity from vapor pressure (hPa) and air temperature (degC)."""
    ea_pa = ea_hpa * 100.0
    return ea_pa / (461.5 * (T + 273.15))


# ~~~~ FUNCTIONS FROM SCHRON ET AL. 2017 USED TO CALIBRATE THE SENSOR ~~~~~#

def WrX(r, x, y):
    """Radial weighting function for point measurements taken within 5 m of sensor."""
    x00 = 3.7
    a00 = 8735
    a01 = 22.689
    a02 = 11720
    a03 = 0.00978
    a04 = 9306
    a05 = 0.003632
    a10 = 2.7925e-002
    a11 = 6.6577
    a12 = 0.028544
    a13 = 0.002455
    a14 = 6.851e-005
    a15 = 12.2755
    a20 = 247970
    a21 = 23.289
    a22 = 374655
    a23 = 0.00191
    a24 = 258552
    a30 = 5.4818e-002
    a31 = 21.032
    a32 = 0.6373
    a33 = 0.0791
    a34 = 5.425e-004

    x0 = x00
    A0 = (a00 * (1 + a03 * x) * np.exp(-a01 * y) + a02 * (1 + a05 * x) - a04 * y)
    A1 = ((-a10 + a14 * x) * np.exp(-a11 * y / (1 + a15 * y)) + a12) * (1 + x * a13)
    A2 = (a20 * (1 + a23 * x) * np.exp(-a21 * y) + a22 - a24 * y)
    A3 = a30 * np.exp(-a31 * y) + a32 - a33 * y + a34 * x

    return (A0 * (np.exp(-A1 * r)) + A2 * np.exp(-A3 * r)) * (1 - np.exp(-x0 * r))


def WrA(r, x, y):
    """Radial weighting function for point measurements taken within 50 m of sensor."""
    a00 = 8735
    a01 = 22.689
    a02 = 11720
    a03 = 0.00978
    a04 = 9306
    a05 = 0.003632
    a10 = 2.7925e-002
    a11 = 6.6577
    a12 = 0.028544
    a13 = 0.002455
    a14 = 6.851e-005
    a15 = 12.2755
    a20 = 247970
    a21 = 23.289
    a22 = 374655
    a23 = 0.00191
    a24 = 258552
    a30 = 5.4818e-002
    a31 = 21.032
    a32 = 0.6373
    a33 = 0.0791
    a34 = 5.425e-004

    A0 = (a00 * (1 + a03 * x) * np.exp(-a01 * y) + a02 * (1 + a05 * x) - a04 * y)
    A1 = ((-a10 + a14 * x) * np.exp(-a11 * y / (1 + a15 * y)) + a12) * (1 + x * a13)
    A2 = (a20 * (1 + a23 * x) * np.exp(-a21 * y) + a22 - a24 * y)
    A3 = a30 * np.exp(-a31 * y) + a32 - a33 * y + a34 * x

    return A0 * (np.exp(-A1 * r)) + A2 * np.exp(-A3 * r)


def WrB(r, x, y):
    """Radial weighting function for point measurements taken over 50 m from sensor."""
    b00 = 39006
    b01 = 15002337
    b02 = 2009.24
    b03 = 0.01181
    b04 = 3.146
    b05 = 16.7417
    b06 = 3727
    b10 = 6.031e-005
    b11 = 98.5
    b12 = 0.0013826
    b20 = 11747
    b21 = 55.033
    b22 = 4521
    b23 = 0.01998
    b24 = 0.00604
    b25 = 3347.4
    b26 = 0.00475
    b30 = 1.543e-002
    b31 = 13.29
    b32 = 1.807e-002
    b33 = 0.0011
    b34 = 8.81e-005
    b35 = 0.0405
    b36 = 26.74

    B0 = (b00 - b01 / (b02 * y + x - 0.13)) * (b03 - y) * np.exp(-b04 * y) - b05 * x * y + b06
    B1 = b10 * (x + b11) + b12 * y
    B2 = (b20 * (1 - b26 * x) * np.exp(-b21 * y * (1 - x * b24)) + b22 - b25 * y) * (2 + x * b23)
    B3 = ((-b30 + b34 * x) * np.exp(-b31 * y / (1 + b35 * x + b36 * y)) + b32) * (2 + x * b33)

    return B0 * (np.exp(-B1 * r)) + B2 * np.exp(-B3 * r)


def D86(r, bd, y):
    """Depth of sensor measurement (86% origin depth)."""
    return (1 / bd * (8.321 + 0.14249 * (0.96655 + np.exp(-0.01 * r)) * (20 + y) / (0.0429 + y)))


def Wd(d, r, bd, y):
    """Depth weighting function."""
    return np.exp(-2 * d / D86(r, bd, y))


def rscaled(r, p, y, Hveg=0):
    """Rescale radius based on pressure, vegetation height and soil moisture."""
    Fp = 0.4922 / (0.86 - np.exp(-p / 1013.25))
    Fveg = 1 - 0.17 * (1 - np.exp(-0.41 * Hveg)) * (1 + np.exp(-9.25 * y))
    return r / Fp / Fveg


def n0_calibration(
    corrected_data,
    country,
    site_id,
    defineaccuracy,
    calib_start_time,
    calib_end_time,
    config_data,
    calib_data_filepath,
    default_dir,
    sm_calc_method
):
    """crspy-lite style N0 calibration transplanted into Petrichor."""

    # --- config mapping: keep Petrichor names, map into crspy-lite variables ---
    bd = _bulk_density_or_none(config_data)
    if bd is None:
        raise ValueError(
            "[CAL] bulk_density is missing. N0 calibration is skipped because "
            "bulk density is required for calibration weighting."
        )

    lw = float(_cfg_get(config_data, "lattice_water", _cfg_get(config_data, "lw", 0.0)))
    soc = float(_cfg_get(config_data, "soil_organic_carbon", _cfg_get(config_data, "soc", 0.0)))

    # convert SOC to water equivelant (see Hawdon et al., 2014)
    wsom = soc * 0.556
    
    Hveg = 0

    # column mapping from Petrichor config
    time_col = TIME_COL
    pressure_col = _cfg_get(config_data, "pressure_col", "PA_1")
    temp_col = _cfg_get(config_data, "temp_col", "TA_1")
    rh_col = _cfg_get(config_data, "rh_col", "RH_1")

    required_vars = list(_cfg_get(config_data, "required_vars", []))
    # AH_1 is always usable as a humidity fallback when the column is present
    # in the corrected data. Previously this was gated on AH_1 appearing in
    # required_vars, which silently disabled the fallback when users only
    # listed RH_1 as required.
    has_ah_fallback = True
    noval = _cfg_get(config_data, "missing_value", -999)
    if isinstance(noval, list):
        noval_scalar = -999
    else:
        noval_scalar = noval

    # output folders in Petrichor style
    data_out = Path(default_dir) / "data"
    fig_out = Path(default_dir) / "figures"
    data_out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~ CALIBRATION DATA READ AND TIDY ~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    print("Fetching calibration data...")
    df = pd.read_csv(calib_data_filepath)

    # Normalize column names first: remove spaces and convert to upper case.
    # This must happen before any column access so that both "date" and "DATE"
    # in the input CSV are handled consistently.
    df.columns = df.columns.astype(str).str.strip().str.upper()

    # Parse calibration timestamps with common date separators.
    # Supported examples:
    # - 21.04.2026 09:00
    # - 21/04/2026 09:00
    # - 21-04-2026 09:00
    calib_raw = (
        df["DATE"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[.\-]", "/", regex=True)
    )

    # First try the normalized explicit format.
    df["DATE"] = pd.to_datetime(
        calib_raw,
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )

    # Fallback to generic day-first parsing for any remaining rows.
    bad_mask = df["DATE"].isna()
    if bad_mask.any():
        df.loc[bad_mask, "DATE"] = pd.to_datetime(
            calib_raw[bad_mask],
            errors="coerce",
            dayfirst=True
        )

    bad_calib_dates = df["DATE"].isna().sum()
    if bad_calib_dates:
        print(f"[CAL][WARN] {bad_calib_dates} calibration timestamps could not be parsed and will be dropped.")

    df = df.dropna(subset=["DATE"]).copy()
    df["DATE"] = df["DATE"].dt.date

    # Validate the columns required by the calibration routine.
    # Required base columns
    required_calib_cols = ["DATE", "DIST", "DEPTH_AVG", "SWV"]
    missing_calib_cols = [c for c in required_calib_cols if c not in df.columns]

    if missing_calib_cols:
        raise KeyError(
            "Calibration CSV is missing required columns: "
            + ", ".join(missing_calib_cols)
        )

    # PROFILE and LOC are alternatives.
    # If PROFILE exists, LOC is not required.
    # If PROFILE does not exist, LOC is needed to create PROFILE.
    if "PROFILE" not in df.columns:
        if "LOC" not in df.columns:
            raise KeyError(
                "Calibration CSV must contain either 'PROFILE' or 'LOC'. "
                "If PROFILE is provided, LOC is not needed."
            )
        profiles = pd.Series(df["LOC"]).dropna().unique().tolist()
        loc_to_profile = {loc: j + 1 for j, loc in enumerate(profiles)}
        df["PROFILE"] = df["LOC"].map(loc_to_profile)

    # Sort unique dates so that per-day iteration order is deterministic.
    unidate = np.array(sorted(df["DATE"].dropna().unique()))
    print("Unique calibration dates found: " + str(unidate))
    print("Done")

    # Use pd.to_numeric (errors='coerce') rather than astype(float) so that
    # stray non-numeric strings become NaN instead of raising ValueError.
    df["DIST"] = pd.to_numeric(df["DIST"], errors="coerce")
    df["DEPTH_AVG"] = pd.to_numeric(df["DEPTH_AVG"], errors="coerce")

    numdays = len(unidate)

    dfCalib = dict()
    for i in range(numdays):
        dfCalib[i] = df.loc[df["DATE"] == unidate[i]]

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~ AVERAGE PRESSURE FOR EACH CALIB DAY ~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    lvl1 = corrected_data.copy()

    # convert Petrichor missing values to NaN
    lvl1 = lvl1.replace(noval, np.nan)
    lvl1 = lvl1.replace(noval_scalar, np.nan)

    # create crspy-lite compatible columns if absent
    if "DT" not in lvl1.columns:
        lvl1["DT"] = pd.to_datetime(lvl1[time_col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    if "PRESS" not in lvl1.columns:
        lvl1["PRESS"] = pd.to_numeric(lvl1[pressure_col], errors="coerce")

    if "TEMP" not in lvl1.columns:
        lvl1["TEMP"] = pd.to_numeric(lvl1[temp_col], errors="coerce")

    if "E_RH" not in lvl1.columns:
        if rh_col in lvl1.columns:
            lvl1["E_RH"] = pd.to_numeric(lvl1[rh_col], errors="coerce")
        else:
            lvl1["E_RH"] = np.nan

    if "E_AH_FLUX" not in lvl1.columns:
        if has_ah_fallback and ("AH_1" in lvl1.columns):
            lvl1["E_AH_FLUX"] = pd.to_numeric(lvl1["AH_1"], errors="coerce")
        else:
            lvl1["E_AH_FLUX"] = np.nan

    if "MOD_CORR" not in lvl1.columns:
        if "N_corr" in lvl1.columns:
            lvl1["MOD_CORR"] = pd.to_numeric(lvl1["N_corr"], errors="coerce")
        else:
            raise KeyError("corrected_data must contain 'N_corr' or 'MOD_CORR' for calibration.")

    lvl1["DATE"] = pd.to_datetime(lvl1["DT"], yearfirst=True, errors="coerce")
    lvl1["DATE"] = lvl1["DATE"].dt.date

    # Only replace the sentinel missing value on numeric columns. The previous
    # frame-wide comparison also touched string/datetime columns (e.g. DT, DATE)
    # which could coerce them in surprising ways.
    numeric_cols = lvl1.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        lvl1[numeric_cols] = lvl1[numeric_cols].mask(lvl1[numeric_cols] == noval_scalar, np.nan)

    dflvl1Days = dict()
    for i in range(numdays):
        dflvl1Days[i] = lvl1.loc[lvl1["DATE"] == unidate[i]]

    def _window_or_day_mean(day_frame: pd.DataFrame, col: str, calib_date, start_time: str, end_time: str) -> float:
        """Return mean over calibration window; if empty, fall back to full day; if still empty, return NaN."""
        if col not in day_frame.columns or len(day_frame) == 0:
            return np.nan

        # Compare as datetimes rather than raw strings: lexicographic string
        # comparison only works when all timestamps share identical formatting.
        dt_series = pd.to_datetime(day_frame["DT"], errors="coerce")
        try:
            t_start = pd.to_datetime(f"{calib_date} {start_time}")
            t_end = pd.to_datetime(f"{calib_date} {end_time}")
        except Exception:
            t_start = None
            t_end = None

        if t_start is not None and t_end is not None:
            in_window = (dt_series > t_start) & (dt_series <= t_end)
            win = day_frame.loc[in_window]
            s_win = pd.to_numeric(win[col], errors="coerce")
            if s_win.notna().any():
                return float(s_win.mean())

        s_day = pd.to_numeric(day_frame[col], errors="coerce")
        if s_day.notna().any():
            return float(s_day.mean())

        return np.nan
    
    avgP = {}
    for i in range(len(dflvl1Days)):
        day_frame = pd.DataFrame.from_dict(dflvl1Days[i])
        avgP[i] = _window_or_day_mean(day_frame, "PRESS", unidate[i], calib_start_time, calib_end_time)

    avgT = {}
    for i in range(len(dflvl1Days)):
        day_frame = pd.DataFrame.from_dict(dflvl1Days[i])
        avgT[i] = _window_or_day_mean(day_frame, "TEMP", unidate[i], calib_start_time, calib_end_time)

    avgAH = {}
    for i in range(len(dflvl1Days)):
        day_frame = pd.DataFrame.from_dict(dflvl1Days[i])
        avgAH[i] = _window_or_day_mean(day_frame, "E_AH_FLUX", unidate[i], calib_start_time, calib_end_time)

    avgRH = {}
    for i in range(len(dflvl1Days)):
        day_frame = pd.DataFrame.from_dict(dflvl1Days[i])
        avgRH[i] = _window_or_day_mean(day_frame, "E_RH", unidate[i], calib_start_time, calib_end_time)

    avgVP = {}
    for i in range(len(dflvl1Days)):
        day_frame = pd.DataFrame.from_dict(dflvl1Days[i])
        avgVP[i] = _window_or_day_mean(day_frame, "VP", unidate[i], calib_start_time, calib_end_time)

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~ THE BIG ITERATIVE LOOP - CALCULATE WEIGHTED THETA ~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    AvgTheta = {}
    successful_day_ids = []

    for i in range(len(dflvl1Days)):
        print("Calibrating to day " + str(i + 1) + "...")

        df1 = pd.DataFrame.from_dict(dfCalib[i]).copy()
        df1["SWV"] = pd.to_numeric(df1["SWV"], errors="coerce")

        # Use median > 1.5 instead of mean > 1 so that a couple of outlier
        # SWV values (e.g. bad probe readings) do not wrongly trigger the
        # percentage-to-decimal conversion on already-decimal data.
        swv_median = df1["SWV"].median()
        if np.isfinite(swv_median) and swv_median > 1.5:
            print("crspy-lite detected SWV is not in decimal format, dividing by 100.")
            df1["SWV"] = df1["SWV"] / 100.0

        CalibTheta = df1["SWV"].mean()
        Accuracy = 1.0

        if "PROFILE" not in df1.columns:
            if "LOC" not in df1.columns:
                raise KeyError(
                    "Calibration data for this day has neither PROFILE nor LOC."
                )

            profiles = pd.Series(df1["LOC"]).dropna().unique().tolist()
            loc_to_id = {loc: j + 1 for j, loc in enumerate(profiles)}
            df1["PROFILE"] = df1["LOC"].map(loc_to_id)

        df1["PROFILE"] = pd.to_numeric(df1["PROFILE"], errors="coerce")

        if df1["PROFILE"].isna().any():
            raise ValueError(
                "PROFILE contains missing or non-numeric values after processing. "
                "Please check calibration CSV."
    )

        # --- choose meteorological inputs day by day ---
        day1temp = avgT.get(i, np.nan)
        day1rh = avgRH.get(i, np.nan)
        day1ah = avgAH.get(i, np.nan)
        day1vp = avgVP.get(i, np.nan)

        if not np.isfinite(day1temp):
            print(f"[CAL][WARN] Temperature is non-finite for day {unidate[i]}. Skipping this calibration day.")
            continue

        hum_source = None

        if np.isfinite(day1rh):
            day1es = es(day1temp)
            day1ea = ea(day1es, day1rh)
            day1hum = pv(day1ea, day1temp) * 1000.0
            hum_source = "RH"

        elif np.isfinite(day1ah):
            day1hum = float(day1ah)
            hum_source = "AH_1"

        elif np.isfinite(day1vp):
            day1hum = pv(day1vp, day1temp) * 1000.0
            hum_source = "VP"

        else:
            print(f"[CAL][WARN] No valid RH, AH_1, or VP for day {unidate[i]}. Skipping this calibration day.")
            continue

        if not np.isfinite(day1hum):
            print(f"[CAL][WARN] Calibration humidity is non-finite for day {unidate[i]}. Skipping this calibration day.")
            continue

        max_iter = 100
        converged = False

        for iter_idx in range(1, max_iter + 1):
            thetainitial = float(CalibTheta)

            if not np.isfinite(thetainitial):
                print(f"[CAL][WARN] Calibration theta became non-finite before iteration for day {unidate[i]}. Skipping this calibration day.")
                break

            # rscaled signature is (r, p, y, Hveg); y is soil moisture (thetainitial),
            # Hveg is vegetation height. Previous order swapped y and Hveg.
            df1["rscale"] = df1.apply(
                lambda row: rscaled(row["DIST"], avgP[i], thetainitial, Hveg), axis=1
            )

            df1["Wd"] = df1.apply(
                lambda row: Wd(row["DEPTH_AVG"], row["rscale"], bd, thetainitial), axis=1
            )
            df1["thetweight"] = df1["SWV"] * df1["Wd"]

            depthdf = df1.groupby("PROFILE", as_index=False)["thetweight"].sum()
            temp = df1.groupby("PROFILE", as_index=False)["Wd"].sum()
            depthdf["Wd_tot"] = temp["Wd"]
            depthdf["Profile_SWV_AVG"] = depthdf["thetweight"] / depthdf["Wd_tot"]

            dictprof = dict(zip(df1.PROFILE, df1.DIST))
            dictprof2 = dict(zip(df1.PROFILE, df1.rscale))

            depthdf["Radius"] = depthdf["PROFILE"].apply(lambda x: dictprof.get(x, None))
            depthdf["rscale"] = depthdf["PROFILE"].apply(lambda x: dictprof2.get(x, None))

            depthdf["day1hum"] = day1hum
            depthdf["TAVG"] = day1temp
            depthdf["Wr"] = np.nan

            mask_b = depthdf["Radius"] > 50
            mask_a = (depthdf["Radius"] > 5) & (depthdf["Radius"] <= 50)
            mask_x = depthdf["Radius"] <= 5

            depthdf.loc[mask_b, "Wr"] = WrB(
                depthdf.loc[mask_b, "rscale"],
                depthdf.loc[mask_b, "day1hum"],
                depthdf.loc[mask_b, "TAVG"] / 100.0,
            )
            depthdf.loc[mask_a, "Wr"] = WrA(
                depthdf.loc[mask_a, "rscale"],
                depthdf.loc[mask_a, "day1hum"],
                depthdf.loc[mask_a, "TAVG"] / 100.0,
            )
            depthdf.loc[mask_x, "Wr"] = WrX(
                depthdf.loc[mask_x, "rscale"],
                depthdf.loc[mask_x, "day1hum"],
                depthdf.loc[mask_x, "TAVG"] / 100.0,
            )

            depthdf["RadWeight"] = depthdf["Profile_SWV_AVG"] * depthdf["Wr"]

            wr_sum = pd.to_numeric(depthdf["Wr"], errors="coerce").sum(min_count=1)
            rad_sum = pd.to_numeric(depthdf["RadWeight"], errors="coerce").sum(min_count=1)

            if (not np.isfinite(wr_sum)) or abs(wr_sum) < 1e-12:
                print(f"[CAL][WARN] Calibration weighting failed for day {unidate[i]}. Skipping this calibration day.")
                break

            CalibTheta_new = rad_sum / wr_sum

            if not np.isfinite(CalibTheta_new):
                print(f"[CAL][WARN] Calibration theta became non-finite for day {unidate[i]}. Skipping this calibration day.")
                break

            Accuracy = abs((CalibTheta_new - thetainitial) / max(abs(thetainitial), 1e-12))
            # Clip the iterate to a physically plausible volumetric range so
            # that a single bad iteration cannot drive the fixed-point search
            # into nonsensical territory (e.g. negative or > porosity).
            CalibTheta = float(np.clip(CalibTheta_new, 1e-4, 0.9))

            if iter_idx % 10 == 0 or Accuracy <= defineaccuracy:
                print(
                    f"[CAL][ITER] day={unidate[i]} source={hum_source} iter={iter_idx} "
                    f"theta={CalibTheta:.6f} accuracy={Accuracy:.6e}"
                )

            if Accuracy <= defineaccuracy:
                converged = True
                break

        if not converged:
            continue

        AvgTheta[i] = CalibTheta
        successful_day_ids.append(i)
        print(f"Done ({hum_source})")

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~ OPTIMISED N0 CALCULATION ~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    print("Finding Optimised N0......")

    tmp = corrected_data.copy()
    tmp = tmp.replace(noval, np.nan)
    tmp = tmp.replace(noval_scalar, np.nan)

    if "MOD_CORR" not in tmp.columns:
        if "N_corr" in tmp.columns:
            tmp["MOD_CORR"] = pd.to_numeric(tmp["N_corr"], errors="coerce")
        else:
            raise KeyError("corrected_data must contain 'N_corr' or 'MOD_CORR' for calibration.")

    tmp["MOD_CORR"] = tmp["MOD_CORR"].replace([np.inf, -np.inf], np.nan)
    tmp["MOD_CORR"] = pd.to_numeric(tmp["MOD_CORR"], errors="coerce")

    n_avg_raw = float(np.nanmean(tmp["MOD_CORR"]))

    if np.isnan(n_avg_raw) or np.isinf(n_avg_raw) or n_avg_raw <= 0:
        # Without a meaningful site-average neutron count we cannot build the
        # N0 search range; abort loudly rather than silently using n_avg=0.
        raise ValueError(
            "MOD_CORR contains only NaN/infinite values or non-positive mean. "
            "Cannot determine a search range for N0."
        )

    n_avg = int(n_avg_raw)
    print(f"Avg N for the site is {n_avg}")
    # Note: crspy-lite previously had a heuristic to cap n_avg at 3000 when
    # n_avg >= 4000. This is intentionally disabled in Petrichor; the search
    # range adapts to the actual site-average neutron count.

    if "DT" not in tmp.columns:
        tmp["DT"] = pd.to_datetime(tmp[time_col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    tmp["DATE"] = pd.to_datetime(tmp["DT"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    tmp["DATE"] = tmp["DATE"].dt.date

    NeutCount = dict()
    for i in range(numdays):
        tmpneut = tmp.loc[tmp["DATE"] == unidate[i]]
        NeutCount[i] = tmpneut

    avgN = dict()
    for i in range(len(NeutCount)):
        tmp_day = pd.DataFrame.from_dict(NeutCount[i])
        # Compare DT as datetime rather than as string; the previous string
        # comparison assumed all DT values were ISO-formatted.
        dt_series = pd.to_datetime(tmp_day["DT"], errors="coerce")
        try:
            t_start = pd.to_datetime(f"{unidate[i]} {calib_start_time}")
            t_end = pd.to_datetime(f"{unidate[i]} {calib_end_time}")
            in_window = (dt_series > t_start) & (dt_series <= t_end)
            win_df = tmp_day.loc[in_window]
        except Exception:
            win_df = tmp_day.iloc[0:0]

        mod_win = pd.to_numeric(win_df["MOD_CORR"], errors="coerce") if len(win_df) else pd.Series(dtype=float)
        check = float(np.nanmean(mod_win)) if mod_win.notna().any() else float("nan")

        if np.isnan(check):
            mod_day = pd.to_numeric(tmp_day["MOD_CORR"], errors="coerce")
            avgN[i] = float(np.nanmean(mod_day)) if mod_day.notna().any() else float("nan")
        else:
            avgN[i] = check

    if len(successful_day_ids) == 0:
        raise ValueError("No valid calibration days available after meteorological screening.")

    RelerrDict = {}
    valid_n0_day_ids = []

    with np.errstate(divide="ignore", invalid="ignore"):
        for i in successful_day_ids:
            vwc = AvgTheta.get(i, np.nan)
            Nave = avgN.get(i, np.nan)

            # A calibration day can have valid soil samples and meteorology but no
            # valid corrected neutron counts. Such a day cannot be used for N0 optimisation.
            if (not np.isfinite(vwc)) or (not np.isfinite(Nave)) or Nave <= 0:
                print(
                    f"[CAL][WARN] Calibration day {unidate[i]} has no valid "
                    f"corrected neutron count for N0 search "
                    f"(theta={vwc}, Nave={Nave}). This day will be excluded."
                )
                continue

            a0_cfg = float(_cfg_get(config_data, "a0", 0.0808))
            a1_cfg = float(_cfg_get(config_data, "a1", 0.372))
            a2_cfg = float(_cfg_get(config_data, "a2", 0.115))

            # Fixed N0 search range.
            # N0 = 0 is excluded from calculation to avoid division by zero in the soil-moisture equation.
            n0_lower = 1
            n0_upper = 10000
            N0_arr = np.arange(n0_lower, n0_upper + 1, dtype=float)

            if sm_calc_method == "desilets":
                sm_arr = desilets_sm_calc(
                    a0_cfg, a1_cfg, a2_cfg,
                    bd, float(Nave), N0_arr, lw, wsom
                )
            elif sm_calc_method == "kohli":
                sm_arr = kohli_sm_calc(
                    a0_cfg, a1_cfg, a2_cfg,
                    bd, float(Nave), N0_arr, lw, wsom
                )
            else:
                raise KeyError(
                    "No other method to calculate soil moisture at this time. "
                    "Please use either 'desilets' or 'kohli' method."
                )

            relerr = pd.Series(np.abs(sm_arr - float(vwc)))
            relerr = relerr.replace([np.inf, -np.inf], np.nan)

            if not np.isfinite(relerr).any():
                print(
                    f"[CAL][WARN] Calibration day {unidate[i]} produced no finite "
                    "N0 error values. This day will be excluded."
                )
                continue

            RelerrDict[i] = relerr
            valid_n0_day_ids.append(i)

    if len(valid_n0_day_ids) == 0:
        raise ValueError(
            "No calibration days have both valid weighted theta and valid corrected "
            "neutron counts for N0 search."
        )

    totalerror = RelerrDict[valid_n0_day_ids[0]].copy()
    for i in valid_n0_day_ids[1:]:
        totalerror = totalerror.add(RelerrDict[i], fill_value=0.0)

    totalerror = totalerror.replace([np.inf, -np.inf], np.nan)

    if not np.isfinite(totalerror).any():
        raise ValueError(
            "N0 search failed because all total error values are NaN or infinite. "
            "Please check calibration-day MOD_CORR availability and calibration SWV."
        )

    totalerror = totalerror.to_frame(name="RelErr")
    totalerror["N0"] = range(n0_lower, n0_upper + 1)
    totalerror = totalerror.dropna(subset=["RelErr"])

    best_idx = totalerror["RelErr"].idxmin()
    N0 = int(totalerror.loc[best_idx, "N0"])

    # Warn if the optimum lies at the edge of the search range: the true minimum
    # may lie outside [n_avg, 2.5 * n_avg] and the result could be biased.
    if N0 <= n0_lower or N0 >= n0_upper:
        print(
            f"[CAL][WARN] Optimum N0={N0} lies at the edge of the fixed search range "
            f"[0, {n0_upper}]. Consider widening the range or reviewing the calibration data."
        )
        
    plt.figure()
    plt.plot(totalerror["N0"], totalerror["RelErr"])
    plt.yscale("log")
    plt.xlabel("N0")
    plt.ylabel("Sum Relative Error (log scale)")
    plt.title("Sum Relative Error plot on log scale across all calibration days")
    plt.savefig(fig_out / f"{country}_{site_id}Relative_Error_Plot.png")
    plt.close()

    return N0
# ==============================================================================
#  compute soil moisture
# ==============================================================================
def desilets_sm_calc(a0, a1, a2, bd, N, N0, lw, wsom):
    sm = (((a0) / ((N / N0) - a1)) - (a2) - lw - wsom) * bd
    return sm


def kohli_sm_calc(a0, a1, a2, bd, N, N0, lw, wsom):
    Nmax = N0 * ((a0 + (a1 * a2)) / a2)
    ah0 = -a2
    ah1 = (a1 * a2) / (a0 + (a1 * a2))
    sm = ((ah0 * ((1 - (N / Nmax)) / (ah1 - (N / Nmax)))) - lw - wsom) * bd
    return sm

def sm_max_calc(bd: float, density: float = PARTICLE_DENSITY) -> float:
    """
    Calculate maximum physically plausible volumetric soil moisture from porosity.

    Parameters
    ----------
    bd : float
        Bulk density (g cm^-3).
    density : float, default 2.65
        Particle density (g cm^-3). Petrichor uses the application-wide
        constant 2.65; the argument remains available for direct scientific
        use of this standalone helper.

    Returns
    -------
    float
        Maximum plausible soil moisture (cm^3 cm^-3).
    """
    return 1.0 - (float(bd) / float(density))

def crns_to_soil_moisture(df: pd.DataFrame, cfg: dict,
                          in_col: str = "N_corr",
                          out_col: str = "SWC",
                          method: str = "desilets") -> pd.DataFrame:
    """Convert corrected neutron counts to volumetric soil moisture."""
    out = df.copy()

    if in_col not in out.columns:
        raise KeyError(f"[SWC] Missing '{in_col}'.")

    N0 = _cfg_get(cfg, "N0", None)
    if N0 in (None, "", "null"):
        raise ValueError("[SWC] N0 is not set in config.")

    bd = _bulk_density_or_none(cfg)
    if bd is None:
        print("[SWC][WARN] bulk_density is missing. Soil moisture calculation is skipped.")
        out[out_col] = np.nan
        out["SM"] = np.nan
        return out

    a0 = float(_cfg_get(cfg, "a0", 0.0808))
    a1 = float(_cfg_get(cfg, "a1", 0.372))
    a2 = float(_cfg_get(cfg, "a2", 0.115))
    lw = float(_cfg_get(cfg, "lattice_water", 0.0))
    soc = float(_cfg_get(cfg, "soil_organic_carbon", 0.0))

    # convert SOC to water equivelant (see Hawdon et al., 2014)
    wsom = soc * 0.556

    N = pd.to_numeric(out[in_col], errors="coerce")

    if method.lower() == "kohli":
        swc = kohli_sm_calc(a0, a1, a2, bd, N, float(N0), lw, wsom)
    else:
        swc = desilets_sm_calc(a0, a1, a2, bd, N, float(N0), lw, wsom)

    out[out_col] = swc
    out["SM"] = swc
    return out

def mod_error(mod: pd.Series) -> pd.Series:
    """
    Estimate neutron count uncertainty following crspy-lite.

    Parameters
    ----------
    mod : pd.Series
        Raw neutron counts (typically MOD / chosen neutron count column).

    Returns
    -------
    pd.Series
        Estimated neutron count error.
    """
    mod_num = pd.to_numeric(mod, errors="coerce")
    m = mod_num.mean()
    sd = np.sqrt(np.abs(mod_num))
    cv = sd / m
    return mod_num * cv

def apply_sm_uncertainty(df: pd.DataFrame, cfg: dict,
                         mod_col: str = "MOD",
                         mod_corr_col: str = "N_corr",
                         sm_col: str = "SM") -> pd.DataFrame:
    """
    Apply crspy-lite style soil moisture uncertainty propagation.

    Output columns:
    - MOD_ERR
    - MOD_CORR_PLUS
    - MOD_CORR_MINUS
    - SM_PLUS_ERR
    - SM_MINUS_ERR

    Note:
    SM_PLUS_ERR / SM_MINUS_ERR are stored as absolute deviations from SM,
    consistent with crspy-lite internals.
    """
    out = df.copy()

    if mod_col not in out.columns:
        raise KeyError(f"[UNC] Missing raw neutron column '{mod_col}'.")
    if mod_corr_col not in out.columns:
        raise KeyError(f"[UNC] Missing corrected neutron column '{mod_corr_col}'.")
    if sm_col not in out.columns:
        raise KeyError(f"[UNC] Missing soil moisture column '{sm_col}'.")

    a0 = float(_cfg_get(cfg, "a0", 0.0808))
    a1 = float(_cfg_get(cfg, "a1", 0.372))
    a2 = float(_cfg_get(cfg, "a2", 0.115))

    bd = _bulk_density_or_none(cfg)
    if bd is None:
        print("[UNC][WARN] bulk_density is missing. Soil moisture uncertainty calculation is skipped.")
        out["MOD_ERR"] = np.nan
        out["MOD_CORR_PLUS"] = np.nan
        out["MOD_CORR_MINUS"] = np.nan
        out["SM_PLUS_ERR"] = np.nan
        out["SM_MINUS_ERR"] = np.nan
        return out

    lw = float(_cfg_get(cfg, "lattice_water", 0.0))
    soc = float(_cfg_get(cfg, "soil_organic_carbon", 0.0))

    # convert SOC to water equivelant (see Hawdon et al., 2014)
    wsom = soc * 0.556

    N0 = float(_cfg_get(cfg, "N0"))
    method = str(_cfg_get(cfg, "theta_method", "desilets")).lower()
    sm_max_cfg = _cfg_get(cfg, "sm_max", None)
    if sm_max_cfg in (None, "", "null"):
        sm_max = sm_max_calc(bd=bd)
    else:
        sm_max = float(sm_max_cfg)
    sm_min = float(_cfg_get(cfg, "qc_theta_min", 0.0))

    out["MOD_ERR"] = np.round(mod_error(out[mod_col]))
    out["MOD_CORR_PLUS"] = pd.to_numeric(out[mod_corr_col], errors="coerce") + pd.to_numeric(out["MOD_ERR"], errors="coerce")
    out["MOD_CORR_MINUS"] = pd.to_numeric(out[mod_corr_col], errors="coerce") - pd.to_numeric(out["MOD_ERR"], errors="coerce")

    if method == "kohli":
        sm_plus = kohli_sm_calc(a0, a1, a2, bd, out["MOD_CORR_MINUS"], N0, lw, wsom)
        sm_minus = kohli_sm_calc(a0, a1, a2, bd, out["MOD_CORR_PLUS"], N0, lw, wsom)
    else:
        sm_plus = desilets_sm_calc(a0, a1, a2, bd, out["MOD_CORR_MINUS"], N0, lw, wsom)
        sm_minus = desilets_sm_calc(a0, a1, a2, bd, out["MOD_CORR_PLUS"], N0, lw, wsom)

    sm_center = pd.to_numeric(out[sm_col], errors="coerce")

    # store as deviations from center, same as crspy-lite
    out["SM_PLUS_ERR"] = np.abs(pd.to_numeric(sm_plus, errors="coerce") - sm_center)
    out["SM_MINUS_ERR"] = np.abs(pd.to_numeric(sm_minus, errors="coerce") - sm_center)

    # constrain error amplitudes using resulting edge values
    plus_edge = sm_center + out["SM_PLUS_ERR"]
    minus_edge = sm_center - out["SM_MINUS_ERR"]

    plus_edge = plus_edge.clip(lower=sm_min, upper=sm_max)
    minus_edge = minus_edge.clip(lower=sm_min, upper=sm_max)

    out["SM_PLUS_ERR"] = np.abs(plus_edge - sm_center)
    out["SM_MINUS_ERR"] = np.abs(sm_center - minus_edge)

    return out

def apply_sm_smoothing(df: pd.DataFrame, cfg: dict,
                       sm_col: str = "SM",
                       out_col: str = "SM_SMOOTH") -> pd.DataFrame:
    """
    Apply Petrichor's standard rolling smoothing to soil moisture.

    Smoothing is always enabled application-wide. The rolling-window length
    remains controlled by ``smwindow`` in the site JSON.

    Rules
    -----
    - If sm_col does not exist, return df unchanged.
    - The unsmoothed hourly values remain in sm_col.
    - The smoothed values are written to out_col.
    """
    out = df.copy()

    smwindow = int(_cfg_get(cfg, "smwindow", 12))

    if sm_col not in out.columns:
        return out

    sm = pd.to_numeric(out[sm_col], errors="coerce")

    # use a moderate min_periods so early points are not all NaN
    min_periods = max(1, smwindow // 2)

    out[out_col] = sm.rolling(
        window=smwindow,
        min_periods=min_periods
    ).mean()

    return out

# ==============================================================================
#  Depth
# ==============================================================================

def compute_effective_depth(df: pd.DataFrame,
                            cfg: dict,
                            sm_col: str = "SM",
                            sm_smooth_col: str = "SM_SMOOTH") -> pd.DataFrame:
    """
    Compute effective sensing depth metrics from soil moisture and pressure.

    Output columns:
    - D86_10m
    - D86_75m
    - D86_150m
    - D86avg
    - D86avg_12h
    """
    out = df.copy()

    bd = _bulk_density_or_none(cfg)
    pressure_col = _cfg_get(cfg, "pressure_col", "PA_1")

    if bd is None:
        print("[DEPTH][WARN] bulk_density is missing. Skipping D86 calculation.")
        out["D86_10m"] = np.nan
        out["D86_75m"] = np.nan
        out["D86_150m"] = np.nan
        out["D86avg"] = np.nan
        out["D86avg_12h"] = np.nan
        return out

    if pressure_col not in out.columns:
        print(f"[DEPTH] Pressure column '{pressure_col}' not found. Skipping D86 calculation.")
        out["D86_10m"] = np.nan
        out["D86_75m"] = np.nan
        out["D86_150m"] = np.nan
        out["D86avg"] = np.nan
        out["D86avg_12h"] = np.nan
        return out

    P = pd.to_numeric(out[pressure_col], errors="coerce")

    sm_raw = pd.to_numeric(out[sm_col], errors="coerce")

    if sm_smooth_col in out.columns:
        sm_smooth = pd.to_numeric(out[sm_smooth_col], errors="coerce")
        sm_for_depth = sm_smooth.fillna(sm_raw)
    else:
        sm_for_depth = sm_raw

    rs10m = rscaled(10.0, P, sm_for_depth, Hveg=0.0)
    rs75m = rscaled(75.0, P, sm_for_depth, Hveg=0.0)
    rs150m = rscaled(150.0, P, sm_for_depth, Hveg=0.0)

    out["D86_10m"] = D86(rs10m, bd, sm_for_depth)
    out["D86_75m"] = D86(rs75m, bd, sm_for_depth)
    out["D86_150m"] = D86(rs150m, bd, sm_for_depth)

    out["D86avg"] = (
        pd.to_numeric(out["D86_10m"], errors="coerce") +
        pd.to_numeric(out["D86_75m"], errors="coerce") +
        pd.to_numeric(out["D86_150m"], errors="coerce")
    ) / 3.0

    out["D86avg_12h"] = pd.to_numeric(out["D86avg"], errors="coerce").rolling(
        window=12,
        min_periods=6
    ).mean()

    return out

def thetaprocess_petrichor(df: pd.DataFrame,
                           cfg: dict,
                           n_col: str = "N_corr",
                           sm_col: str = "SM",
                           sm_smooth_col: str = "SM_SMOOTH") -> pd.DataFrame:
    """
    Petrichor adaptation of crspy-lite theta processing.
    """
    out = df.copy()

    if n_col not in out.columns:
        raise KeyError(f"[THETA] Missing neutron column '{n_col}'.")

    out = crns_to_soil_moisture(
        df=out,
        cfg=cfg,
        in_col=n_col,
        out_col=sm_col,
        method=str(_cfg_get(cfg, "theta_method", "desilets"))
    )

    bd = float(_cfg_get(cfg, "bulk_density", 1.43))

    sm_min = float(_cfg_get(cfg, "qc_theta_min", 0.0))

    sm_max_cfg = _cfg_get(cfg, "sm_max", None)
    if sm_max_cfg in (None, "", "null"):
        sm_max = sm_max_calc(bd=bd)
    else:
        sm_max = float(sm_max_cfg)

    out[sm_col] = pd.to_numeric(out[sm_col], errors="coerce")
    out[sm_col] = out[sm_col].clip(lower=sm_min, upper=sm_max)

    out = apply_sm_smoothing(out, cfg, sm_col=sm_col, out_col=sm_smooth_col)
    out = compute_effective_depth(out, cfg, sm_col=sm_col, sm_smooth_col=sm_smooth_col)

    return out

# ==============================================================================
# QUALITY CONTROL
# ==============================================================================
def apply_crns_qc(
    df: pd.DataFrame,
    cfg: dict,
    raw_n_col: str = "N",
    corr_n_col: str = "N_corr",
    batt_col: str = "BATT"
) -> pd.DataFrame:
    """
    Apply crspy-lite-like sequential QC.

    Logic
    -----
    1) Remove rows with corrected neutron count > 1.075 * N0  -> FLAG 3
    2) Remove rows with corrected neutron count < belowN0% * N0 -> FLAG 2
    3) Remove rows with battery voltage < 10 V -> FLAG 4
    4) On the remaining rows only, compute raw-count jump QC -> FLAG 1
    5) Merge flags back to the full hourly dataframe and create QC_OK
    """
    out = df.copy()

    time_col = TIME_COL
    below_n0_frac = float(_cfg_get(cfg, "belowN0", 30)) / 100.0
    timestepdiff_frac = float(_cfg_get(cfg, "timestepdiff", 20)) / 100.0

    if "QC_FLAG1" not in out.columns:
        out["QC_FLAG1"] = False
    else:
        out["QC_FLAG1"] = False

    if "QC_FLAG2" not in out.columns:
        out["QC_FLAG2"] = False
    else:
        out["QC_FLAG2"] = False

    if "QC_FLAG3" not in out.columns:
        out["QC_FLAG3"] = False
    else:
        out["QC_FLAG3"] = False

    if "QC_FLAG4" not in out.columns:
        out["QC_FLAG4"] = False
    else:
        out["QC_FLAG4"] = False

    out["FLAG"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")

    N0_val = _cfg_get(cfg, "N0", None)
    if N0_val in (None, "", "null"):
        print("[QC] N0 missing -> only jump and battery checks can be applied.")
        N0_val = None
    else:
        N0_val = float(N0_val)

    # Keep a stable row identifier so we can flag rows after sequential filtering
    out["_ROWID_"] = np.arange(len(out))

    work = out.copy()

    if time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.sort_values(time_col).reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 1: high corrected neutron count -> FLAG 3
    # ------------------------------------------------------------------
    if (N0_val is not None) and (corr_n_col in work.columns):
        n_corr = pd.to_numeric(work[corr_n_col], errors="coerce")
        bad_high = n_corr > (1.075 * N0_val)

        if bad_high.any():
            bad_ids = work.loc[bad_high, "_ROWID_"].to_numpy()
            out.loc[out["_ROWID_"].isin(bad_ids), "QC_FLAG3"] = True
            out.loc[out["_ROWID_"].isin(bad_ids), "FLAG"] = 3

        work = work.loc[~bad_high].copy()

    # ------------------------------------------------------------------
    # Step 2: low corrected neutron count -> FLAG 2
    # ------------------------------------------------------------------
    if (N0_val is not None) and (corr_n_col in work.columns):
        n_corr = pd.to_numeric(work[corr_n_col], errors="coerce")
        bad_low = n_corr < (below_n0_frac * N0_val)

        if bad_low.any():
            bad_ids = work.loc[bad_low, "_ROWID_"].to_numpy()
            out.loc[out["_ROWID_"].isin(bad_ids), "QC_FLAG2"] = True
            out.loc[out["_ROWID_"].isin(bad_ids), "FLAG"] = 2

        work = work.loc[~bad_low].copy()

    # ------------------------------------------------------------------
    # Step 3: low battery voltage -> FLAG 4
    # ------------------------------------------------------------------
    if batt_col in work.columns:
        batt = pd.to_numeric(work[batt_col], errors="coerce")
        bad_batt = batt < 10.0

        if bad_batt.any():
            bad_ids = work.loc[bad_batt, "_ROWID_"].to_numpy()
            out.loc[out["_ROWID_"].isin(bad_ids), "QC_FLAG4"] = True
            out.loc[out["_ROWID_"].isin(bad_ids), "FLAG"] = 4

        work = work.loc[~bad_batt].copy()

    # ------------------------------------------------------------------
    # Step 4: raw neutron jump on the already-filtered remaining rows -> FLAG 1
    # ------------------------------------------------------------------
    if raw_n_col in work.columns:
        raw_n = pd.to_numeric(work[raw_n_col], errors="coerce")
        prev_raw = raw_n.shift(1)

        with np.errstate(divide="ignore", invalid="ignore"):
            prcntdiff = (raw_n - prev_raw) / prev_raw

        bad_jump = prcntdiff.abs() > timestepdiff_frac
        bad_jump = bad_jump.fillna(False)

        if bad_jump.any():
            bad_ids = work.loc[bad_jump, "_ROWID_"].to_numpy()
            out.loc[out["_ROWID_"].isin(bad_ids), "QC_FLAG1"] = True
            out.loc[out["_ROWID_"].isin(bad_ids), "FLAG"] = 1

        work = work.loc[~bad_jump].copy()

    # ------------------------------------------------------------------
    # Missing-data failure
    # Only flag rows where the raw neutron count N is missing.
    # Missing BATT or missing corr_n are not treated as failures:
    # - Missing BATT just means the battery check is skipped for that row.
    # - corr_n NaN typically follows from N being NaN anyway.
    # ------------------------------------------------------------------
    missing_fail = pd.Series(False, index=out.index)

    if raw_n_col in out.columns:
        missing_fail = missing_fail | pd.to_numeric(out[raw_n_col], errors="coerce").isna()

    out["QC_OK"] = ~(
        out["QC_FLAG1"] |
        out["QC_FLAG2"] |
        out["QC_FLAG3"] |
        out["QC_FLAG4"] |
        missing_fail
    )

    # Mask the corrected neutron count used downstream
    if corr_n_col in out.columns:
        out.loc[~out["QC_OK"], corr_n_col] = np.nan

    # Also mask N_corr when MOD_CORR is the active QC/theta column,
    # so downstream diagnostics are less confusing.
    if (corr_n_col == "MOD_CORR") and ("N_corr" in out.columns):
        out.loc[~out["QC_OK"], "N_corr"] = np.nan

    total = len(out)
    ok = int(out["QC_OK"].sum())
    if total > 0:
        print(f"[QC] crspy-lite-like QC applied: {ok}/{total} ({ok/total*100:.1f}%) kept")

    out = out.drop(columns=["_ROWID_"], errors="ignore")
    return out

# ==============================================================================
# RUN QC and THETA
# ==============================================================================

def run_crns_qc_and_theta_pipeline(df: pd.DataFrame,
                                   cfg: dict,
                                   neutron_col: str,
                                   corr_n_col: str = "N_corr",
                                   swc_crns_col: str = "SWC_CRNS") -> pd.DataFrame:
    """
    Run QC, theta processing, and uncertainty using existing helpers.

    This wrapper keeps main.py lightweight and avoids duplicating
    soil-moisture / smoothing / depth logic.
    """
    out = df.copy()

    # ------------------------------------------------------------------
    # 1) QC
    # ------------------------------------------------------------------
    out = apply_crns_qc(
        df=out,
        cfg=cfg,
        raw_n_col="N",
        corr_n_col=corr_n_col,
        batt_col="BATT"
    )

    # ------------------------------------------------------------------
    # 2) Theta + smoothing + depth
    # ------------------------------------------------------------------
    N0_val = _cfg_get(cfg, "N0", None)
    bd = _bulk_density_or_none(cfg)

    if N0_val in (None, "", "null"):
        print("[SWC] N0 missing. Soil moisture cannot be computed.")
        out[swc_crns_col] = np.nan
        out["SM"] = np.nan
        out["SM_SMOOTH"] = np.nan
        out["D86_10m"] = np.nan
        out["D86_75m"] = np.nan
        out["D86_150m"] = np.nan
        out["D86avg"] = np.nan
        out["D86avg_12h"] = np.nan
        return out

    if bd is None:
        print(
            "[SWC][WARN] bulk_density is missing. "
            "QC and corrected neutron counts are kept, but SM, D86, and SM uncertainty are skipped."
        )
        out[swc_crns_col] = np.nan
        out["SM"] = np.nan
        out["SM_SMOOTH"] = np.nan
        out["D86_10m"] = np.nan
        out["D86_75m"] = np.nan
        out["D86_150m"] = np.nan
        out["D86avg"] = np.nan
        out["D86avg_12h"] = np.nan

        if "N" in out.columns:
            out["MOD"] = pd.to_numeric(out["N"], errors="coerce")
        else:
            out["MOD"] = pd.to_numeric(out[neutron_col], errors="coerce")

        return out
    try:
        out = thetaprocess_petrichor(
            df=out,
            cfg=cfg,
            n_col=corr_n_col,
            sm_col="SM",
            sm_smooth_col="SM_SMOOTH"
        )

        # keep SWC_CRNS for compatibility with your export/output logic
        out[swc_crns_col] = pd.to_numeric(out["SM"], errors="coerce")

        if "QC_OK" in out.columns:
            out.loc[~out["QC_OK"], swc_crns_col] = np.nan
            out.loc[~out["QC_OK"], "SM"] = np.nan
            if "SM_SMOOTH" in out.columns:
                out.loc[~out["QC_OK"], "SM_SMOOTH"] = np.nan

        # ------------------------------------------------------------------
        # 3) Uncertainty
        # ------------------------------------------------------------------
        try:
            if "N" in out.columns:
                out["MOD"] = pd.to_numeric(out["N"], errors="coerce")
            else:
                out["MOD"] = pd.to_numeric(out[neutron_col], errors="coerce")
            out = apply_sm_uncertainty(
                df=out,
                cfg=cfg,
                mod_col="MOD",
                mod_corr_col=corr_n_col,
                sm_col="SM"
            )
        except Exception as e:
            print(f"[UNC][WARN] Soil moisture uncertainty calculation failed: {e}")

        total = len(out)
        valid = int(pd.to_numeric(out["SM"], errors="coerce").notna().sum())
        if total > 0:
            print(f"[SWC] Soil moisture computed: {valid}/{total} valid points ({valid/total*100:.1f}%)")

    except Exception as e:
        print(f"[SWC][ERROR] Soil moisture estimation failed: {e}")
        out[swc_crns_col] = np.nan
        out["SM"] = np.nan
        out["SM_SMOOTH"] = np.nan
        out = compute_effective_depth(out, cfg, sm_col="SM", sm_smooth_col="SM_SMOOTH")

    return out

# ==============================================================================
#  UTILITIES
# ==============================================================================

def load_nmdb_table(default_dir: str) -> Optional[pd.DataFrame]:
    """Load previously downloaded nmdb_station_counts.txt."""
    path = Path(default_dir)  / "data" / "nmdb_station_counts.txt"
    if not path.exists():
        print(f"[NMDB] File not found: {path}")
        return None
    try:
        df = pd.read_csv(path, header=None, delimiter=";")
        if df.shape[1] == 2:
            df.columns = ["DT", "N_COUNT"]
        else:
            df = pd.read_csv(path, sep=";", names=["DT", "N_COUNT"], engine="python")
        df["DT"] = pd.to_datetime(df["DT"], errors="coerce")
        df["N_COUNT"] = pd.to_numeric(df["N_COUNT"], errors="coerce")
        df = df.dropna(subset=["DT"])
        print(f"[NMDB] Loaded table: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[NMDB] Failed to load table: {e}")
        return None


def ensure_plot_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Populate standard plotting columns if missing."""
    out = df.copy()
    time_col = TIME_COL
    if "DATETIME" not in out.columns:
        out["DATETIME"] = pd.to_datetime(out.get(time_col, pd.NaT), errors="coerce")
    raw_col = NEUTRON_COL
    if "MOD" not in out.columns and raw_col in out.columns:
        out["MOD"] = pd.to_numeric(out[raw_col], errors="coerce")
    for cand in ["N_corr_agb", "N_corr_int", "N_corr_hum", "N_corr_press", raw_col]:
        if "MOD_CORR" not in out.columns and cand in out.columns:
            out["MOD_CORR"] = pd.to_numeric(out[cand], errors="coerce")
            break
    out["F_PRESSURE"]  = out.get("F_PRESSURE",  1.0)
    out["F_HUMIDITY"]  = out.get("F_HUMIDITY",  1.0)
    out["F_INTENSITY"] = out.get("F_INTENSITY", 1.0)
    out["F_AGB"]       = out.get("F_AGB", 1.0)

    if "SM" not in out.columns and "SWC" in out.columns:
        out["SM"] = pd.to_numeric(out["SWC"], errors="coerce")
    if "SM" in out.columns:
        if "SM_PLUS_ERR" not in out.columns:
            out["SM_PLUS_ERR"] = pd.NA
        if "SM_MINUS_ERR" not in out.columns:
            out["SM_MINUS_ERR"] = pd.NA
    if "D86avg" not in out.columns:
        out["D86avg"] = pd.NA
    return out

# ==============================================================================
#  write summary
# ==============================================================================
def write_site_summary(site_out: Path,
                       site_id: str,
                       cfg: dict,
                       Rc: Optional[float] = None,
                       beta_B: Optional[float] = None,
                       p0_ref: Optional[float] = None,
                       rhov0_ref: Optional[float] = None,
                       jung_ref: Optional[float] = None,
                       N0: Optional[float] = None,
                       theta_method: str = "desilets",
                       neutron_col: Optional[str] = None,
                       extra: Optional[dict] = None,
                       df: Optional[pd.DataFrame] = None,
                       input_path: Optional[Path] = None,
                       field_json_path: Optional[Path] = None) -> None:
    """
    Write a detailed one-run site summary with sections and glossary.
    """
    logs_dir = site_out / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    created_uk = datetime.now(ZoneInfo("Europe/London")).strftime("%Y-%m-%d %H:%M:%S %Z")
    summary_path = logs_dir / "site_summary.txt"

    time_col = TIME_COL
    neutron_col = neutron_col or NEUTRON_COL
    pressure_col = _cfg_get(cfg, "pressure_col", "PA_1")
    temp_col = _cfg_get(cfg, "temp_col", "TA_1")
    rh_col = _cfg_get(cfg, "rh_col", "RH_1")
    dew_col = _cfg_get(cfg, "dewpoint_col", None)
    agb_col = _cfg_get(cfg, "agb_col", None)

    def _fmt_value(value, mode: str = "auto") -> str:
        """Format values for summary output without forcing fixed decimals."""
        if value is None:
            return "NA"

        if isinstance(value, bool):
            return str(value)

        if mode == "int":
            try:
                return str(int(round(float(value))))
            except Exception:
                return str(value)

        if isinstance(value, (int, np.integer)):
            return str(value)

        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return str(value)
            return format(float(value), ".15g")

        return str(value)

    def _fmt_dt(value) -> str:
        """Format datetime-like values."""
        if value is None:
            return "NA"
        try:
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isna(ts):
                return "NA"
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(value)

    def _series_summary(frame: pd.DataFrame, col: str) -> dict:
        """Return count/min/mean/max for one numeric column."""
        if frame is None or col not in frame.columns:
            return {"count": 0, "min": None, "mean": None, "max": None}
        s = pd.to_numeric(frame[col], errors="coerce")
        return {
            "count": int(s.notna().sum()),
            "min": float(s.min()) if s.notna().any() else None,
            "mean": float(s.mean()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
        }

    def _count_true(frame: pd.DataFrame, col: str) -> int:
        """Count True values in a QC-like column."""
        if frame is None or col not in frame.columns:
            return 0
        try:
            return int(frame[col].astype(bool).sum())
        except Exception:
            return 0

    def _count_valid(frame: pd.DataFrame, col: str) -> int:
        """Count non-null numeric/string values in a column."""
        if frame is None or col not in frame.columns:
            return 0
        return int(pd.Series(frame[col]).notna().sum())

    lines: list[str] = []

    # ------------------------------------------------------------------
    # [Run information]
    # ------------------------------------------------------------------
    lines.append("[Run Information]")
    lines.append(f"Created: {created_uk}")
    lines.append(f"Site ID: {site_id}")
    lines.append(f"Data source: {(str(_cfg_get(cfg, 'data_source', 'local')) or 'local').upper()}")
    lines.append(f"Output directory: {site_out}")

    if input_path is not None:
        lines.append(f"Input file: {Path(input_path)}")
    if field_json_path is not None:
        lines.append(f"Field config JSON: {Path(field_json_path)}")

    data_url = (str(_cfg_get(cfg, "data_url", "")) or "").strip()
    if data_url:
        lines.append(f"Data URL: {data_url}")

    lines.append("")

    # ------------------------------------------------------------------
    # [Time coverage and row counts]
    # ------------------------------------------------------------------
    lines.append("[Time Coverage and Row Counts]")
    lines.append(f"time_col: {time_col}")

    if df is not None and time_col in df.columns:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        valid_ts = ts.dropna()
        lines.append(f"Total rows in dataframe: {len(df)}")
        lines.append(f"Valid timestamps: {len(valid_ts)}")
        lines.append(f"Start time: {_fmt_dt(valid_ts.min() if len(valid_ts) else None)}")
        lines.append(f"End time: {_fmt_dt(valid_ts.max() if len(valid_ts) else None)}")
    elif df is not None:
        lines.append(f"Total rows in dataframe: {len(df)}")
        lines.append("Valid timestamps: 0")
        lines.append("Start time: NA")
        lines.append("End time: NA")
    else:
        lines.append("Total rows in dataframe: NA")
        lines.append("Valid timestamps: NA")
        lines.append("Start time: NA")
        lines.append("End time: NA")

    lines.append("")

    # ------------------------------------------------------------------
    # [Column mapping]
    # ------------------------------------------------------------------
    lines.append("[Column Mapping]")
    lines.append(f"neutron_col: {neutron_col}")
    lines.append(f"pressure_col: {pressure_col}")
    lines.append(f"temp_col: {temp_col}")
    lines.append(f"rh_col: {rh_col}")
    lines.append(f"dewpoint_col: {dew_col}")
    lines.append(f"agb_col: {agb_col}")
    lines.append("")

    # ------------------------------------------------------------------
    # [Processing controls]
    # ------------------------------------------------------------------
    lines.append("[Processing Controls]")
    lines.append(f"header_rows: {HEADER_ROWS}")
    lines.append(f"time_col: {TIME_COL}")
    lines.append(
        f"timestamp_format: {DEFAULT_TIMESTAMP_FORMAT} "
        "(automatic fallback for non-matching values)"
    )
    lines.append(f"net_min_seconds: {_fmt_value(_cfg_get(cfg, 'net_min_seconds', None), 0)}")
    lines.append(f"intensity_method: {_cfg_get(cfg, 'intensity_method', 'NA')}")
    lines.append(f"theta_method: {theta_method}")
    lines.append(f"era5_area_buffer_deg: {ERA5_AREA_BUFFER_DEG}")
    lines.append(f"apply_sm_smoothing: {APPLY_SM_SMOOTHING}")
    lines.append(f"smwindow: {_fmt_value(_cfg_get(cfg, 'smwindow', None), 0)}")
    lines.append(f"qc_theta_min: {_fmt_value(_cfg_get(cfg, 'qc_theta_min', 0.0))}")
    lines.append(f"sm_max: {_fmt_value(_cfg_get(cfg, 'sm_max', None))}")
    lines.append("")

    # ------------------------------------------------------------------
    # [Core parameters]
    # ------------------------------------------------------------------
    lines.append("[Core Parameters]")
    lines.append(f"Rc: {_fmt_value(Rc)}")
    lines.append(f"beta_B: {_fmt_value(beta_B)}")
    lines.append(f"p0_ref: {_fmt_value(p0_ref)}")
    lines.append(f"rhov0_ref: {_fmt_value(rhov0_ref)}")
    lines.append(f"jung_ref: {_fmt_value(jung_ref)}")
    lines.append(f"N0: {int(round(float(N0)))}" if N0 is not None else "N0: NA")
    lines.append(f"a0: {_fmt_value(_cfg_get(cfg, 'a0', None))}")
    lines.append(f"a1: {_fmt_value(_cfg_get(cfg, 'a1', None))}")
    lines.append(f"a2: {_fmt_value(_cfg_get(cfg, 'a2', None))}")
    lines.append(f"bulk_density: {_fmt_value(_cfg_get(cfg, 'bulk_density', None))}")
    lines.append(f"density: {PARTICLE_DENSITY}")
    lines.append(f"lattice_water: {_fmt_value(_cfg_get(cfg, 'lattice_water', None))}")
    lines.append(f"soil_organic_carbon: {_fmt_value(_cfg_get(cfg, 'soil_organic_carbon', None))}")
    lines.append("")

    # ------------------------------------------------------------------
    # [Data availability]
    # ------------------------------------------------------------------
    lines.append("[Data Availability]")
    key_cols = [
        neutron_col, "N_corr_press", "N_corr_hum", "N_corr_int", "N_corr",
        "F_PRESSURE", "F_HUMIDITY", "F_INTENSITY", "F_AGB",
        "SM", "SM_SMOOTH", "D86_10m", "D86_75m", "D86_150m", "D86avg", "D86avg_12h"
    ]
    if df is not None:
        for col in key_cols:
            if col in df.columns:
                lines.append(f"{col}: {_count_valid(df, col)} valid row(s)")
    else:
        lines.append("Dataframe not provided; availability summary skipped.")
    lines.append("")

    # ------------------------------------------------------------------
    # [QC summary]
    # ------------------------------------------------------------------
    lines.append("[QC Summary]")
    if df is not None:
        total_rows = len(df)
        qc_ok = _count_true(df, "QC_OK")
        lines.append(f"Total rows: {total_rows}")
        lines.append(f"QC_OK rows: {qc_ok}")
        lines.append(f"QC pass rate: {_fmt_value((qc_ok / total_rows * 100.0) if total_rows > 0 else None)} %")
        lines.append(f"QC_FLAG1 count: {_count_true(df, 'QC_FLAG1')}")
        lines.append(f"QC_FLAG2 count: {_count_true(df, 'QC_FLAG2')}")
        lines.append(f"QC_FLAG3 count: {_count_true(df, 'QC_FLAG3')}")
        lines.append(f"QC_FLAG4 count: {_count_true(df, 'QC_FLAG4')}")
    else:
        lines.append("Dataframe not provided; QC summary skipped.")
    lines.append("")

    # ------------------------------------------------------------------
    # [Soil moisture summary]
    # ------------------------------------------------------------------
    lines.append("[Soil Moisture Summary]")
    if df is not None:
        for col in ["SM", "SM_SMOOTH", "SWC_CRNS"]:
            if col in df.columns:
                s = _series_summary(df, col)
                lines.append(
                    f"{col}: count={s['count']}, min={_fmt_value(s['min'])}, "
                    f"mean={_fmt_value(s['mean'])}, max={_fmt_value(s['max'])}"
                )
    else:
        lines.append("Dataframe not provided; soil-moisture summary skipped.")
    lines.append("")

    # ------------------------------------------------------------------
    # [Effective depth summary]
    # ------------------------------------------------------------------
    lines.append("[Effective Depth Summary]")
    if df is not None:
        for col in ["D86_10m", "D86_75m", "D86_150m", "D86avg", "D86avg_12h"]:
            if col in df.columns:
                s = _series_summary(df, col)
                lines.append(
                    f"{col}: count={s['count']}, min={_fmt_value(s['min'])}, "
                    f"mean={_fmt_value(s['mean'])}, max={_fmt_value(s['max'])}"
                )
    else:
        lines.append("Dataframe not provided; depth summary skipped.")
    lines.append("")

    # ------------------------------------------------------------------
    # [Generated files]
    # ------------------------------------------------------------------
    lines.append("[Generated Files]")
    data_dir = site_out / "data"
    fig_dir = site_out / "figures"

    if data_dir.exists():
        data_files = sorted(p.name for p in data_dir.iterdir() if p.is_file())
        lines.append("Data files:")
        if data_files:
            for name in data_files:
                lines.append(f"  - {name}")
        else:
            lines.append("  - none")
    else:
        lines.append("Data files: none")

    if fig_dir.exists():
        fig_files = sorted(p.name for p in fig_dir.iterdir() if p.is_file())
        lines.append("Figure files:")
        if fig_files:
            for name in fig_files:
                lines.append(f"  - {name}")
        else:
            lines.append("  - none")
    else:
        lines.append("Figure files: none")

    lines.append("")

    # ------------------------------------------------------------------
    # [Extra metadata]
    # ------------------------------------------------------------------
    if extra:
        lines.append("[Extra Metadata]")
        for k, v in extra.items():
            lines.append(f"{k}: {_fmt_value(v)}")
        lines.append("")

    # ------------------------------------------------------------------
    # [Glossary]
    # ------------------------------------------------------------------
    lines.append("[Glossary]")
    glossary = OrderedDict([
        ("Rc", "Cutoff rigidity (GV), representing geomagnetic shielding."),
        ("beta_B", "Barometric coefficient (hPa^-1) used in pressure correction."),
        ("p0_ref", "Reference pressure (hPa)."),
        ("rhov0_ref", "Reference absolute humidity (g m^-3)."),
        ("jung_ref", "Reference NMDB count used for intensity correction."),
        ("N0", "Calibration constant used to convert corrected neutron counts to soil moisture."),
        ("SM", "Hourly soil moisture estimated from corrected neutron counts."),
        ("SM_SMOOTH", "Smoothed soil moisture, usually rolling-mean output."),
        ("D86_10m", "Effective sensing depth estimated using representative radius 10 m."),
        ("D86_75m", "Effective sensing depth estimated using representative radius 75 m."),
        ("D86_150m", "Effective sensing depth estimated using representative radius 150 m."),
        ("D86avg", "Mean of D86_10m, D86_75m, and D86_150m."),
        ("D86avg_12h", "12-hour rolling mean of D86avg."),
        ("QC_OK", "Final QC pass flag after all active QC checks."),
        ("QC_FLAG1", "Raw neutron jump flag."),
        ("QC_FLAG2", "Lower corrected-neutron bound flag."),
        ("QC_FLAG3", "Upper corrected-neutron bound flag."),
        ("QC_FLAG4", "Battery-voltage flag."),
    ])
    for k, v in glossary.items():
        lines.append(f"{k}: {v}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
