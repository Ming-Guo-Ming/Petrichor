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
from datetime import datetime
from zoneinfo import ZoneInfo

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import urllib.request
from scipy.interpolate import griddata

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

# ==============================================================================
# IO HELPERS
# ==============================================================================

def read_local_input_data(input_path: str, header_rows: int = 2) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Read local CSV with flexible headers (0/1/2 rows)."""
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

def parse_timestamp_series(series: pd.Series, timestamp_format: str | None = None) -> pd.Series:
    """
    Parse timestamps using a user-defined format when provided.

    Parameters
    ----------
    series : pd.Series
        Input timestamp series.
    timestamp_format : str or None, default None
        Datetime format string, for example "%d/%m/%Y %H:%M".
        If None or empty, pandas automatic parsing is used.

    Returns
    -------
    pd.Series
        Parsed datetime series.
    """
    s = pd.Series(series, copy=True)

    # If already datetime-like, do not parse again with a string format.
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

    if timestamp_format not in (None, "", "auto", "AUTO", "Auto"):
        return pd.to_datetime(raw, format=timestamp_format, errors="coerce")

    return pd.to_datetime(raw, errors="coerce", dayfirst=True)

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
    time_col: str,
    timestamp_format: str | None = None
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
    return hourly

def replace_missing_with_nan(df: pd.DataFrame, missing_value) -> pd.DataFrame:
    """Replace blanks and placeholder values (scalar or list) with NA."""
    out = df.replace(r'^\s*$', pd.NA, regex=True)
    out = out.replace(missing_value, pd.NA)
    return out


def continuous_hourly_timestamps(
    df: pd.DataFrame,
    time_col: str,
    timestamp_format: str | None = None
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


def check_variables_in_data(df: pd.DataFrame, config: dict) -> None:
    """Validate required/additional columns."""
    required = _cfg_get(config, "required_vars", [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("[ERROR] Missing required columns:", ", ".join(missing))
        raise ValueError("Required variables missing.")
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
    return out, _cfg_get(cfg, "neutron_col", n_col), {"net_min_seconds": net_min, "net_scaling_applied": False}


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


# Coarse fallback Rc grid
try:
    from petrichor.rc_table import cutoff_rigidity as _RC_TABLE
except Exception:
    _RC_TABLE = np.ones((181, 361)) * 4.5

def rc_retrieval(latitude: float, longitude: float) -> float:
    """Interpolate cutoff rigidity from a global grid."""
    print(f"[Rc] Computing cutoff rigidity at lat={latitude}, lon={longitude}")
    xq = float(longitude)
    yq = float(latitude)
    if xq < 0:
        xq = -xq + 180.0
    Z = np.array(_RC_TABLE)
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

    beta_coeff = abs((term1 + term2 + term3 + term4) / (x0 - x))
    beta_coeff = str(round(float(beta_coeff), 5))
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
        Name of input neutron column (defaults to cfg["neutron_col"] or "N").
    out_col : str, default "N_PRESS_ONLY"
        Name of the output corrected column.

    Returns
    -------
    pandas.DataFrame
        Copy of df with the new pressure-corrected column added.
    """
    neutron_col  = in_col or _cfg_get(cfg, "neutron_col", "N")
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

    required_vars = list(_cfg_get(cfg, "required_vars", []))
    use_ah = ("AH_1" in required_vars) and ("AH_1" in out.columns)

    dew_col = _cfg_get(cfg, "dewpoint_col", None)
    T_col = _cfg_get(cfg, "temp_col", "TA_1")
    RH_col = _cfg_get(cfg, "rh_col", "RH_1")

    if use_ah:
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
        Config with at least "time_col"; may include site latitude/elevation and Rc.
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
    tcol = _cfg_get(cfg, "time_col", "TIMESTAMP")
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

    nmdb['DT'] = pd.to_datetime(nmdb['DT'], errors='coerce')
    nmdb = nmdb.dropna(subset=['DT']).sort_values('DT')
    nmdb = nmdb.set_index('DT').resample('H').mean().interpolate().reset_index()

    merged = pd.merge_asof(
        out[[tcol, in_col]].sort_values(tcol),
        nmdb.rename(columns={'DT': tcol}).sort_values(tcol),
        on=tcol, direction='nearest', tolerance=pd.Timedelta('1H')
    )

    N_hum  = pd.to_numeric(merged[in_col], errors="coerce")
    counts = pd.to_numeric(merged['N_COUNT'], errors="coerce")
    ref    = float(jung_ref)
    method = str(_cfg_get(cfg, "intensity_method", "hawdon2014")).lower()

    if method == "hawdon2014":
        Rc_site = _cfg_get(cfg, 'rc', None)
        Rc_ref  = _cfg_get(cfg, 'nmdb_rc', Rc_site)
        if Rc_site in (None, "", "null"):
            raise KeyError("[INT] Hawdon2014 requires 'rc'.")
        f_i = _intensity_factor_hawdon2014(
            ref, counts,
            float(Rc_site),
            float(Rc_ref if Rc_ref not in (None, "", "null") else Rc_site)
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
        ref_lat  = _cfg_get(cfg, "nmdb_latitude", site_lat)
        ref_elev = _cfg_get(cfg, "nmdb_elevation", site_elev)
        ref_depth = _atmospheric_depth_gcm2(float(ref_elev), float(ref_lat))
        Rc_ref   = float(_cfg_get(cfg, "nmdb_rc", Rc_site))
        tau, _K  = _location_factor_tau(site_depth, float(Rc_site), ref_depth, Rc_ref)
        f_i = _intensity_factor_mcjannet2023(ref, counts, tau)

    else:
        raise ValueError(f"[INT] Unknown intensity_method '{method}'.")

    out[out_col] = N_hum * f_i
    out["F_INTENSITY"] = pd.Series(f_i).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return out


# --- AGB (VEGETATION) CORRECTION ------------------------------------------
def _agb_factor(agb: pd.Series) -> pd.Series:
    """N_agb = N_int * (1/(1 - 0.009*AGB))."""

    """
    Above-ground biomass correction factor f_v.

    Purpose
    -------
    Accounts for neutron moderation by above-ground biomass (vegetation).
    The factor typically reduces/increases counts depending on biomass load.

    Parameters
    ----------
    agbval : float or array-like
        Above-ground biomass (e.g., kg m^-2), depending on your chosen parameterization.

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
    return 1.0 / (1.0 - 0.009 * agb)

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
    agb_col = _cfg_get(cfg, "agb_col", None)
    if agb_col and (agb_col in out.columns):
        agb = pd.to_numeric(out[agb_col], errors="coerce")
        out[out_col] = pd.to_numeric(out[in_col], errors="coerce") * _agb_factor(agb)
    else:
        out[out_col] = out[in_col]
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
    out_col_int: str = "N_corr_int",
    out_col_agb: str = "N_corr"
) -> pd.DataFrame:
    """
    Apply pressure, humidity, intensity, and AGB corrections in sequence
    using the standalone correction functions as the single source of truth.

    This wrapper only orchestrates the correction chain and organizes
    diagnostic outputs for plotting and debugging.
    """
    tcol = _cfg_get(cfg, "time_col", "TIMESTAMP")
    ncol = _cfg_get(cfg, "neutron_col", "N")

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

        # Optional diagnostic: store absolute humidity if needed for plotting
        required_vars = list(_cfg_get(cfg, "required_vars", []))
        use_ah = ("AH_1" in required_vars) and ("AH_1" in out.columns)

        dew_col = _cfg_get(cfg, "dewpoint_col", None)
        T_col = _cfg_get(cfg, "temp_col", "TA_1")
        RH_col = _cfg_get(cfg, "rh_col", "RH_1")

        try:
            if use_ah:
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
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 6) Round factors to 3 decimals for crspy-lite compatibility test
    # ------------------------------------------------------------------
    Fp = pd.to_numeric(Fp, errors="coerce").round(3)
    Fh = pd.to_numeric(Fh, errors="coerce").round(3)
    Fi = pd.to_numeric(out["F_INTENSITY"], errors="coerce").round(3)
    Fv = pd.to_numeric(Fv, errors="coerce").round(3)

    out["F_PRESSURE"] = Fp
    out["F_HUMIDITY"] = Fh
    out["F_AGB"] = Fv

    return out


# ==============================================================================
# CALIBRATION (N0) & THETA
# ==============================================================================


def desilets_sm_calc(a0, a1, a2, bd, N, N0, lw, soc):
    """Soil moisture processing equation for the Desilets equation."""
    
    sm = (((a0) / ((N / N0) - a1)) - (a2) - lw - soc) * bd
    return sm

def kohli_sm_calc(a0, a1, a2, bd, N, N0, lw, soc):
    """Kohli et al. (2021) soil moisture processing equation."""
    Nmax = N0 * ((a0 + (a1 * a2)) / (a2))
    ah0 = -a2
    ah1 = (a1 * a2) / (a0 + (a1 * a2))
    sm = ((ah0 * ((1 - (N / Nmax)) / (ah1 - (N / Nmax)))) - lw - soc) * bd
    return sm


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
    bd = float(_cfg_get(config_data, "bulk_density", _cfg_get(config_data, "bd", np.nan)))
    lw = float(_cfg_get(config_data, "lattice_water", _cfg_get(config_data, "lw", 0.0)))
    soc = float(_cfg_get(config_data, "soil_organic_carbon", _cfg_get(config_data, "soc", 0.0)))

    # convert SOC to water equivelant (see Hawdon et al., 2014)
    wsom = soc * 0.556
    
    Hveg = 0

    if math.isnan(bd):
        print("Bulk density (bd) is nan value. Please add a valid value to the config file.")

    # column mapping from Petrichor config
    time_col = _cfg_get(config_data, "time_col", "TIMESTAMP")
    pressure_col = _cfg_get(config_data, "pressure_col", "PA_1")
    temp_col = _cfg_get(config_data, "temp_col", "TA_1")
    rh_col = _cfg_get(config_data, "rh_col", "RH_1")

    required_vars = list(_cfg_get(config_data, "required_vars", []))
    has_ah_fallback = ("AH_1" in required_vars)
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

    # Parse calibration timestamps explicitly as day-first format.
    df["DATE"] = pd.to_datetime(
        df["DATE"].astype(str).str.strip(),
        format="%d.%m.%Y %H:%M",
        errors="coerce"
    )

    bad_calib_dates = df["DATE"].isna().sum()
    if bad_calib_dates:
        print(f"[CAL][WARN] {bad_calib_dates} calibration timestamps could not be parsed and will be dropped.")

    df = df.dropna(subset=["DATE"]).copy()
    df["DATE"] = df["DATE"].dt.date

    unidate = df.DATE.unique()
    print("Unique calibration dates found: " + str(unidate))
    print("Done")

    df["DIST"] = df["DIST"].astype(float)
    df["DEPTH_AVG"] = df["DEPTH_AVG"].astype(float)

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

    if lvl1["E_RH"].notna().sum() > 0:
        hum_mode = "rh"
    else:
        hum_mode = "vp"
        print("No external relative humidity detected. Falling back to vapor pressure if available.")

    lvl1[lvl1 == noval_scalar] = np.nan

    dflvl1Days = dict()
    for i in range(numdays):
        dflvl1Days[i] = lvl1.loc[lvl1["DATE"] == unidate[i]]

    avgP = dict()
    for i in range(len(dflvl1Days)):
        tmp = pd.DataFrame.from_dict(dflvl1Days[i])
        tmp = tmp[(tmp["DT"] > str(unidate[i]) + " " + str(calib_start_time)) &
                  (tmp["DT"] <= str(unidate[i]) + " " + str(calib_end_time))]
        check = float(np.nanmean(tmp["PRESS"], axis=0))
        if np.isnan(check):
            tmp = pd.DataFrame.from_dict(dflvl1Days[i])
            avgP[i] = float(np.nanmean(tmp["PRESS"], axis=0))
        else:
            avgP[i] = check

    avgT = dict()
    for i in range(len(dflvl1Days)):
        tmp = pd.DataFrame.from_dict(dflvl1Days[i])
        tmp = tmp[(tmp["DT"] > str(unidate[i]) + " " + str(calib_start_time)) &
                  (tmp["DT"] <= str(unidate[i]) + " " + str(calib_end_time))]
        check = float(np.nanmean(tmp["TEMP"], axis=0))
        if np.isnan(check):
            tmp = pd.DataFrame.from_dict(dflvl1Days[i])
            avgT[i] = float(np.nanmean(tmp["TEMP"], axis=0))
        else:
            avgT[i] = check

    # Optional fallback absolute humidity from flux data.
    # This follows the crspy-lite calibration logic:
    # if RH/VP-based humidity cannot be computed, use E_AH_FLUX when available.
    try:
        avgAH = dict()
        for i in range(len(dflvl1Days)):
            tmp = pd.DataFrame.from_dict(dflvl1Days[i])
            tmp = tmp[(tmp["DT"] > str(unidate[i]) + " " + str(calib_start_time)) &
                      (tmp["DT"] <= str(unidate[i]) + " " + str(calib_end_time))]
            check = float(np.nanmean(tmp["E_AH_FLUX"], axis=0))

            if np.isnan(check):
                tmp = pd.DataFrame.from_dict(dflvl1Days[i])
                avgAH[i] = float(np.nanmean(tmp["E_AH_FLUX"], axis=0))
            else:
                avgAH[i] = check
    except Exception:
        pass

    if hum_mode == "rh":
        avgRH = dict()
        for i in range(len(dflvl1Days)):
            tmp = pd.DataFrame.from_dict(dflvl1Days[i])
            tmp = tmp[(tmp["DT"] > str(unidate[i]) + " " + str(calib_start_time)) &
                      (tmp["DT"] <= str(unidate[i]) + " " + str(calib_end_time))]
            check = float(np.nanmean(tmp["E_RH"], axis=0))
            if np.isnan(check):
                tmp = pd.DataFrame.from_dict(dflvl1Days[i])
                avgRH[i] = float(np.nanmean(tmp["E_RH"], axis=0))
            else:
                avgRH[i] = check

    else:
        avgVP = dict()
        for i in range(len(dflvl1Days)):
            tmp = pd.DataFrame.from_dict(dflvl1Days[i])
            tmp = tmp[(tmp["DT"] > str(unidate[i]) + " " + str(calib_start_time)) &
                      (tmp["DT"] <= str(unidate[i]) + " " + str(calib_end_time))]
            check = float(np.nanmean(tmp["VP"], axis=0)) if "VP" in tmp.columns else np.nan
            if np.isnan(check):
                tmp = pd.DataFrame.from_dict(dflvl1Days[i])
                avgVP[i] = float(np.nanmean(tmp["VP"])) if "VP" in tmp.columns else np.nan
            else:
                avgVP[i] = check

    """
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~ THE BIG ITERATIVE LOOP - CALCULATE WEIGHTED THETA ~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    AvgTheta = dict()

    for i in range(len(dflvl1Days)):
        print("Calibrating to day " + str(i + 1) + "...")
        df1 = pd.DataFrame.from_dict(dfCalib[i])
        df1["SWV"] = pd.to_numeric(df1["SWV"], errors="coerce")

        if df1["SWV"].mean() > 1:
            print("crspy-lite detected SWV is not in decimal format, dividing by 100.")
            df1["SWV"] = df1["SWV"] / 100

        CalibTheta = df1["SWV"].mean()
        Accuracy = 1

        if "PROFILE" not in df1.columns:
            profiles = df1["LOC"].unique()
            numprof = len(profiles)
            pfnum = []
            for row in df1["LOC"]:
                for j in range(numprof):
                    if row == profiles[j]:
                        pfnum.append(j + 1)
            df1["PROFILE"] = pfnum

        max_iter = 100

        for iter_idx in range(1, max_iter + 1):
            thetainitial = float(CalibTheta)

            if not np.isfinite(thetainitial):
                raise ValueError(
                    f"Calibration theta became non-finite before iteration for day {unidate[i]}."
                )

            df1["rscale"] = df1.apply(
                lambda row: rscaled(row["DIST"], avgP[i], Hveg, thetainitial), axis=1
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

            day1temp = avgT[i]

            if hum_mode == "rh":
                day1rh = avgRH[i]
                day1es = es(day1temp)
                day1ea = ea(day1es, day1rh)
                day1hum = pv(day1ea, day1temp) * 1000.0

            elif hum_mode == "ah":
                day1hum = avgAH[i]

            else:
                day1vp = avgVP[i]
                day1hum = pv(day1vp, day1temp) * 1000.0

            if not np.isfinite(day1hum):
                raise ValueError(
                    f"Calibration humidity is non-finite for day {unidate[i]}. "
                    f"Check RH / AH / VP inputs around the calibration window."
                )

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
                raise ValueError(
                    f"Calibration weighting failed for day {unidate[i]}: Wr sum is invalid."
                )

            CalibTheta_new = rad_sum / wr_sum

            if not np.isfinite(CalibTheta_new):
                raise ValueError(
                    f"Calibration theta became non-finite for day {unidate[i]} during iteration."
                )

            Accuracy = abs((CalibTheta_new - thetainitial) / max(abs(thetainitial), 1e-12))
            CalibTheta = float(CalibTheta_new)
            AvgTheta[i] = CalibTheta

            if iter_idx % 10 == 0 or Accuracy <= defineaccuracy:
                print(
                    f"[CAL][ITER] day={unidate[i]} iter={iter_idx} "
                    f"theta={CalibTheta:.6f} accuracy={Accuracy:.6e}"
                )

            if Accuracy <= defineaccuracy:
                break
        else:
            raise RuntimeError(
                f"Calibration did not converge for day {unidate[i]} after {max_iter} iterations. "
                f"Last theta={CalibTheta}, last accuracy={Accuracy}"
            )

        print("Done")

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
    n_avg = n_avg_raw

    if np.isnan(n_avg) or np.isinf(n_avg):
        print("Error: MOD_CORR contains only NaN values or results in infinity.")
        n_avg = 0
    else:
        n_avg = int(n_avg)
    
    #if n_avg >= 4000:
        #print(f"Avg N for the site is {n_avg} due to presumed errors in raw data. It has been capped at 3,000")
        #n_avg = 3000

        """
        TODO need to reconsider this for broader issue. What value for n_avg when averages are way off.
        For now hard code 3000, but this will vary depending on sensor/location. Perhaps some kind of mode?
        """
        
    #else:
    
        print(f"Avg N for the site is {n_avg}")

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
        tmp_day = tmp_day[(tmp_day["DT"] > str(unidate[i]) + " " + str(calib_start_time)) &
                          (tmp_day["DT"] <= str(unidate[i]) + " " + str(calib_end_time))]
        check = float(np.nanmean(tmp_day["MOD_CORR"]))

        if np.isnan(check):
            tmp_day = pd.DataFrame.from_dict(NeutCount[i])
            avgN[i] = float(np.nanmean(tmp_day["MOD_CORR"]))
        else:
            avgN[i] = check

    RelerrDict = dict()
    with np.errstate(divide="ignore"):
        for i in range(numdays):
            N0 = pd.Series(range(n_avg, int(n_avg * 2.5)))
            vwc = AvgTheta[i]
            Nave = avgN[i]
            sm = pd.DataFrame(columns=["sm"])
            reler = pd.DataFrame(columns=["RelErr"])

            if sm_calc_method == "desilets":
                for j in range(len(N0)):
                    sm.loc[j] = desilets_sm_calc(
                        float(_cfg_get(config_data, "a0", 0.0808)),
                        float(_cfg_get(config_data, "a1", 0.372)),
                        float(_cfg_get(config_data, "a2", 0.115)),
                        bd, Nave, N0.loc[j], lw, soc
                    )
                    reler.loc[j] = abs(sm.iat[j, 0] - vwc)

            elif sm_calc_method == "kohli":
                for j in range(len(N0)):
                    sm.loc[j] = kohli_sm_calc(
                        float(_cfg_get(config_data, "a0", 0.0808)),
                        float(_cfg_get(config_data, "a1", 0.372)),
                        float(_cfg_get(config_data, "a2", 0.115)),
                        bd, Nave, N0.loc[j], lw, soc
                    )
                    reler.loc[j] = abs(sm.iat[j, 0] - vwc)
            else:
                raise KeyError("No other method to calculate soil moisture at this time. Please use either 'desilets' or 'kohli' method.")

            RelerrDict[i] = reler["RelErr"]
            reler["N0"] = range(n_avg, int(n_avg * 2.5))
            # reler.to_csv(data_out / f"{country}_SITE_{sitenum}_error_{unidate[i]}.csv", index=False)

    totalerror = RelerrDict[0]
    for i in range(len(unidate) - 1):
        tmp_err = RelerrDict[i + 1]
        totalerror = totalerror + tmp_err

    minimum_error = min(totalerror)

    totalerror = totalerror.to_frame()
    totalerror["N0"] = range(n_avg, int(n_avg * 2.5))

    minindex = totalerror.loc[totalerror.RelErr == minimum_error]
    N0 = minindex["N0"].item()
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

def sm_max_calc(bd: float, density: float = 2.65) -> float:
    """
    Calculate maximum physically plausible volumetric soil moisture from porosity.

    Parameters
    ----------
    bd : float
        Bulk density (g cm^-3).
    density : float, default 2.65
        Particle density (g cm^-3).

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

    a0 = float(_cfg_get(cfg, "a0", 0.0808))
    a1 = float(_cfg_get(cfg, "a1", 0.372))
    a2 = float(_cfg_get(cfg, "a2", 0.115))
    bd = float(_cfg_get(cfg, "bulk_density", 1.43))
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
    bd = float(_cfg_get(cfg, "bulk_density", 1.43))
    lw = float(_cfg_get(cfg, "lattice_water", 0.0))
    soc = float(_cfg_get(cfg, "soil_organic_carbon", 0.0))

    # convert SOC to water equivelant (see Hawdon et al., 2014)
    wsom = soc * 0.556

    N0 = float(_cfg_get(cfg, "N0"))
    method = str(_cfg_get(cfg, "theta_method", "desilets")).lower()
    density = float(_cfg_get(cfg, "density", 2.65))

    sm_max_cfg = _cfg_get(cfg, "sm_max", None)
    if sm_max_cfg in (None, "", "null"):
        sm_max = sm_max_calc(bd=bd, density=density)
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
    Apply optional rolling smoothing to soil moisture.

    Controlled by JSON config:
    - apply_sm_smoothing: bool
    - smwindow: int

    Rules
    -----
    - If apply_sm_smoothing is False, do nothing and return df unchanged.
    - If sm_col does not exist, return df unchanged.
    - Output is written to out_col.
    """
    out = df.copy()

    apply_flag = bool(_cfg_get(cfg, "apply_sm_smoothing", True))
    smwindow = int(_cfg_get(cfg, "smwindow", 12))

    if not apply_flag:
        return out

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

    bd = float(_cfg_get(cfg, "bulk_density", 1.43))
    pressure_col = _cfg_get(cfg, "pressure_col", "PA_1")

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
    density = float(_cfg_get(cfg, "density", 2.65))

    sm_min = float(_cfg_get(cfg, "qc_theta_min", 0.0))

    sm_max_cfg = _cfg_get(cfg, "sm_max", None)
    if sm_max_cfg in (None, "", "null"):
        sm_max = sm_max_calc(bd=bd, density=density)
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

    time_col = _cfg_get(cfg, "time_col", "TIMESTAMP")
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
    # ------------------------------------------------------------------
    missing_fail = pd.Series(False, index=out.index)

    if raw_n_col in out.columns:
        missing_fail = missing_fail | pd.to_numeric(out[raw_n_col], errors="coerce").isna()

    if corr_n_col in out.columns:
        missing_fail = missing_fail | pd.to_numeric(out[corr_n_col], errors="coerce").isna()

    if batt_col in out.columns:
        missing_fail = missing_fail | pd.to_numeric(out[batt_col], errors="coerce").isna()

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

    if N0_val in (None, "", "null"):
        print("[SWC] N0 missing. Soil moisture cannot be computed.")
        out[swc_crns_col] = np.nan
        out["SM"] = np.nan
        out["SM_SMOOTH"] = np.nan
        out = compute_effective_depth(out, cfg, sm_col="SM", sm_smooth_col="SM_SMOOTH")
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
    time_col = _cfg_get(cfg, "time_col", "TIMESTAMP")
    if "DATETIME" not in out.columns:
        out["DATETIME"] = pd.to_datetime(out.get(time_col, pd.NaT), errors="coerce")
    raw_col = _cfg_get(cfg, "neutron_col", "N")
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

    time_col = _cfg_get(cfg, "time_col", "TIMESTAMP")
    neutron_col = neutron_col or _cfg_get(cfg, "neutron_col", "N")
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
    lines.append(f"header_rows: {_fmt_value(_cfg_get(cfg, 'header_rows', None), 0)}")
    lines.append(f"net_min_seconds: {_fmt_value(_cfg_get(cfg, 'net_min_seconds', None), 0)}")
    lines.append(f"intensity_method: {_cfg_get(cfg, 'intensity_method', 'NA')}")
    lines.append(f"theta_method: {theta_method}")
    lines.append(f"apply_sm_smoothing: {_cfg_get(cfg, 'apply_sm_smoothing', True)}")
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
    lines.append(f"density: {_fmt_value(_cfg_get(cfg, 'density', None))}")
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