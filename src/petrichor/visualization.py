# -*- coding: utf-8 -*-
"""
visualization.py - figure generation utilities for Petrichor outputs

This module contains plotting helpers used to create quicklook figures and
supporting diagnostic plots from processed Petrichor data.

Current figure types include:
- multi-panel quicklook overview
- correction-factor time series
- raw versus corrected neutron-count comparisons
- single-factor diagnostic plots
- pressure and absolute-humidity diagnostic series

Design goals:
- write figures only under the site-specific figure directory
- remain tolerant to missing columns
- keep plotting code lightweight and dependency-minimal
- generate publication-ready or report-ready diagnostics from one processed dataframe

The main public entry point is run_all_visualizations(),
which prepares plotting columns and dispatches the individual figure functions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .processing import TIME_COL


# -------- Internal utilities ---------------------------------------------------

# Use Times New Roman for all text (ticks, labels, legends, numbers)
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 13,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

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

def _ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Ensure a datetime Series exists and is named 'DATETIME'.
    If 'DATETIME' exists, use it. Otherwise, convert from `time_col`.
    """
    if "DATETIME" in df.columns:
        dt = pd.to_datetime(df["DATETIME"], errors="coerce")
    else:
        dt = pd.to_datetime(df[time_col], errors="coerce")
    return dt


def _safe_get(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Return column if present; else a NaN series of correct length.
    This allows plotting code to remain simple without KeyErrors.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index, name=col)


def _daily_agg(
    dt: pd.Series,
    series: pd.Series,
    how: str = "mean",
    min_valid_hours: int = 12
) -> Tuple[pd.Series, pd.Series]:
    """
    Aggregate a time series to daily resolution.

    Rules
    -----
    - For daily mean, require at least `min_valid_hours` valid hourly values.
      Otherwise the daily value is set to NaN.
    - For daily sum, keep the current behavior with min_count=1.
    """
    tmp = pd.DataFrame({
        "DT": pd.to_datetime(dt, errors="coerce"),
        "val": pd.to_numeric(series, errors="coerce"),
    })

    tmp = tmp.dropna(subset=["DT"]).set_index("DT").sort_index()

    if tmp.empty:
        return pd.DatetimeIndex([]), pd.Series(dtype="float64")

    if how == "sum":
        out = tmp["val"].resample("D").sum(min_count=1)
    else:
        out = tmp["val"].resample("D").mean()
        valid_counts = tmp["val"].resample("D").count()
        out[valid_counts < int(min_valid_hours)] = np.nan

    return out.index, out

def _resample_agg(dt: pd.Series, series: pd.Series, freq: str = "D", how: str = "mean") -> Tuple[pd.Series, pd.Series]:
    """
    Aggregate a time series to an arbitrary temporal frequency.

    Parameters
    ----------
    dt : pd.Series
        Datetime series.
    series : pd.Series
        Numeric series to aggregate.
    freq : str, default "D"
        Pandas resampling frequency, e.g.:
        - "h"  : hourly
        - "D"  : daily
        - "MS" : month start
        - "YS" : year start
    how : str, default "mean"
        Aggregation method: "mean" or "sum".

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Resampled datetime index and aggregated values.
    """
    tmp = pd.DataFrame({
        "DT": pd.to_datetime(dt, errors="coerce"),
        "val": pd.to_numeric(series, errors="coerce"),
    })

    tmp = tmp.dropna(subset=["DT"]).set_index("DT").sort_index()

    if how == "sum":
        out = tmp["val"].resample(freq).sum(min_count=1)
    else:
        out = tmp["val"].resample(freq).mean()

    return out.index, out
# -------- Public API -----------------------------------------------------------

def plot_quicklook(df: pd.DataFrame, time_col: str, fig_dir: Path | str) -> None:
    """
    Create a compact 4-panel quicklook figure and save it to `fig_dir/quicklook.png`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe at hourly (or sub-hourly) resolution.
    time_col : str
        Name of the timestamp column in ``df``. Normal Petrichor runs use the
        fixed input column ``TIMESTAMP``; derived plotting frames may pass
        ``DATETIME`` explicitly.
    fig_dir : Path or str
        Destination directory for the figure. We only write under this directory,
        and we do NOT create any 'outputs/figures' folder here.
    """
    # --- Prepare destination directory ---
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)  # e.g., output/<site>/figures
    out_png = fig_dir / "quicklook.png"

    # --- Prepare time and columns (tolerant to missing fields) ---
    dt = _ensure_datetime(df, time_col)

    mod_raw   = _safe_get(df, "MOD")         # raw counts
    mod_corr  = _safe_get(df, "MOD_CORR")    # corrected counts
    f_press   = _safe_get(df, "F_PRESSURE")  # pressure factor
    f_hum     = _safe_get(df, "F_HUMIDITY")  # humidity factor
    f_int     = _safe_get(df, "F_INTENSITY") # intensity factor

    # soil moisture and optional uncertainty band
    sm_raw    = _safe_get(df, "SM").rename("SM")
    sm_smooth = _safe_get(df, "SM_SMOOTH").rename("SM_SMOOTH")
    sm_p      = _safe_get(df, "SM_PLUS_ERR")
    sm_m      = _safe_get(df, "SM_MINUS_ERR")

    depth_raw = _safe_get(df, "D86avg")
    depth_smooth = _safe_get(df, "D86avg_12h")

    # --- Aggregate to daily for smoother plots ---
    # Counts (mean), factors (mean), depth (mean), soil moisture (mean), rainfall (sum) if present
    day_mod_x,  day_mod_raw  = _daily_agg(dt, mod_raw, how="mean")
    _,            day_mod_cor  = _daily_agg(dt, mod_corr, how="mean")

    day_fp_x,   day_fp       = _daily_agg(dt, f_press, how="mean")
    _,            day_fh       = _daily_agg(dt, f_hum,   how="mean")
    _,            day_fi       = _daily_agg(dt, f_int,   how="mean")

    day_sm_x, day_sm_raw = _daily_agg(dt, sm_raw, how="mean")
    _, day_sm_smooth = _daily_agg(dt, sm_smooth, how="mean")
    _, day_sm_p = _daily_agg(dt, sm_p, how="mean")
    _, day_sm_m = _daily_agg(dt, sm_m, how="mean")

    day_d_x, day_depth_raw = _daily_agg(dt, depth_raw, how="mean")
    _, day_depth_smooth = _daily_agg(dt, depth_smooth, how="mean")

    # --- Build figure ---
    plt.rcParams["font.size"] = 13
    fig, axs = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
#    fig.suptitle("Petrichor Quicklook", fontsize=16)

    # (1) Raw vs corrected counts
    ax = axs[0]
    has_any_counts = np.isfinite(day_mod_raw).any() or np.isfinite(day_mod_cor).any()
    if has_any_counts:
        if np.isfinite(day_mod_raw).any():
            ax.plot(day_mod_x, day_mod_raw, lw=1.0, label="Raw counts (daily mean)")
        if np.isfinite(day_mod_cor).any():
            ax.plot(day_mod_x, day_mod_cor, lw=1.0, label="Corrected counts (daily mean)")
        ax.set_ylabel("Neutron Count (cph)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No MOD/MOD_CORR columns to plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Neutron Count (cph)")

    # (2) Correction factors
    ax = axs[1]
    has_any_factor = np.isfinite(day_fp).any() or np.isfinite(day_fh).any() or np.isfinite(day_fi).any()
    if has_any_factor:
        if np.isfinite(day_fp).any():
            ax.plot(day_fp_x, day_fp, lw=1.0, label=r"$\mathit{f}_p$")
        if np.isfinite(day_fh).any():
            ax.plot(day_fp_x, day_fh, lw=1.0, label=r"$\mathit{f}_h$")
        if np.isfinite(day_fi).any():
            ax.plot(day_fp_x, day_fi, lw=1.0, label=r"$\mathit{f}_i$")
        ax.set_ylabel("Correction factor (-)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No correction factors in dataframe", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Correction factor (-)")

    # (3) Soil moisture with uncertainty band
    ax = axs[2]

    has_sm_smooth = np.isfinite(day_sm_smooth).any()
    has_sm_raw = np.isfinite(day_sm_raw).any()

    if has_sm_smooth or has_sm_raw:
        # choose one center curve only
        if has_sm_smooth:
            sm_center = day_sm_smooth
        else:
            sm_center = day_sm_raw

        ax.plot(day_sm_x, sm_center, lw=1.2, label="Soil moisture (daily mean)")

        # SM_PLUS_ERR / SM_MINUS_ERR are error magnitudes, not absolute bounds
        if np.isfinite(day_sm_p).any() and np.isfinite(day_sm_m).any():
            upper = sm_center + day_sm_p
            lower = sm_center - day_sm_m
            ax.fill_between(day_sm_x, lower, upper, alpha=0.25, label="Uncertainty band")

        ax.set_ylabel("SM (cm$^3$/cm$^3$)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No SM column in dataframe", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("SM (cm$^3$/cm$^3$)")

    # (4) Effective depth
    ax = axs[3]

    has_depth_smooth = np.isfinite(day_depth_smooth).any()
    has_depth_raw = np.isfinite(day_depth_raw).any()

    if has_depth_smooth or has_depth_raw:
        if has_depth_smooth:
            ax.plot(day_d_x, day_depth_smooth, lw=1.0, label="Effective depth (daily mean)")
        else:
            ax.plot(day_d_x, day_depth_raw, lw=1.0, label="Effective depth (daily mean)")

        ax.set_ylabel("Depth (cm)")
        ax.invert_yaxis()
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No depth data in dataframe", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Depth (cm)")
        
    # Shared x formatting
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for ax in axs:
        ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    
# -------- plot_sm_colourts  -----------------------
def plot_sm_colourts(df: pd.DataFrame, time_col: str, fig_dir: Path | str) -> None:
    """
    Create a crspy-style coloured soil-moisture time series plot.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_png = fig_dir / "soil_moisture_colourts.png"

    dt = _ensure_datetime(df, time_col)

    # Prefer smoothed soil moisture if available
    sm = _safe_get(df, "SM_SMOOTH")
    if not np.isfinite(sm).any():
        sm = _safe_get(df, "SM")

    tmp = pd.DataFrame({
        "DT": pd.to_datetime(dt, errors="coerce"),
        "SM": pd.to_numeric(sm, errors="coerce"),
    }).dropna(subset=["DT"]).sort_values("DT")

    if tmp.empty or (not np.isfinite(tmp["SM"]).any()):
        print("[PLOT] No SM or SM_SMOOTH column found, skip soil_moisture_colourts.png")
        return

    sm_valid = tmp["SM"][np.isfinite(tmp["SM"])]
    ymax = float(sm_valid.max())

    if not np.isfinite(ymax) or ymax <= 0:
        print("[PLOT] Soil moisture is empty or non-positive, skip soil_moisture_colourts.png")
        return

    lower_bound = 0.0
    ymax_plot = ymax * 1.05

    # Build a diverging palette similar to legacy crspy style
    nsteps = 50
    cmap = plt.get_cmap("coolwarm_r")
    base_colors = [cmap(i / (nsteps - 1)) for i in range(nsteps)]

    # Remove the pale centre colours
    keep_low = int(np.ceil(nsteps * 0.30))
    keep_high = int(np.ceil(nsteps * 0.55))
    fill_colors = base_colors[:keep_low] + base_colors[keep_high:]

    thresholds = np.linspace(lower_bound, ymax, len(fill_colors), endpoint=False)

    fig, ax = plt.subplots(figsize=(15, 3.75))

    # Black outline
    ax.plot(tmp["DT"], tmp["SM"], lw=0.25, color="black")

    # Layered coloured fill
    for thr, color in zip(thresholds, fill_colors):
        ax.fill_between(
            tmp["DT"],
            lower_bound,
            tmp["SM"],
            where=tmp["SM"] > thr,
            facecolor=color,
            alpha=0.20,
            interpolate=True,
        )

    ax.set_ylim(lower_bound, ymax_plot)
    ax.set_ylabel("Soil Moisture - Volumetric (cm$^3$/cm$^3$)")
    ax.set_xlabel("Date")
    ax.set_title("Soil Moisture over time")
    ax.grid(True, alpha=0.15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_hourly_unsmoothed_sm(
    df: pd.DataFrame,
    time_col: str,
    fig_dir: Path | str,
) -> None:
    """Plot the QC-passed hourly soil moisture before rolling smoothing."""
    if "SM" not in df.columns:
        print("[PLOT] No SM column found, skip soil_moisture_hourly_unsmoothed.png")
        return

    tmp = pd.DataFrame({
        "DT": _ensure_datetime(df, time_col),
        "SM": pd.to_numeric(df["SM"], errors="coerce"),
    }).dropna(subset=["DT", "SM"]).sort_values("DT")

    if tmp.empty:
        print("[PLOT] No valid unsmoothed SM values, skip soil_moisture_hourly_unsmoothed.png")
        return

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(
        tmp["DT"],
        tmp["SM"],
        color="tab:blue",
        linewidth=0.7,
        label="Hourly soil moisture (unsmoothed)",
    )
    ax.set_title("Hourly Soil Moisture Before Smoothing")
    ax.set_xlabel("Date")
    ax.set_ylabel("SM (cm$^3$/cm$^3$)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "soil_moisture_hourly_unsmoothed.png", dpi=300)
    plt.close(fig)


# -------- plot_sm_multiscale  -----------------------
def plot_sm_multiscale(df: pd.DataFrame, time_col: str, fig_dir: Path | str) -> None:
    """
    Create a 4-panel soil-moisture figure with hourly, daily,
    monthly, and yearly mean series.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_png = fig_dir / "soil_moisture_multiscale.png"

    dt = _ensure_datetime(df, time_col)

    # Prefer smoothed soil moisture if available; otherwise fall back to raw SM
    sm = _safe_get(df, "SM_SMOOTH")
    if not np.isfinite(sm).any():
        sm = _safe_get(df, "SM")

    if not np.isfinite(sm).any():
        print("[PLOT] No SM or SM_SMOOTH column found, skip soil_moisture_multiscale.png")
        return

    hour_x, hour_y = _resample_agg(dt, sm, freq="h", how="mean")
    day_x, day_y = _resample_agg(dt, sm, freq="D", how="mean")
    month_x, month_y = _resample_agg(dt, sm, freq="MS", how="mean")
    year_x, year_y = _resample_agg(dt, sm, freq="YS", how="mean")

    plt.rcParams["font.size"] = 13
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=False)

    panels = [
        (axs[0], hour_x, hour_y, "Hourly mean soil moisture", "%Y-%m"),
        (axs[1], day_x, day_y, "Daily mean soil moisture", "%Y-%m"),
        (axs[2], month_x, month_y, "Monthly mean soil moisture", "%Y-%m"),
        (axs[3], year_x, year_y, "Yearly mean soil moisture", "%Y"),
    ]

    for ax, x, y, title, date_fmt in panels:
        if len(y) > 0 and np.isfinite(y).any():
            ax.plot(x, y, lw=1.0)
        else:
            ax.text(
                0.5, 0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

        ax.set_title(title)
        ax.set_ylabel("SM (cm$^3$/cm$^3$)")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))

    axs[-1].set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# -------- Extra figure helpers (single-purpose figures) -----------------------

def save_individual_correction_factor_plots(df: pd.DataFrame, time_col: str, fig_dir: Path | str) -> None:
    """
    Save four separate figures, each containing ONLY one correction factor
    (pressure / humidity / intensity / vegetation) as a time series (daily mean).

    Files saved (if column exists):
      - factor_pressure.png
      - factor_humidity.png
      - factor_intensity.png
      - factor_vegetation.png
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dt = _ensure_datetime(df, time_col)

    # Fetch columns safely (NaN if missing)
    f_map = [
        ("F_PRESSURE",  "Pressure factor",  "factor_pressure.png"),
        ("F_HUMIDITY",  "Humidity factor",  "factor_humidity.png"),
        ("F_INTENSITY", "Intensity factor", "factor_intensity.png"),
        ("F_AGB",       "Vegetation factor","factor_vegetation.png"),
    ]

    for col, label, fname in f_map:
        series = _safe_get(df, col)
        day_x, day_y = _daily_agg(dt, series, how="mean")

        # If the series is completely NaN, skip saving
        if not np.isfinite(day_y).any():
            continue

        plt.rcParams["font.size"] = 13
        fig, ax = plt.subplots(1, 1, figsize=(12, 3.2))
        ax.plot(day_x, day_y, lw=1.0)
        ax.set_ylabel("Factor (-)")
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=300)
        plt.close(fig)


def save_single_factor_corrected_counts_plot(df: pd.DataFrame, time_col: str, fig_dir: Path | str) -> None:
    """
    Plot raw N vs N corrected by ONE factor at a time (no stacking), all in one figure.
    This lets you see how each factor alone would change N.

    Inputs expected:
      - 'MOD'   : raw neutron count (cph) at the native timestep
      - 'F_*'   : correction factors at the native timestep (if present)

    What we do:
      - Build per-timestep N_press_only = MOD * F_PRESSURE (if available); same for others
      - Aggregate each series to daily mean for cleaner comparison
      - Save one line plot with all series overlaid

    File saved:
      - N_single_factor_effects.png
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dt = _ensure_datetime(df, time_col)
    N_raw = _safe_get(df, "MOD")  # raw counts at native timestep

    # If there's no raw N, nothing to do
    if not np.isfinite(N_raw).any():
        return

    # Build per-timestep "apply one factor only" series
    Fp = _safe_get(df, "F_PRESSURE")
    Fh = _safe_get(df, "F_HUMIDITY")
    Fi = _safe_get(df, "F_INTENSITY")
    Fv = _safe_get(df, "F_AGB")

    # Multiply at native resolution, then aggregate to daily mean
    series_list = [("Raw (N)", N_raw, None)]

    if np.isfinite(Fp).any():
        series_list.append(("N with pressure only", N_raw * Fp, "press"))
    if np.isfinite(Fh).any():
        series_list.append(("N with humidity only", N_raw * Fh, "hum"))
    if np.isfinite(Fi).any():
        series_list.append(("N with intensity only", N_raw / Fi, "int"))
    if np.isfinite(Fv).any():
        series_list.append(("N with vegetation only", N_raw * Fv, "veg"))

    # If we only have raw N, skip plotting
    if len(series_list) <= 1:
        return

    # Aggregate to daily
    day_x = None
    daily_curves = []
    for label, s, _tag in series_list:
        x, y = _daily_agg(dt, s, how="mean")
        daily_curves.append((label, x, y))
        if day_x is None:
            day_x = x

    # Plot
    plt.rcParams["font.size"] = 13
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    for label, x, y in daily_curves:
        if np.isfinite(y).any():
            ax.plot(x, y, lw=1.0, label=label)

    ax.set_ylabel("Neutron Count (daily mean cph)")
    ax.set_title("Raw vs single-factor corrected N")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(fig_dir / "N_single_factor_effects.png", dpi=300)
    plt.close(fig)


def save_raw_vs_single_factor_plots(df: pd.DataFrame, time_col: str, fig_dir: Path | str) -> None:
    """
    Save four separate figures. Each figure shows ONLY:
      - Raw N (daily mean from MOD)
      - N corrected by ONE factor (daily mean from MOD * F_*)
    Factors handled: pressure, humidity, intensity, vegetation.

    Output files:
      - raw_vs_pressure.png
      - raw_vs_humidity.png
      - raw_vs_intensity.png
      - raw_vs_vegetation.png
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dt   = _ensure_datetime(df, time_col)
    Nraw = _safe_get(df, "MOD")

    # If no raw N, nothing to do
    if not np.isfinite(Nraw).any():
        return

    factor_specs = [
        ("F_PRESSURE",  "Pressure only",  "raw_vs_pressure.png"),
        ("F_HUMIDITY",  "Humidity only",  "raw_vs_humidity.png"),
        ("F_INTENSITY", "Intensity only", "raw_vs_intensity.png"),
        ("F_AGB",       "Vegetation only","raw_vs_vegetation.png"),
    ]

    # Pre-aggregate raw to daily mean once
    day_x, day_raw = _daily_agg(dt, Nraw, how="mean")

    for fcol, title_suffix, fname in factor_specs:
        F = _safe_get(df, fcol)
        if not np.isfinite(F).any():
            # Skip if this factor does not exist
            continue

        # Apply single factor at native resolution, then daily mean
        if fcol == "F_INTENSITY":
            native_corr = Nraw / F
        else:
            native_corr = Nraw * F

        if fcol == "F_INTENSITY":
            _, day_corr = _daily_agg(dt, Nraw / F, how="mean")
        else:
            _, day_corr = _daily_agg(dt, Nraw * F, how="mean")

        # If both are NaN, skip
        if (not np.isfinite(day_raw).any()) and (not np.isfinite(day_corr).any()):
            continue

        plt.rcParams["font.size"] = 13
        fig, ax = plt.subplots(1, 1, figsize=(12, 3.2))

        # Map factor column name -> short math subscript (p/h/i/v)
        _FACTOR_TAG = {
            "F_PRESSURE": "p",
            "N_PRESS_ONLY": "p",
            "F_HUMIDITY": "h",
            "N_HUM_ONLY": "h",
            "F_INTENSITY": "i",
            "N_INT_ONLY": "i",
            "F_AGB": "v",
            "N_AGB_ONLY": "v",
        }

        # ... inside your plotting block:
        if np.isfinite(day_raw).any():
            ax.plot(day_x, day_raw, lw=1.0, label="raw N")

        if np.isfinite(day_corr).any():
            tag = _FACTOR_TAG.get(fcol, "")
            if tag:
                # Use italic f with subscript via mathtext: N × f_tag
                pretty = rf"$N \times \mathit{{f}}_{tag}$"
            else:
                # Fallback if the column name is unknown
                pretty = f"N with {fcol}"
            ax.plot(day_x, day_corr, lw=1.0, label=pretty)

        '''
        if np.isfinite(day_raw).any():
            ax.plot(day_x, day_raw, lw=1.0, label="Raw N (daily mean)")
        if np.isfinite(day_corr).any():
            ax.plot(day_x, day_corr, lw=1.0, label=f"N × {fcol} (daily mean)")
        '''

        ax.set_ylabel("Neutron Count (cph)")
        ax.set_title(f"Raw vs {title_suffix} correction")
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=300)
        plt.close(fig)
'''
Flowing are the intermediate parameters e.g.  pv, rhov

'''
def plot_pressure_series(df, cfg, fig_dir):
    """Plot time series of pressure P and reference pressure p0."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    time_col = TIME_COL
    pressure_col = _cfg_get(cfg, "pressure_col", "PA_1")
    p0 = _cfg_get(cfg, "p0_ref", None)

    t = pd.to_datetime(df[time_col], errors="coerce")
    P = pd.to_numeric(df[pressure_col], errors="coerce")

    plt.figure(figsize=(14,4))
    plt.plot(t, P, label="Pressure $p$", lw=1)
    if p0 not in (None, "", "null"):
        plt.axhline(float(p0), color="r", ls="--", label=f"Reference pressure $p$0={p0:.2f}")
    plt.ylabel("Pressure (hPa)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "pressure_series.png", dpi=300)
    plt.close()


def plot_pv_series(df, cfg, fig_dir):
    """Plot time series of absolute humidity ρv and baseline ρv0."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    time_col = TIME_COL
    t = pd.to_datetime(df[time_col], errors="coerce")
    pv = pd.to_numeric(df["pv"], errors="coerce") if "pv" in df else None
    pv0 = _cfg_get(cfg, "rhov0_ref", None)

    if pv is None:
        print("[PV] No pv column found, skip plotting.")
        return

    plt.figure(figsize=(14,4))
    plt.plot(t, pv, lw=1, label="Absolute humidity $rhov$")
    if pv0 not in (None, "", "null"):
        plt.axhline(float(pv0), color="r", ls="--", label=f"$rhov$0={pv0:.4f}")
    plt.ylabel("Absolute humidity (kg/m³)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "rhov_series.png", dpi=300)
    plt.close()


def plot_agb_series(df: pd.DataFrame, cfg: dict, fig_dir) -> None:
    """Plot above-ground biomass (AGB) time series.

    The line shows the stepwise AGB series mapped to the CRNS time axis.
    Markers show only the original measured observations retained in the
    transient ``_agb_daily_lookup`` before forward filling.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Determine AGB column
    agb_col = _cfg_get(cfg, "agb_col", None)
    if agb_col and agb_col in df.columns:
        src_col = agb_col
    elif "AGB" in df.columns:
        src_col = "AGB"
    else:
        return  # no AGB data

    agb = pd.to_numeric(df[src_col], errors="coerce")

    display_unit = "kg/m²"
    if "AGB_KGM2" in df.columns:
        agb_plot_vals = pd.to_numeric(df["AGB_KGM2"], errors="coerce")
    else:
        agb_plot_vals = agb

    t = pd.to_datetime(
        df["DATETIME"] if "DATETIME" in df.columns else df[TIME_COL],
        errors="coerce",
    )

    # Preserve the sparse measurements separately from the completed series.
    measured_lookup = _cfg_get(cfg, "_agb_daily_lookup", {})
    if isinstance(measured_lookup, dict) and measured_lookup:
        measured_agb = pd.Series(
            measured_lookup,
            dtype="float64",
        )
        measured_agb.index = pd.to_datetime(measured_agb.index, errors="coerce")
        measured_agb = measured_agb[
            measured_agb.index.notna() & measured_agb.notna()
        ].sort_index()
    else:
        measured_agb = pd.Series(dtype="float64")

    mask = t.notna() & agb_plot_vals.notna() & (agb_plot_vals > 0)
    if not measured_agb.empty:
        mask &= t >= measured_agb.index.min()
    t_plot   = t[mask]
    agb_plot = agb_plot_vals[mask]

    if t_plot.empty:
        return

    # Sort by date (DATE_AGB rows are not necessarily in time order)
    order    = t_plot.argsort()
    t_plot   = t_plot.iloc[order]
    agb_plot = agb_plot.iloc[order]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(
        t_plot,
        agb_plot,
        where="post",
        color="#4a7c59",
        linewidth=1.2,
        label="Stepwise AGB used for correction",
    )
    if not measured_agb.empty:
        ax.scatter(
            measured_agb.index,
            measured_agb.values,
            color="#4a7c59",
            marker="o",
            s=24,
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
            label="Measured AGB",
        )
    ax.set_ylabel(f"Above-Ground Biomass ({display_unit})")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "agb_series.png", dpi=600)
    plt.close(fig)


def run_all_visualizations(df: pd.DataFrame,
                           cfg: dict,
                           site_out: Path | str,
                           time_col: str,
                           neutron_col: str) -> None:
    """
    Unified plotting entry point for Petrichor.

    Plotting rules
    --------------
    - Remove all QC-failed rows before any plotting or daily aggregation.
    - Daily mean plots require at least 12 valid hourly values per day.
    - Raw neutron count prefers column 'N' if available.
    """
    fig_dir = Path(site_out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df.copy()

    if "DATETIME" not in df_plot.columns:
        df_plot["DATETIME"] = pd.to_datetime(df_plot[time_col], errors="coerce")

    # Remove all QC-failed rows before plotting
    if "QC_OK" in df_plot.columns:
        qc_mask = df_plot["QC_OK"].fillna(False).astype(bool)
        df_plot = df_plot.loc[qc_mask].copy()
    else:
        print("[PLOT][WARN] QC_OK column not found. Plots will use all rows.")

    if df_plot.empty:
        print("[PLOT][WARN] No QC-passed rows remain after filtering. Skip plotting.")
        return

    # Raw / corrected neutron counts
    if "N" in df_plot.columns:
        df_plot["MOD"] = pd.to_numeric(df_plot["N"], errors="coerce")
    elif "MOD" in df_plot.columns:
        df_plot["MOD"] = pd.to_numeric(df_plot["MOD"], errors="coerce")
    else:
        df_plot["MOD"] = pd.to_numeric(df_plot[neutron_col], errors="coerce")

    if "N_corr" in df_plot.columns:
        df_plot["MOD_CORR"] = pd.to_numeric(df_plot["N_corr"], errors="coerce")
    elif "N_corr_int" in df_plot.columns:
        df_plot["MOD_CORR"] = pd.to_numeric(df_plot["N_corr_int"], errors="coerce")
    elif "N_corr_hum" in df_plot.columns:
        df_plot["MOD_CORR"] = pd.to_numeric(df_plot["N_corr_hum"], errors="coerce")
    else:
        df_plot["MOD_CORR"] = df_plot["MOD"]

    for c in ["F_PRESSURE", "F_HUMIDITY", "F_INTENSITY", "F_AGB"]:
        if c not in df_plot.columns:
            df_plot[c] = 1.0
        else:
            df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")

    if "SM" not in df_plot.columns:
        if "SWC_CRNS" in df_plot.columns:
            df_plot["SM"] = pd.to_numeric(df_plot["SWC_CRNS"], errors="coerce")
        elif "SWC" in df_plot.columns:
            df_plot["SM"] = pd.to_numeric(df_plot["SWC"], errors="coerce")
        elif "SWC_1" in df_plot.columns:
            df_plot["SM"] = pd.to_numeric(df_plot["SWC_1"], errors="coerce")

    if "SM_SMOOTH" in df_plot.columns:
        df_plot["SM_SMOOTH"] = pd.to_numeric(df_plot["SM_SMOOTH"], errors="coerce")

    if "D86avg" not in df_plot.columns:
        df_plot["D86avg"] = pd.NA

    plot_quicklook(df_plot, time_col="DATETIME", fig_dir=fig_dir)
    plot_sm_colourts(df_plot, time_col="DATETIME", fig_dir=fig_dir)
    plot_hourly_unsmoothed_sm(df_plot, time_col="DATETIME", fig_dir=fig_dir)
    plot_sm_multiscale(df_plot, time_col="DATETIME", fig_dir=fig_dir)
    plot_pressure_series(df_plot, cfg, fig_dir)
    plot_pv_series(df_plot, cfg, fig_dir)
    save_individual_correction_factor_plots(df_plot, time_col="DATETIME", fig_dir=fig_dir)
    save_raw_vs_single_factor_plots(df_plot, time_col="DATETIME", fig_dir=fig_dir)
    # AGB series uses the full (pre-QC) df so DATE_AGB rows are not lost
    plot_agb_series(df, cfg, fig_dir)
