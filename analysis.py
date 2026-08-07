# _*_ coding: UTF-8 _*_
"""
analysis.py - statistics, aggregation, metadata, and output writing for Petrichor

This module handles post-processing products generated after the core correction
and soil-moisture workflow has finished.

Main responsibilities:
- build basic run statistics
- aggregate time series to daily, monthly, and yearly outputs
- export full and aggregated CSV files
- export descriptive statistics tables
- build and write run metadata JSON
- provide one consolidated export entry point for main.py

This module does not perform the physical neutron corrections itself.
Its job is to turn processed dataframes into reproducible output products.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import hashlib
import subprocess
import math
import numbers

import pandas as pd

from src.processing import PARTICLE_DENSITY


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


def _sha256_file(path: Path) -> str | None:
    """Return SHA256 hex digest of a file, or None if not readable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _sha256_json(obj: dict) -> str | None:
    """Return SHA256 hex digest of a JSON-serializable dict with stable ordering."""
    try:
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
    except Exception:
        return None


def _git_commit(project_root: Path) -> str | None:
    """Return current git commit hash if available, else None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _round_nested_numbers(obj, decimals: int = 3):
    """
    Recursively round floats inside nested dict/list structures.
    Keep ints, bools, strings, and datetimes unchanged.
    """
    if isinstance(obj, dict):
        return {k: _round_nested_numbers(v, decimals) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_round_nested_numbers(v, decimals) for v in obj]

    if isinstance(obj, bool):
        return obj

    if isinstance(obj, numbers.Integral):
        return obj

    if isinstance(obj, numbers.Real):
        value = float(obj)
        if math.isfinite(value):
            return round(value, decimals)
        return value

    return obj


def basic_stats_and_aggregates(
    df: pd.DataFrame,
    time_col: str,
    qc_col: str = "QC_OK",
    mean_cols: list | None = None,
    sum_cols: list | None = None,
):
    """
    Basic statistics and temporal aggregation.

    By default, aggregation uses QC_OK == True rows if qc_col exists.
    """
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not in df columns")

    df2 = df.copy()
    df2[time_col] = pd.to_datetime(df2[time_col], errors="coerce")
    df2 = df2.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

    if qc_col in df2.columns:
        df2_qc = df2[df2[qc_col].astype(bool)].copy()
    else:
        df2_qc = df2

    if mean_cols is None:
        cand = [
            "N",
            "N_corr",
            "SM",
            "SM_SMOOTH",
            "SWC_CRNS",
            "D86_10m",
            "D86_75m",
            "D86_150m",
            "D86avg",
            "D86avg_12h",
            "PA_1",
            "TA_1",
            "RH_1",
            "F_PRESSURE",
            "F_HUMIDITY",
            "F_INTENSITY",
            "F_AGB",
        ]
        mean_cols = [c for c in cand if c in df2_qc.columns]

    if sum_cols is None:
        sum_cols = [c for c in ["PREC"] if c in df2_qc.columns]

    mean_cols = [c for c in (mean_cols or []) if c in df2_qc.columns]
    sum_cols = [c for c in (sum_cols or []) if c in df2_qc.columns]

    mean_df = (
        df2_qc[mean_cols].select_dtypes(include="number")
        if mean_cols
        else df2_qc.select_dtypes(include="number")
    )
    sum_df = (
        df2_qc[sum_cols].select_dtypes(include="number")
        if sum_cols
        else None
    )

    stats = {
        "count_total": int(len(df2)),
        "count_qc": int(len(df2_qc)),
        "qc_pass_rate": float(len(df2_qc) / len(df2)) if len(df2) else 0.0,
        "time_start": df2.index.min(),
        "time_end": df2.index.max(),
        "describe": mean_df.describe().T,
    }

    daily_mean = mean_df.resample("D").mean()
    monthly_mean = mean_df.resample("MS").mean()
    yearly_mean = mean_df.resample("YS").mean()

    if sum_df is not None and len(sum_df.columns) > 0:
        # Preserve a fully missing period as NaN instead of reporting a false
        # zero total. A period with at least one observation is still summed.
        daily_sum = sum_df.resample("D").sum(min_count=1)
        monthly_sum = sum_df.resample("MS").sum(min_count=1)
        yearly_sum = sum_df.resample("YS").sum(min_count=1)

        daily = pd.concat([daily_mean, daily_sum], axis=1).reset_index()
        monthly = pd.concat([monthly_mean, monthly_sum], axis=1).reset_index()
        yearly = pd.concat([yearly_mean, yearly_sum], axis=1).reset_index()
    else:
        daily = daily_mean.reset_index()
        monthly = monthly_mean.reset_index()
        yearly = yearly_mean.reset_index()

    for frame in (daily, monthly, yearly):
        frame.rename(columns={frame.columns[0]: time_col}, inplace=True)

    return stats, daily, monthly, yearly


def write_timeseries_outputs(
    site_out: Path,
    df: pd.DataFrame,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    yearly: pd.DataFrame,
    time_col: str = "TIMESTAMP",
    decimals: int = 3,
):
    """
    Write main timeseries and aggregated tables to CSV.

    Rules
    -----
    - Most floating-point variables are rounded to `decimals` only at export time.
    - Neutron-count-related columns are exported as integers without decimal places.
    - Missing values in integer-like columns remain blank in CSV.
    """
    data_dir = Path(site_out) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    integer_count_cols = {
        "N",
        "N_CPH",
        "N_corr_press",
        "N_corr_hum",
        "N_corr_int",
        "N_corr",
        "N_HUM_ONLY",
        "N_INT_ONLY",
        "N_AGB_ONLY",
        "MOD",
        "MOD_CORR",
        "MOD_ERR",
        "MOD_CORR_PLUS",
        "MOD_CORR_MINUS",
    }

    # Preferred column order for timeseries_full.csv.
    # Columns not listed here are appended at the end in their original order.
    _COLUMN_ORDER = [
        # Time
        time_col,
        # Raw sensor data
        "N", "PA_1", "PA_2", "TA_1", "TA_2", "RH_1", "RH_2", "BATT",
        # AGB vegetation
        "AGB", "AGB_KGM2", "DATE", "DATE_AGB", "AGB_CORRECTION_APPLIED",
        # Correction factors
        "F_PRESSURE", "F_HUMIDITY", "F_INTENSITY", "F_AGB",
        # Neutron correction chain → final corrected count
        "N_corr_press", "N_corr_hum", "N_corr_int",
        "MOD_CORR", "MOD_ERR", "MOD_CORR_PLUS", "MOD_CORR_MINUS",
        # QC
        "QC_FLAG1", "QC_FLAG2", "QC_FLAG3", "QC_FLAG4", "QC_OK",
        # Soil moisture
        "SM", "SM_SMOOTH", "SM_PLUS_ERR", "SM_MINUS_ERR",
        # Effective sensing depth
        "D86_10m", "D86_75m", "D86_150m", "D86avg", "D86avg_12h",
    ]

    def _prepare_export_frame(frame: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
        out = frame.copy()
        # Drop internal/intermediate columns that should not appear in output.
        # SWC_CRNS is an exact copy of SM (see processing.py run_crns_qc_and_theta_pipeline).
        # pv is the intermediate absolute humidity value used only for the humidity correction factor.
        out = out.drop(
            columns=["NET_SCALED_FLAG", "MOD", "NET_MIN_SECONDS", "N_corr", "FLAG",
                     "SWC_CRNS", "pv"],
            errors="ignore",
        )

        num_cols = out.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            out[num_cols] = out[num_cols].round(decimals)

        cols_to_int = [c for c in integer_count_cols if c in out.columns]
        for col in cols_to_int:
            s = pd.to_numeric(out[col], errors="coerce").round()
            out[col] = s.astype("Int64")

        # Reorder columns: preferred list first, then any remaining columns.
        ordered = [c for c in _COLUMN_ORDER if c in out.columns]
        remaining = [c for c in out.columns if c not in ordered]
        out = out[ordered + remaining]

        return out

    df_out = _prepare_export_frame(df, decimals=decimals)
    daily_out = _prepare_export_frame(daily, decimals=decimals)
    monthly_out = _prepare_export_frame(monthly, decimals=decimals)
    yearly_out = _prepare_export_frame(yearly, decimals=decimals)

    df_out.to_csv(
        data_dir / "timeseries_full.csv",
        index=False,
        float_format=f"%.{decimals}f",
    )

    daily_out.to_csv(
        data_dir / "timeseries_daily_mean.csv",
        index=False,
        float_format=f"%.{decimals}f",
    )
    monthly_out.to_csv(
        data_dir / "timeseries_monthly_mean.csv",
        index=False,
        float_format=f"%.{decimals}f",
    )
    yearly_out.to_csv(
        data_dir / "timeseries_yearly_mean.csv",
        index=False,
        float_format=f"%.{decimals}f",
    )

def write_descriptive_stats(site_out: Path, stats: dict, decimals: int = 3) -> None:
    """Write descriptive statistics table to CSV."""
    try:
        if not isinstance(stats, dict):
            return
        if "describe" not in stats:
            return

        describe_df = stats["describe"]
        if not hasattr(describe_df, "to_csv"):
            return

        data_dir = Path(site_out) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        desc_path = data_dir / "stats_describe.csv"
        describe_df = describe_df.copy()

        num_cols = describe_df.select_dtypes(include="number").columns
        describe_df[num_cols] = describe_df[num_cols].round(decimals)

        describe_df.to_csv(desc_path, index=True, float_format=f"%.{decimals}f")
        print(f"[OUT] Descriptive statistics written to: {desc_path.name}")

    except Exception as e:
        print(f"[OUT][WARNING] Failed to write descriptive statistics: {e}")


def build_qc_summary(df: pd.DataFrame, qc_cols: list[str] | None = None) -> dict:
    """Build a simple count summary for QC flag columns."""
    if qc_cols is None:
        qc_cols = ["QC_FLAG1", "QC_FLAG2", "QC_FLAG3", "QC_FLAG4", "QC_OK"]

    qc_summary: dict = {}
    for col in qc_cols:
        if col in df.columns:
            qc_summary[col] = int(df[col].astype(bool).sum())

    return qc_summary


def build_run_metadata(
    *,
    df: pd.DataFrame,
    stats: dict,
    site_id: str,
    run_cfg: dict,
    project_root: Path,
    input_path: Path,
    field_json_path: Path,
    Rc=None,
    beta_B=None,
    p0_ref=None,
    rhov0_ref=None,
    jung_ref=None,
    nmdb_station=None,
) -> dict:
    """Build one metadata dictionary for the current run."""
    metadata: dict = {}

    metadata["site_id"] = site_id
    metadata["field_id"] = _cfg_get(run_cfg, "field_id", None)
    metadata["created_utc"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata["git_commit"] = _git_commit(project_root)

    metadata["input_csv_path"] = str(input_path)
    metadata["input_csv_sha256"] = _sha256_file(Path(input_path))
    metadata["field_json_path"] = str(field_json_path)
    metadata["field_json_sha256"] = _sha256_file(Path(field_json_path))
    metadata["config_sha256"] = _sha256_json(run_cfg)

    if isinstance(stats, dict):
        for key in ("count_total", "count_qc", "qc_pass_rate", "time_start", "time_end"):
            if key in stats:
                value = stats[key]
                if hasattr(value, "isoformat"):
                    value = value.isoformat()
                metadata[key] = value

    metadata["parameters"] = {
        "Rc": Rc,
        "beta_B": beta_B,
        "p0_ref": p0_ref,
        "rhov0_ref": rhov0_ref,
        "jung_ref": jung_ref,
        "N0": _cfg_get(run_cfg, "N0"),
        "theta_method": _cfg_get(run_cfg, "theta_method", "desilets"),
        "intensity_method": _cfg_get(run_cfg, "intensity_method"),
        "nmdb_station": nmdb_station,
        "a0": _cfg_get(run_cfg, "a0"),
        "a1": _cfg_get(run_cfg, "a1"),
        "a2": _cfg_get(run_cfg, "a2"),
        "bulk_density": _cfg_get(run_cfg, "bulk_density"),
        "density": PARTICLE_DENSITY,
        "sm_max": _cfg_get(run_cfg, "sm_max"),
        "lattice_water": _cfg_get(run_cfg, "lattice_water"),
        "soil_organic_carbon": _cfg_get(run_cfg, "soil_organic_carbon"),
    }

    qc_summary = build_qc_summary(df)
    if qc_summary:
        metadata["qc_summary"] = qc_summary

    return metadata


def write_run_metadata(site_out: Path, metadata: dict, decimals: int = 3) -> None:
    """Write run metadata JSON."""
    try:
        data_dir = Path(site_out) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        meta_path = data_dir / "run_metadata.json"
        metadata = _round_nested_numbers(metadata, decimals=decimals)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[OUT] Run metadata written to: {meta_path.name}")

    except Exception as e:
        print(f"[OUT][WARNING] Failed to write metadata: {e}")


def run_analysis_exports(
    *,
    df: pd.DataFrame,
    time_col: str,
    site_out: Path,
    site_id: str,
    run_cfg: dict,
    project_root: Path,
    input_path: Path,
    field_json_path: Path,
    Rc=None,
    beta_B=None,
    p0_ref=None,
    rhov0_ref=None,
    jung_ref=None,
    nmdb_station=None,
    neutron_col=None,
    mean_cols: list[str] | None = None,
    sum_cols: list[str] | None = None,
    decimals: int = 3,
):
    """
    One-call analysis/export entry point for main.py.

    This function performs:
    1) basic statistics and aggregation
    2) CSV export of full / daily / monthly / yearly tables
    3) descriptive statistics export
    4) run metadata export
    5) site summary text export
    """
    if mean_cols is None:
        mean_cols = [
            "SM",
            "SM_SMOOTH",
            "D86avg",
            "D86avg_12h",
            "F_INTENSITY",
            "F_PRESSURE",
            "F_HUMIDITY",
        ]

    if sum_cols is None:
        sum_cols = ["PREC"]

    stats, daily, monthly, yearly = basic_stats_and_aggregates(
        df=df,
        time_col=time_col,
        qc_col="QC_OK",
        mean_cols=mean_cols,
        sum_cols=sum_cols,
    )

    write_timeseries_outputs(
        site_out=site_out,
        df=df,
        daily=daily,
        monthly=monthly,
        yearly=yearly,
        time_col=time_col,
        decimals=decimals,
    )
    print(f"[OUT] Time series tables written to: {Path(site_out) / 'data'}")

    write_descriptive_stats(site_out=site_out, stats=stats, decimals=decimals)

    metadata = build_run_metadata(
        df=df,
        stats=stats,
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
    )
    write_run_metadata(site_out=site_out, metadata=metadata, decimals=decimals)

    try:
        from src.processing import write_site_summary

        write_site_summary(
            site_out=site_out,
            site_id=site_id,
            cfg=run_cfg,
            Rc=Rc,
            beta_B=beta_B,
            p0_ref=p0_ref,
            rhov0_ref=rhov0_ref,
            jung_ref=jung_ref,
            N0=_cfg_get(run_cfg, "N0"),
            theta_method=_cfg_get(run_cfg, "theta_method", "desilets"),
            neutron_col=neutron_col,
            extra={
                "nmdb_station": nmdb_station,
            },
            df=df,
            input_path=input_path,
            field_json_path=field_json_path,
        )
    except Exception as e:
        print(f"[OUT][WARNING] Failed to write site summary: {e}")

    return stats, daily, monthly, yearly, metadata
