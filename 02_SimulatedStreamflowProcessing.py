# -*- coding: utf-8 -*-
"""
02_SimulatedStreamflowProcessing.py
================================================================================
Combined driver for the two independent stages that consume the raw GHM
simulated daily streamflow (Ji et al., 2025). Stage A and Stage B are
SIBLINGS, not a producer -> consumer pair: both load the same raw NetCDF
independently, using their own pre-processing logic, and write to separate
stage folders.

Stage A -- simulated streamflow feature extraction (was 03_Streamflow-Process.py)
    Computes per-station statistical features (AMS stats, rolling-window
    annual maxima, daily-flow percentiles, high/low-flow frequency/duration,
    seasonality) directly from the raw NetCDF via inline xarray/pandas logic.

Stage B -- GEV fit on simulated AMS (was 04_Sim_GEV-Fit.py)
    Fits a GEV distribution (MLE, L-moments start) to the simulated annual
    maximum series per station, delegating all shared fitting/plotting logic
    to src/gev_fit_common.py.

This is a data-preparation stage: it produces intermediate CSV/NetCDF-derived
tables and diagnostic figures used by downstream training scripts, not a
manuscript figure directly.

Inputs
------
    data/raw/Sim-Dis/HY_stremflow-Cara-GRDC-35_cleaned.nc
        Dims  : merit (5969 stations) x time (14976 days, 1980-01-01..2020-12-31)
        Var   : streamflow (merit, time) float32, raw units m3/s
    data/proceed/Caravan-GRDC/station_locations.xlsx
        area_km2 per station, used by Stage A for Q normalisation
    data/proceed/Caravan-GRDC/00_GRDC-Caravan-Process/4_Cara-GRDC-35.nc
        Fallback source of station areas for Stage A if station_locations.xlsx
        is missing (Stage 1 output of 00_GRDC-Caravan-Process.py)
    data/proceed/Caravan-GRDC/03_Streamflow-Process/station_locations.xlsx
        area_km2 per station, used by Stage B for Q normalisation to mm/day

Outputs
-------
    data/proceed/Caravan-GRDC/03_Streamflow-Process/
        sim_flow_features_per_station.csv   (one row per station)
        report.txt                          (summary statistics)
        log.txt

    figures/Caravan-GRDC/03_Streamflow-Process/
        ams_distribution.png, rolling_features.png, flow_percentiles.png,
        high_low_flow.png, seasonality.png, feature_correlation.png

    data/proceed/Caravan-GRDC/04_Sim_GEV-Fit/
        sim_gev_station_params.csv          (mu, sigma, xi, Q_T per station)

    figures/Caravan-GRDC/04_Sim_GEV-Fit/
        freq_curves_sample.png, gev_param_distributions.png,
        ks_pvalue_distribution.png

Pre-processing (Stage A)
-------------------------
  Before computing any feature, raw Q (m3/s) is area-normalised to runoff
  depth (mm/day) using the same catchment area as Caravan observations:

      Q_mm_day = Q_m3s * 86.4 / area_km2

  This removes basin-size as a dominant signal and lets the model learn
  genuine hydrological regimes, enabling transfer to ungauged / future
  scenarios. Stations without a matching area_km2 entry are skipped
  (NaN row).

Features computed by Stage A (per station)
--------------------------------------------
  === AMS-based ===
  ams_mean, ams_median, ams_std, ams_cv, ams_min, ams_max, ams_skew, n_years

  === Rolling-window annual max (mirrors prec_roll*_ann_max_mean) ===
  roll3d_ann_max_mean, roll5d_ann_max_mean, roll7d_ann_max_mean,
  roll30d_ann_max_mean

  === Daily flow statistics ===
  flow_mean, flow_std, flow_cv, flow_q5, flow_q10, flow_q25, flow_q50,
  flow_q75, flow_q90, flow_q95

  === High / low flow frequency & duration (mirrors prec high/low) ===
  high_flow_freq  Fraction of days > 9 x median daily flow
  high_flow_dur   Mean run length of consecutive high-flow days
  low_flow_freq   Fraction of days < 0.1 x mean daily flow
  low_flow_dur    Mean run length of consecutive low-flow days
  zero_flow_frac  Fraction of days with flow <= 0

  === Seasonality ===
  seasonality_ratio  MaxMonthlyMean / MeanAnnualFlow (0=flat; high=flashy)
  peak_month         Calendar month of highest mean flow (1-12)
================================================================================
"""
from __future__ import annotations

import sys
import time
import shutil
import logging
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.stats import skew as scipy_skew

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import DATA_RAW, DATA_PROCEED, FIGURE_ROOT, stage_dir
from src.gev_fit_common import (
    GEVFitConfig,
    configure_plot_style,
    fit_all_stations,
    plot_freq_curves,
    plot_param_distributions,
    plot_ks_distribution,
)

# ============================================================
# Shared paths / logging (used by both stages)
# ============================================================
OUT_DATA_A = stage_dir(DATA_PROCEED, "03_Streamflow-Process")
OUT_FIG_A = stage_dir(FIGURE_ROOT, "03_Streamflow-Process")
OUT_DATA_B = stage_dir(DATA_PROCEED, "04_Sim_GEV-Fit")
OUT_FIG_B = stage_dir(FIGURE_ROOT, "04_Sim_GEV-Fit")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DATA_A / "log.txt", mode="w", encoding="utf-8"),
    ],
)
LOG = logging.getLogger(__name__)


# ================================================================================
# Stage A: simulated streamflow feature extraction (was 03_Streamflow-Process.py)
# ================================================================================

# ── Stage A paths & constants ────────────────────────────────────────────────
SIM_NC_PATH = DATA_RAW / "Sim-Dis" / "HY_stremflow-Cara-GRDC-35_cleaned.nc"
AREA_XLSX_A = DATA_PROCEED / "station_locations.xlsx"
AREA_SOURCE_NC = DATA_PROCEED / "00_GRDC-Caravan-Process" / "4_Cara-GRDC-35.nc"

ROLL_WINDOWS = [3, 5, 7, 30]
STRICT_POS = True
MIN_VALID_YR = 5


# ── Inspect metadata ───────────────────────────────────────────────────────
def inspect_metadata(ds: xr.Dataset) -> dict:
    """
    Print and return all available metadata (global attrs, variable attrs,
    coordinate attrs).
    """
    LOG.info("=" * 70)
    LOG.info("NC FILE METADATA INSPECTION")
    LOG.info("=" * 70)
    LOG.info(f"  File  : {SIM_NC_PATH}")
    LOG.info(f"  Dims  : {dict(ds.dims)}")
    LOG.info(f"  Coords: {list(ds.coords)}")

    LOG.info("")
    LOG.info("── Global attributes (dataset level) ──")
    if ds.attrs:
        for k, v in ds.attrs.items():
            LOG.info(f"    {k}: {v}")
    else:
        LOG.info("    (none)")

    LOG.info("")
    LOG.info("── Variable attributes ──")
    units_found = {}
    for vname in list(ds.data_vars) + list(ds.coords):
        v = ds[vname]
        if v.attrs:
            attr_str = "  ".join(f"{k}={val}" for k, val in v.attrs.items())
            LOG.info(f"    {vname}: {attr_str}")
            if "units" in v.attrs:
                units_found[vname] = v.attrs["units"]
        else:
            LOG.info(f"    {vname}: (no attrs)")

    LOG.info("")
    if units_found:
        LOG.info(f"  ✓ Units found in attrs: {units_found}")
    else:
        LOG.info("  ⚠ No 'units' attribute found in any variable.")
        LOG.info("    Simulated streamflow units assumed to be m3/s based on data range.")

    LOG.info("=" * 70)
    return units_found


# ── Feature extraction ─────────────────────────────────────────────────────
def _run_lengths(bool_arr: np.ndarray) -> float:
    """Mean run length of True spans in a boolean array. Returns 0 if none."""
    if not bool_arr.any():
        return 0.0
    diff = np.diff(bool_arr.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if bool_arr[0]:
        starts = np.concatenate([[0], starts])
    if bool_arr[-1]:
        ends = np.concatenate([ends, [len(bool_arr)]])
    lengths = ends - starts
    return float(lengths.mean()) if len(lengths) > 0 else 0.0


def extract_features_for_station(sf_daily: np.ndarray, times_pd: pd.DatetimeIndex) -> dict:
    """
    Extract all features for a single station.

    Parameters
    ----------
    sf_daily : 1-D float32 array (n_days,) -- raw daily streamflow
    times_pd : pd.DatetimeIndex -- corresponding dates

    Returns
    -------
    Dict of scalar features (float or int). All NaN if insufficient data.
    """
    NAN_REC = {
        "n_years": 0,
        "ams_mean": np.nan, "ams_median": np.nan, "ams_std": np.nan,
        "ams_cv": np.nan, "ams_min": np.nan, "ams_max": np.nan,
        "ams_skew": np.nan,
        "roll3d_ann_max_mean": np.nan,
        "roll5d_ann_max_mean": np.nan,
        "roll7d_ann_max_mean": np.nan,
        "roll30d_ann_max_mean": np.nan,
        "flow_mean": np.nan, "flow_std": np.nan, "flow_cv": np.nan,
        "flow_q5": np.nan, "flow_q10": np.nan, "flow_q25": np.nan,
        "flow_q50": np.nan, "flow_q75": np.nan, "flow_q90": np.nan,
        "flow_q95": np.nan,
        "high_flow_freq": np.nan, "high_flow_dur": np.nan,
        "low_flow_freq": np.nan, "low_flow_dur": np.nan,
        "zero_flow_frac": np.nan,
        "seasonality_ratio": np.nan, "peak_month": np.nan,
    }

    # Build pandas Series for easy resampling
    sf = pd.Series(sf_daily.astype(float), index=times_pd)
    sf_valid = sf.copy()
    if STRICT_POS:
        sf_valid[sf_valid <= 0] = np.nan

    # ── AMS ──────────────────────────────────────────────────────────────
    ams = sf_valid.resample("YE").max().dropna()
    n_yr = len(ams)
    if n_yr < MIN_VALID_YR:
        return NAN_REC

    res = {"n_years": int(n_yr)}
    ams_vals = ams.values
    res["ams_mean"] = float(np.mean(ams_vals))
    res["ams_median"] = float(np.median(ams_vals))
    res["ams_std"] = float(np.std(ams_vals, ddof=1)) if n_yr > 1 else np.nan
    res["ams_cv"] = res["ams_std"] / res["ams_mean"] if res["ams_mean"] > 0 else np.nan
    res["ams_min"] = float(np.min(ams_vals))
    res["ams_max"] = float(np.max(ams_vals))
    res["ams_skew"] = float(scipy_skew(ams_vals)) if n_yr >= 3 else np.nan

    # ── Rolling-window annual max mean (mirrors prec_roll*_ann_max_mean) ─
    # Use non-NaN series for rolling
    sf_pos = sf.where(sf > 0) if STRICT_POS else sf
    for w in ROLL_WINDOWS:
        roll_max = sf_pos.rolling(w, min_periods=w).max()
        ann_maxima = roll_max.resample("YE").max().dropna()
        res[f"roll{w}d_ann_max_mean"] = (
            float(ann_maxima.mean()) if len(ann_maxima) >= MIN_VALID_YR else np.nan
        )

    # ── Daily flow statistics ─────────────────────────────────────────────
    all_vals = sf.dropna().values
    if len(all_vals) > 0:
        res["flow_mean"] = float(np.mean(all_vals))
        res["flow_std"] = float(np.std(all_vals, ddof=1))
        res["flow_cv"] = res["flow_std"] / res["flow_mean"] if res["flow_mean"] > 0 else np.nan
        for pct in [5, 10, 25, 50, 75, 90, 95]:
            res[f"flow_q{pct}"] = float(np.percentile(all_vals, pct))
        res["zero_flow_frac"] = float(np.mean(sf.values <= 0))
    else:
        for k in ["flow_mean", "flow_std", "flow_cv",
                   "flow_q5", "flow_q10", "flow_q25", "flow_q50",
                   "flow_q75", "flow_q90", "flow_q95", "zero_flow_frac"]:
            res[k] = np.nan

    # ── High / low flow frequency & duration ─────────────────────────────
    # High flow: > 9 x median  (WHO/WMO "flood threshold" analogy to high_prec_freq)
    # Low  flow: < 0.1 x mean  (drought baseline)
    sf_arr = sf.values
    mean_f = res.get("flow_mean", np.nan)
    med_f = res.get("flow_q50", np.nan)

    if np.isfinite(mean_f) and np.isfinite(med_f) and med_f > 0:
        high_mask = sf_arr > 9.0 * med_f
        low_mask = (sf_arr > 0) & (sf_arr < 0.1 * mean_f)
        res["high_flow_freq"] = float(np.mean(high_mask))
        res["high_flow_dur"] = _run_lengths(high_mask)
        res["low_flow_freq"] = float(np.mean(low_mask))
        res["low_flow_dur"] = _run_lengths(low_mask)
    else:
        res["high_flow_freq"] = np.nan
        res["high_flow_dur"] = np.nan
        res["low_flow_freq"] = np.nan
        res["low_flow_dur"] = np.nan

    # ── Seasonality ───────────────────────────────────────────────────────
    monthly_mean = sf.groupby(sf.index.month).mean()
    if mean_f and mean_f > 0 and len(monthly_mean) == 12:
        res["seasonality_ratio"] = float(monthly_mean.max() / mean_f)
        res["peak_month"] = int(monthly_mean.idxmax())
    else:
        res["seasonality_ratio"] = np.nan
        res["peak_month"] = np.nan

    return res


def _infer_station_key(ds_ref: xr.Dataset) -> str:
    for key in ["station", "merit", "gauge_id", "station_id"]:
        if key in ds_ref.coords or key in ds_ref.data_vars:
            return key
    raise KeyError(
        "No station id key found in area source NC "
        "(expected one of station/merit/gauge_id/station_id)."
    )


def _infer_area_key(ds_ref: xr.Dataset) -> str:
    for key in ["static_area", "area_km2", "area", "drainage_area"]:
        if key in ds_ref.data_vars:
            return key
    raise KeyError(
        "No area variable found in area source NC "
        "(expected one of static_area/area_km2/area/drainage_area)."
    )


def _build_area_table_from_source_nc(source_nc: Path, target_station_ids: np.ndarray) -> pd.DataFrame:
    if not source_nc.exists():
        raise FileNotFoundError(f"Area source NC not found: {source_nc}")

    ds_ref = xr.open_dataset(source_nc)
    try:
        sid_key = _infer_station_key(ds_ref)
        area_key = _infer_area_key(ds_ref)

        src_ids = ds_ref[sid_key].values.astype(str)
        src_area = ds_ref[area_key].values.astype(float)
        area_df = pd.DataFrame({"station_id": src_ids, "area_km2": src_area})

        area_df["station_id"] = area_df["station_id"].astype(str).str.strip()
        area_df = area_df[np.isfinite(area_df["area_km2"]) & (area_df["area_km2"] > 0)].copy()

        target_set = set(pd.Series(target_station_ids).astype(str).str.strip())

        # Exact ID matching first.
        exact = area_df[area_df["station_id"].isin(target_set)].copy()
        if not exact.empty:
            return exact[["station_id", "area_km2"]].drop_duplicates("station_id")

        # Fallback: match suffix IDs (e.g., GRDC_12345 -> 12345).
        area_df["station_id_norm"] = area_df["station_id"].str.split("_").str[-1].str.strip()
        norm = area_df[area_df["station_id_norm"].isin(target_set)].copy()
        if norm.empty:
            raise ValueError(
                "No station IDs matched between target flow NC and area source NC."
            )

        norm = norm.rename(columns={"station_id_norm": "station_id"})
        return norm[["station_id", "area_km2"]].drop_duplicates("station_id")
    finally:
        ds_ref.close()


def _ensure_area_file(
    area_file_path: Path,
    source_nc: Path,
    target_station_ids: np.ndarray,
) -> Path:
    generated_path = OUT_DATA_A / area_file_path.name
    if generated_path.exists():
        return generated_path

    # If configured area table exists elsewhere, copy it into this stage folder.
    if area_file_path.exists():
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(area_file_path, generated_path)
        LOG.info(f"  Copied area table to stage folder: {generated_path}")
        return generated_path

    # Otherwise auto-build from source NC into this stage folder.
    LOG.warning(
        f"  area table not found at configured path: {area_file_path}; "
        f"auto-building to stage folder: {generated_path}"
    )
    area_df = _build_area_table_from_source_nc(source_nc, target_station_ids)
    generated_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        area_df.to_excel(generated_path, index=False)
        LOG.info(f"  Auto-generated area file: {generated_path} ({len(area_df):,} rows)")
        return generated_path
    except Exception as exc:
        csv_path = generated_path.with_suffix(".csv")
        area_df.to_csv(csv_path, index=False, encoding="utf-8")
        LOG.warning(f"  Failed to write xlsx ({exc}); wrote csv instead: {csv_path}")
        return csv_path


def _read_area_table(area_file_path: Path) -> pd.DataFrame:
    if area_file_path.suffix.lower() == ".csv":
        return pd.read_csv(area_file_path, usecols=["station_id", "area_km2"])
    return pd.read_excel(area_file_path, usecols=["station_id", "area_km2"])


def run_feature_extraction(ds: xr.Dataset) -> pd.DataFrame:
    """
    Main loop: iterate over all stations, extract features, collect results.
    Raw Q (m3/s) is area-normalised to Q (mm/day) before feature computation.

    Returns pd.DataFrame with station_id as index.
    """
    station_ids = ds["merit"].values.astype(str)
    times_pd = pd.to_datetime(ds["time"].values)
    sf_matrix = ds["streamflow"].values  # (n_stations, n_time) float32

    # ── Load catchment areas ──────────────────────────────────────────────
    area_file = _ensure_area_file(AREA_XLSX_A, AREA_SOURCE_NC, station_ids)
    area_df = _read_area_table(area_file)
    area_df["station_id"] = area_df["station_id"].astype(str).str.strip()
    area_map: dict[str, float] = dict(zip(area_df["station_id"], area_df["area_km2"]))
    LOG.info(f"  Area data loaded: {len(area_map):,} entries from {area_file.name}")

    n_st = len(station_ids)
    n_match = sum(1 for s in station_ids if s in area_map and area_map[s] > 0)
    LOG.info(f"  Stations with valid area: {n_match:,} / {n_st:,}")
    if n_match == 0:
        LOG.error("  No station IDs matched between NC and area xlsx. Check ID format!")
        sys.exit(1)

    LOG.info(f"Extracting features: {n_st} stations x {len(times_pd)} days ...")
    LOG.info(f"  STRICT_POS       : {STRICT_POS}")
    LOG.info(f"  MIN_VALID_YR     : {MIN_VALID_YR}")
    LOG.info(f"  Roll windows (d) : {ROLL_WINDOWS}")
    LOG.info(f"  Unit normalisation: Q_mm_day = Q_m3s x 86.4 / area_km2")

    rows = []
    skipped = 0
    start_t = time.time()
    for i in tqdm(range(n_st), desc="  Stations", unit="stn"):
        sid = station_ids[i]
        area = area_map.get(sid, np.nan)
        if not np.isfinite(area) or area <= 0:
            # No area → cannot normalise → emit NaN row
            rec = {"station_id": sid, "n_years": 0}
            rows.append(rec)
            skipped += 1
            continue

        # Area normalisation: m3/s → mm/day
        # Q [mm/day] = Q [m3/s] x 86400 s/day x 1000 mm/m
        #                        / (area_km2 x 1e6 m2/km2)
        #            = Q [m3/s] x 86.4 / area_km2
        sf_norm = sf_matrix[i].astype(float) * 86.4 / area

        rec = extract_features_for_station(sf_norm, times_pd)
        rec["station_id"] = sid
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("station_id")
    elapsed = time.time() - start_t
    LOG.info(f"  Done in {elapsed:.1f} s  ({elapsed/n_st:.3f} s/station)")
    LOG.info(f"  Skipped (no area): {skipped:,}")
    return df


# ── Statistical summary ─────────────────────────────────────────────────────
def print_stats_summary(df: pd.DataFrame) -> str:
    """Print and return text summary of all features."""
    sep = "=" * 72
    lines = [
        sep,
        "  Stage A -- Simulated Streamflow: Per-Station Feature Summary",
        sep,
        f"  NC file : {SIM_NC_PATH.name}",
        f"  Stations total     : {len(df)}",
        f"  Stations with data : {int(df['n_years'].gt(0).sum())}",
        "",
    ]

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    lines.append(f"  {'Feature':<28s}  {'Mean':>10s}  {'Std':>10s}  "
                 f"{'P10':>8s}  {'P50':>8s}  {'P90':>8s}  {'N_valid':>8s}")
    lines.append("  " + "-" * 88)

    for col in numeric_cols:
        v = df[col].dropna().values
        if len(v) == 0:
            continue
        lines.append(
            f"  {col:<28s}  {np.mean(v):>10.4f}  {np.std(v):>10.4f}  "
            f"{np.percentile(v,10):>8.4f}  {np.percentile(v,50):>8.4f}  "
            f"{np.percentile(v,90):>8.4f}  {len(v):>8d}"
        )

    lines.append(sep)
    text = "\n".join(lines)
    LOG.info("\n" + text)
    (OUT_DATA_A / "report.txt").write_text(text, encoding="utf-8")
    LOG.info(f"  report.txt saved → {OUT_DATA_A / 'report.txt'}")
    return text


# ── Figures ───────────────────────────────────────────────────────────────
def _apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
    })


def fig_ams_distribution(df: pd.DataFrame):
    _apply_style()
    vals = df["ams_mean"].dropna().values
    cv = df["ams_cv"].dropna().values
    nyr = df["n_years"].dropna().values

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # AMS mean – log scale
    ax = axes[0, 0]
    ax.hist(np.log10(vals[vals > 0]), bins=60, color="#1565C0", alpha=0.7,
            edgecolor="white", density=True)
    ax.set_xlabel("log₁₀(Mean AMS)  [mm/day]", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Mean AMS (log₁₀ scale)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # AMS mean – linear scale CDF
    ax = axes[0, 1]
    sorted_v = np.sort(vals[vals > 0])
    cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
    ax.plot(np.log10(sorted_v), cdf, color="#1565C0", lw=1.5)
    ax.set_xlabel("log₁₀(Mean AMS)  [mm/day]", fontsize=10)
    ax.set_ylabel("Cumulative Fraction", fontsize=10)
    ax.set_title("CDF of Mean AMS", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # AMS CV histogram
    ax = axes[1, 0]
    ax.hist(cv[np.isfinite(cv)], bins=60, color="#AD1457", alpha=0.7,
            edgecolor="white", density=True)
    ax.set_xlabel("CV of AMS", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Inter-Annual Variability (CV)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Number of valid years
    ax = axes[1, 1]
    ax.hist(nyr[nyr > 0], bins=40, color="#2E7D32", alpha=0.7,
            edgecolor="white", density=True)
    ax.set_xlabel("Number of valid AMS years", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Record Length Distribution", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Simulated Annual Maximum Streamflow (mm/day) – Station Overview",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_FIG_A / "ams_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOG.info("  ams_distribution.png saved")


def fig_rolling_features(df: pd.DataFrame):
    _apply_style()
    roll_cols = [f"roll{w}d_ann_max_mean" for w in ROLL_WINDOWS]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    colors = ["#1565C0", "#7B1FA2", "#AD1457", "#2E7D32"]

    for ax, col, color, w in zip(axes, roll_cols, colors, ROLL_WINDOWS):
        vals = df[col].dropna().values
        vals = vals[vals > 0]
        if len(vals):
            ax.hist(np.log10(vals), bins=50, color=color, alpha=0.75,
                    edgecolor="white", density=True)
        ax.set_xlabel(f"log₁₀  [mm/day]", fontsize=9)
        ax.set_title(f"{w}-day rolling max\n(annual mean)", fontsize=10,
                     fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.text(0.97, 0.95, f"n={len(vals):,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")

    fig.suptitle("Rolling-Window Annual Max Streamflow (log₁₀, mm/day) – All Stations",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_FIG_A / "rolling_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOG.info("  rolling_features.png saved")


def fig_flow_percentiles(df: pd.DataFrame):
    _apply_style()
    pct_cols = ["flow_q5", "flow_q10", "flow_q25", "flow_q50",
                "flow_q75", "flow_q90", "flow_q95"]
    pct_labels = ["Q5", "Q10", "Q25", "Q50", "Q75", "Q90", "Q95"]

    data_list = [df[c].dropna().values for c in pct_cols]
    data_log = [np.log10(d[d > 0]) for d in data_list]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data_log, labels=pct_labels, patch_artist=True,
                     widths=0.5, showfliers=False, medianprops=dict(color="black", lw=2))
    cmap = plt.cm.get_cmap("Spectral", len(pct_cols))
    for patch, c in zip(bp["boxes"], range(len(pct_cols))):
        patch.set_facecolor(cmap(c))
        patch.set_alpha(0.75)

    ax.set_xlabel("Flow percentile", fontsize=11)
    ax.set_ylabel("log₁₀(Flow)  [mm/day]", fontsize=11)
    ax.set_title("Flow Duration Curve Percentiles (mm/day) – Across All Stations",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_FIG_A / "flow_percentiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOG.info("  flow_percentiles.png saved")


def fig_high_low_flow(df: pd.DataFrame):
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    specs = [
        ("high_flow_freq", "#AD1457", "High-flow frequency\n(days > 9×Q50)", axes[0, 0]),
        ("high_flow_dur", "#7B1FA2", "High-flow mean duration\n(consecutive days)", axes[0, 1]),
        ("low_flow_freq", "#1565C0", "Low-flow frequency\n(days < 0.1×mean)", axes[1, 0]),
        ("low_flow_dur", "#1E88E5", "Low-flow mean duration\n(consecutive days)", axes[1, 1]),
    ]
    for col, color, title, ax in specs:
        vals = df[col].dropna().values
        vals = vals[np.isfinite(vals)]
        if len(vals):
            ax.hist(vals, bins=55, color=color, alpha=0.75,
                    edgecolor="white", density=True)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.text(0.97, 0.95, f"n={len(vals):,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")

    fig.suptitle("High/Low Flow Frequency & Duration – All Stations",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_FIG_A / "high_low_flow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOG.info("  high_low_flow.png saved")


def fig_seasonality(df: pd.DataFrame):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Seasonality ratio distribution
    ax = axes[0]
    vals = df["seasonality_ratio"].dropna().values
    vals = vals[np.isfinite(vals) & (vals > 0)]
    ax.hist(vals, bins=60, color="#1565C0", alpha=0.75,
            edgecolor="white", density=True)
    ax.axvline(1.0, color="red", ls="--", lw=1.0, label="uniform=1")
    ax.set_xlabel("Seasonality Ratio  (MaxMonthMean / AnnualMean)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Streamflow Seasonality Ratio", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Peak month distribution (rose chart as bar)
    ax = axes[1]
    months = df["peak_month"].dropna().values.astype(int)
    months = months[(months >= 1) & (months <= 12)]
    counts = np.bincount(months, minlength=13)[1:]  # counts per month 1–12
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    bars = ax.bar(month_labels, counts, color="#AD1457", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Month of Peak Flow", fontsize=10)
    ax.set_ylabel("Number of Stations", fontsize=10)
    ax.set_title("Peak Month Distribution", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    str(c), ha="center", va="bottom", fontsize=7)

    fig.suptitle("Simulated Streamflow Seasonality – Station Distribution",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_FIG_A / "seasonality.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOG.info("  seasonality.png saved")


def fig_feature_correlation(df: pd.DataFrame):
    _apply_style()
    num_df = df.select_dtypes(include=np.number).dropna(how="all", axis=1)
    corr = num_df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")

    labels = corr.columns.tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            if abs(v) > 0.6:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="black" if abs(v) < 0.85 else "white")

    ax.set_title("Spearman Correlation Matrix – Simulated Flow Features (mm/day)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_FIG_A / "feature_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("  feature_correlation.png saved")


def run_stage_a() -> None:
    """Stage A entrypoint: simulated streamflow feature extraction."""
    t0 = time.time()
    LOG.info("=" * 72)
    LOG.info("Stage A -- Simulated Streamflow Feature Extraction")
    LOG.info("=" * 72)

    # ── 1. Open NC and inspect metadata ──────────────────────────────────
    LOG.info(f"\nOpening NC file: {SIM_NC_PATH}")
    if not SIM_NC_PATH.exists():
        LOG.error(f"  File NOT found: {SIM_NC_PATH}")
        sys.exit(1)

    ds = xr.open_dataset(SIM_NC_PATH)
    units_info = inspect_metadata(ds)

    n_stations = int(ds.dims["merit"])
    n_time = int(ds.dims["time"])
    t_start = pd.to_datetime(ds["time"].values[0]).strftime("%Y-%m-%d")
    t_end = pd.to_datetime(ds["time"].values[-1]).strftime("%Y-%m-%d")
    LOG.info(f"\n  Stations  : {n_stations:,}")
    LOG.info(f"  Time steps: {n_time:,}  ({t_start} → {t_end})")
    if units_info:
        LOG.info(f"  Units     : {units_info}")
    else:
        LOG.info(f"  Units     : not specified in metadata → assumed m3/s (will be area-normalised to mm/day)")

    # ── 2. Feature extraction ─────────────────────────────────────────────
    LOG.info("")
    df = run_feature_extraction(ds)
    ds.close()

    # ── 3. Save CSV ───────────────────────────────────────────────────────
    csv_path = OUT_DATA_A / "sim_flow_features_per_station.csv"
    df.to_csv(csv_path, encoding="utf-8")
    LOG.info(f"\n  CSV saved → {csv_path}  ({len(df)} rows × {len(df.columns)} cols)")

    # ── 4. Statistical summary ────────────────────────────────────────────
    LOG.info("")
    print_stats_summary(df)

    # ── 5. Figures ────────────────────────────────────────────────────────
    LOG.info("\nGenerating figures ...")
    fig_ams_distribution(df)
    fig_rolling_features(df)
    fig_flow_percentiles(df)
    fig_high_low_flow(df)
    fig_seasonality(df)
    fig_feature_correlation(df)

    # ── 6. Final summary ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    valid_stns = int(df["n_years"].gt(0).sum())
    LOG.info(f"\nStage A done in {elapsed:.1f} s")
    LOG.info(f"  Valid stations : {valid_stns} / {len(df)}")
    LOG.info(f"  Features       : {len(df.columns)}")
    LOG.info(f"  Output data    : {OUT_DATA_A}")
    LOG.info(f"  Output figures : {OUT_FIG_A}")
    LOG.info("=" * 72)


# ================================================================================
# Stage B: GEV fit on simulated AMS (was 04_Sim_GEV-Fit.py, delegates to
# src/gev_fit_common.py)
# ================================================================================

# ── Stage B paths & constants (mirrors configs/base.yaml: sim_gev_fit) ──────
SIM_GEV_NC_PATH = DATA_RAW / "Sim-Dis" / "HY_stremflow-Cara-GRDC-35_cleaned.nc"
SIM_GEV_AREA_XLSX = DATA_PROCEED / "03_Streamflow-Process" / "station_locations.xlsx"

SIM_GEV_RETURN_PERIODS = [2, 5, 10, 20, 50, 100]
SIM_GEV_MIN_YEARS = 5
SIM_GEV_N_SAMPLE = 24
SIM_GEV_SEED = 42
SIM_GEV_XI_CLIP_MIN = -0.8
SIM_GEV_XI_CLIP_MAX = 0.5
SIM_GEV_METHOD = "mle"
SIM_GEV_CSV_NAME = "sim_gev_station_params.csv"
SIM_GEV_NORMALIZE_TO_MM_DAY = True
SIM_GEV_UNIT_SCALE_FACTOR = 86.4


def run_stage_b() -> None:
    """Stage B entrypoint: GEV MLE fitting for all stations on simulated AMS."""
    cfg = GEVFitConfig(
        nc_file=SIM_GEV_NC_PATH,
        out_csv=OUT_DATA_B,
        out_fig=OUT_FIG_B,
        return_periods=SIM_GEV_RETURN_PERIODS,
        min_years=SIM_GEV_MIN_YEARS,
        n_sample=SIM_GEV_N_SAMPLE,
        seed=SIM_GEV_SEED,
        xi_clip_min=SIM_GEV_XI_CLIP_MIN,
        xi_clip_max=SIM_GEV_XI_CLIP_MAX,
        method=SIM_GEV_METHOD,
        csv_name=SIM_GEV_CSV_NAME,
        normalize_to_mm_day=SIM_GEV_NORMALIZE_TO_MM_DAY,
        area_xlsx=SIM_GEV_AREA_XLSX,
        unit_scale_factor=SIM_GEV_UNIT_SCALE_FACTOR,
    )

    configure_plot_style()

    t0 = time.time()
    LOG.info("=" * 60)
    LOG.info("  Stage B -- GEV MLE fitting for all stations (simulated AMS)")
    LOG.info("=" * 60)
    LOG.info("  Input NC: %s", cfg.nc_file)
    LOG.info("  Output CSV dir: %s", cfg.out_csv)
    LOG.info("  normalize_to_mm_day: %s", cfg.normalize_to_mm_day)
    if cfg.normalize_to_mm_day:
        LOG.info("  area_xlsx: %s", cfg.area_xlsx)
        LOG.info("  unit_scale_factor: %.3f", cfg.unit_scale_factor)

    df = fit_all_stations(cfg, LOG)

    csv_path = cfg.out_csv / cfg.csv_name
    df.to_csv(csv_path, index=False)
    LOG.info(f"  Parameters saved -> {csv_path}")

    plot_freq_curves(df, cfg, LOG)
    plot_param_distributions(df, cfg, LOG)
    plot_ks_distribution(df, cfg, LOG)

    elapsed = time.time() - t0
    LOG.info(f"  Stage B done in {elapsed:.1f} s  |  Figures -> {cfg.out_fig}")


# ============================================================
# Main
# ============================================================
def main():
    run_stage_a()
    run_stage_b()


if __name__ == "__main__":
    main()
