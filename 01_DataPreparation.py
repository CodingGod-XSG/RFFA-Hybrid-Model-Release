"""
01_DataPreparation.py  --  Data preparation pipeline (Stages 1-3)
================================================================================
Data preparation stage -- does not correspond to a manuscript figure directly.

This file merges three formerly-separate scripts into one strictly linear
pipeline (Stage 1 -> Stage 2 -> Stage 3). Each stage's internal logic
(algorithms, thresholds, formulas, random seeds) is unchanged from the
original scripts; only path/config plumbing was restructured to use
`src.paths` instead of `configs/base.yaml`.

Stage 1  (was 00_GRDC-Caravan-Process.py)
    Merge Caravan + GRDC-Caravan raw timeseries/attributes, extract AMS and
    scalar features per station, deduplicate overlapping gauges, and keep
    only stations with >=35 years of valid annual-maximum streamflow (AMS).
    Inputs:
        data/raw/Caravan/timeseries/csv/<region>/*.csv
        data/raw/Caravan/attributes/<region>/attributes_*_<region>.csv
        data/raw/GRDC-Caravan/timeseries/csv/grdc/*.csv
        data/raw/GRDC-Caravan/attributes/grdc/attributes_*_grdc.csv
    Outputs (data/proceed/Caravan-GRDC/00_GRDC-Caravan-Process/):
        1_Cara-GRDC-Merge.nc, 2_Cara-GRDC-rem-Dup.nc, 3_Cara-GRDC-35.nc,
        4_Cara-GRDC-35.nc, 4_Cara-GRDC.nc (alias), Processing_Report.md
    Figures (figures/Caravan-GRDC/00_GRDC-Caravan-Process/):
        correlation_table.csv, Fig1_Overview_Correlation.png,
        Fig2_Category_Correlation.png, Fig3_Scatter_Top_Features.png

Stage 2  (was 01_GEV-Fit.py, delegates to src/gev_fit_common.py)
    At-site GEV (MLE) fitting of AMS for every Stage-1 station.
    Input:
        data/proceed/Caravan-GRDC/00_GRDC-Caravan-Process/4_Cara-GRDC-35.nc
    Outputs (data/proceed/Caravan-GRDC/01_GEV-Fit/):
        gev_station_params.csv
    Figures (figures/Caravan-GRDC/01_GEV-Fit/):
        freq_curves_sample.png, gev_param_distributions.png,
        ks_pvalue_distribution.png

Stage 3  (was 02_Data-Clean.py)
    Quality-filter stations using the Stage-2 GEV fit: drop stations with an
    extreme shape parameter (xi outside [-0.5, 0.5]) or a KS goodness-of-fit
    p-value < 0.05.
    Inputs:
        data/proceed/Caravan-GRDC/00_GRDC-Caravan-Process/4_Cara-GRDC-35.nc
        data/proceed/Caravan-GRDC/01_GEV-Fit/gev_station_params.csv
    Outputs (data/proceed/Caravan-GRDC/02_Data-Clean/):
        gev_cleaned.csv, 4_Cara-GRDC-35_cleaned.nc, filter_summary.csv,
        stats_before.csv, stats_after.csv, removal_reasons.csv, report.txt
    Figures (figures/Caravan-GRDC/02_Data-Clean/):
        01_filter_funnel.png ... 10_q_ratio.png
================================================================================
"""
from __future__ import annotations

import os
import sys
import time
import warnings
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import DATA_RAW, DATA_PROCEED, FIGURE_ROOT, stage_dir  # noqa: E402

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata, t as scipy_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

from src.gev_fit_common import (  # noqa: E402
    GEVFitConfig,
    configure_plot_style,
    fit_all_stations,
    plot_freq_curves,
    plot_param_distributions,
    plot_ks_distribution,
)

# ============================================================
# Shared setup (was duplicated across all three original scripts)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ================================================================================
# Stage 1: Merge Caravan + GRDC-Caravan, extract AMS/features, dedupe, filter >=35yr
# (was 00_GRDC-Caravan-Process.py)
# ================================================================================

# ---- Stage 1 configuration (was configs/base.yaml: paths.*, pipeline.*) ----
CARAVAN_ROOT = DATA_RAW / "Caravan"
GRDC_ROOT = DATA_RAW / "GRDC-Caravan"

S1_DATA_DIR = stage_dir(DATA_PROCEED, "00_GRDC-Caravan-Process")
S1_FIGURE_DIR = stage_dir(FIGURE_ROOT, "00_GRDC-Caravan-Process")

OUT_NC_MERGE = S1_DATA_DIR / "1_Cara-GRDC-Merge.nc"
OUT_NC_DEDUP = S1_DATA_DIR / "2_Cara-GRDC-rem-Dup.nc"
OUT_NC_35 = S1_DATA_DIR / "3_Cara-GRDC-35.nc"
OUT_NC_FEAT_35 = S1_DATA_DIR / "4_Cara-GRDC-35.nc"
OUT_NC_FEAT = S1_DATA_DIR / "4_Cara-GRDC.nc"  # alias of the 35yr dataset
S1_REPORT_PATH = S1_DATA_DIR / "Processing_Report.md"

CARAVAN_REGIONS = ["camels", "camelsaus", "camelsbr", "camelscl",
                   "camelsgb", "hysets", "lamah"]

KEY_DYNAMIC_VARS = [
    "streamflow",
    "total_precipitation_sum",
    "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
    "temperature_2m_mean",
    "snow_depth_water_equivalent_mean",
    "surface_net_solar_radiation_mean",
    "volumetric_soil_water_layer_1_mean",
    "volumetric_soil_water_layer_2_mean",
]

ROLL_WINDOWS = [3, 5, 7, 30]
DIST_THRESH_KM = 5.0
AREA_REL_THRESH = 0.05
MIN_YEARS_35 = 35
N_WORKERS = os.cpu_count()


# ---- Helper utilities ----
def haversine_matrix(lat_a, lon_a, lat_b, lon_b):
    """Vectorised haversine: returns (n_a, n_b) distance matrix (km)."""
    R = 6371.0
    lat_a = np.radians(lat_a)[:, None]
    lat_b = np.radians(lat_b)[None, :]
    lon_a = np.radians(lon_a)[:, None]
    lon_b = np.radians(lon_b)[None, :]
    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _worker_extract_features(args):
    """
    Core Worker: reads one CSV with usecols, extracts AMS and scalar features,
    immediately discards the full daily DataFrame to conserve memory.

    Returns:
        (gauge_id, years_array, ams_values_array, scalar_dict)
        or (gauge_id, None, None, None) on failure.
    """
    fpath, gid, key_vars, roll_windows = args
    try:
        avail_cols = pd.read_csv(fpath, nrows=0).columns.tolist()
        cols_to_read = ['date'] + [v for v in key_vars if v in avail_cols]
        df = pd.read_csv(fpath, usecols=cols_to_read, parse_dates=['date'], index_col='date')
        if df.empty:
            return gid, None, None, None

        res = {}

        # ── 1. Annual Maximum Streamflow (AMS) ──────────────────────────────
        if 'streamflow' in df.columns:
            sf = df['streamflow']
            sf_valid = sf[sf > 0]                                # strict positive
            ams = sf_valid.resample('YE').max().dropna()
        else:
            ams = pd.Series(dtype=float)

        if len(ams) == 0:
            return gid, None, None, None

        res['mean_ann_max_sf'] = float(ams.mean())

        # ── 2. Long-term means for remaining dynamic vars ────────────────────
        for v in key_vars:
            if v != 'streamflow' and v in df.columns:
                res[f'ltmean_{v}'] = float(df[v].mean())

        # ── 3. Rolling precipitation features ───────────────────────────────
        if 'total_precipitation_sum' in df.columns:
            prec = df['total_precipitation_sum'].copy()
            prec[prec < 0] = np.nan
            for w in roll_windows:
                roll_p = prec.rolling(w, min_periods=w).sum()
                ann_max = roll_p.resample('YE').max()
                res[f'prec_roll{w}d_ann_max_mean'] = float(ann_max.mean())

        # ── 4. Rolling temperature features ─────────────────────────────────
        if 'temperature_2m_mean' in df.columns:
            temp = df['temperature_2m_mean']
            for w in roll_windows:
                roll_t = temp.rolling(w, min_periods=w).mean()
                res[f'temp_roll{w}d_ann_mean'] = float(roll_t.resample('YE').mean().mean())
                res[f'temp_roll{w}d_ann_max'] = float(roll_t.resample('YE').max().mean())

        return gid, ams.index.year.values, ams.values, res

    except Exception as exc:
        # Log to stderr so tqdm bar stays clean; no crash propagation
        print(f"[WORKER WARN] {gid}: {exc}", file=sys.stderr)
        return gid, None, None, None


class DataLoader:
    def iter_ts_paths_list(self) -> list:
        paths = []
        ts_root = CARAVAN_ROOT / "timeseries" / "csv"
        for region in CARAVAN_REGIONS:
            for fp in (ts_root / region).glob("*.csv"):
                paths.append((str(fp), fp.stem))
        grdc_ts = GRDC_ROOT / "timeseries" / "csv" / "grdc"
        if grdc_ts.exists():
            for fp in grdc_ts.glob("*.csv"):
                paths.append((str(fp), fp.stem))
        return sorted(paths)

    def load_static(self) -> pd.DataFrame:
        log.info("Loading static attributes …")
        frames = []
        attr_root = CARAVAN_ROOT / "attributes"
        for region in CARAVAN_REGIONS:
            region_dir = attr_root / region
            for ftype in ["caravan", "hydroatlas", "other"]:
                fp = region_dir / f"attributes_{ftype}_{region}.csv"
                if fp.exists():
                    df = pd.read_csv(fp, index_col="gauge_id", low_memory=False)
                    df["_source"] = "caravan"; df["_region"] = region
                    frames.append(df)
        grdc_attr = GRDC_ROOT / "attributes" / "grdc"
        for ftype in ["caravan", "hydroatlas", "other", "additional"]:
            fp = grdc_attr / f"attributes_{ftype}_grdc.csv"
            if fp.exists():
                df = pd.read_csv(fp, index_col="gauge_id", low_memory=False)
                df["_source"] = "grdc"; df["_region"] = "grdc"
                frames.append(df)
        if not frames:
            raise RuntimeError("No static attribute files found – check paths.")
        combined = pd.concat(frames, axis=0, join="outer")
        return combined.groupby(combined.index).last()


class Step21_ExtractAndMerge:
    """Parallel feature extraction; constructs unified lightweight NetCDF."""

    def run(self, static_df: pd.DataFrame, loader: "DataLoader") -> xr.Dataset:
        log.info("=== Step 2.1 – Extracting Features & Merging (Read & Reduce) ===")
        tasks = loader.iter_ts_paths_list()
        log.info(f"  Total stations to process: {len(tasks)}  |  Workers: {N_WORKERS}")

        station_results = {}
        all_scalar_keys = set()
        global_min_yr = 2100
        global_max_yr = 1700

        worker_args = [(fp, gid, KEY_DYNAMIC_VARS, ROLL_WINDOWS) for fp, gid in tasks]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(_worker_extract_features, arg) for arg in worker_args]
            for fut in tqdm(as_completed(futs), total=len(tasks), desc="  Extract"):
                gid, yrs, ams_vals, scalars = fut.result()
                if yrs is not None and len(yrs) > 0:
                    station_results[gid] = (yrs, ams_vals, scalars)
                    all_scalar_keys.update(scalars.keys())
                    global_min_yr = min(global_min_yr, int(yrs.min()))
                    global_max_yr = max(global_max_yr, int(yrs.max()))

        stations = sorted(station_results.keys())
        n_st = len(stations)
        years_axis = np.arange(global_min_yr, global_max_yr + 1)
        n_yr = len(years_axis)
        year_to_idx = {yr: i for i, yr in enumerate(years_axis)}
        log.info(f"  Valid stations: {n_st}  |  Year range: {global_min_yr}–{global_max_yr}")

        # ── Build AMS matrix ────────────────────────────────────────────────
        ams_matrix = np.full((n_st, n_yr), np.nan, dtype=np.float32)
        scalar_keys = sorted(all_scalar_keys)
        scalar_arrs = {k: np.full(n_st, np.nan, dtype=np.float32) for k in scalar_keys}

        for i, gid in enumerate(stations):
            yrs, ams_vals, scalars = station_results[gid]
            idx = [year_to_idx[y] for y in yrs if y in year_to_idx]
            valid_mask = [y in year_to_idx for y in yrs]
            ams_matrix[i, idx] = ams_vals[valid_mask]
            for k in scalar_keys:
                scalar_arrs[k][i] = scalars.get(k, np.nan)

        # ── Assemble xr.Dataset ─────────────────────────────────────────────
        data_vars = {
            "ann_max_streamflow": xr.DataArray(ams_matrix, dims=["station", "year"])
        }
        for k, arr in scalar_arrs.items():
            data_vars[k] = xr.DataArray(arr, dims=["station"])

        # Merge static attributes (numeric as float32, string preserved as str)
        for col in static_df.columns:
            if col.startswith("_"):
                continue
            vals = static_df.reindex(stations)[col].values
            try:
                data_vars[f"static_{col}"] = xr.DataArray(
                    vals.astype(np.float32), dims=["station"]
                )
            except (ValueError, TypeError):
                data_vars[f"static_{col}"] = xr.DataArray(
                    vals.astype(str), dims=["station"]
                )

        ds = xr.Dataset(data_vars, coords={"station": stations, "year": years_axis})
        ds.attrs["description"] = "Extracted Annual Features – Caravan + GRDC"
        ds.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

        self._save(ds, OUT_NC_MERGE)
        log.info("  Step 2.1 Complete.")
        return ds

    @staticmethod
    def _save(ds: xr.Dataset, path: Path):
        enc = {v: {"zlib": True, "complevel": 4}
               for v in ds.data_vars if ds[v].dtype.kind == "f"}
        ds.to_netcdf(path, format="NETCDF4", encoding=enc)
        log.info(f"  Saved → {path}")


class Step22_Deduplicate:
    """Remove duplicate gauges using coordinate + drainage-area criteria."""

    def run(self, ds: xr.Dataset) -> xr.Dataset:
        log.info("=== Step 2.2 – Deduplication ===")
        stations = np.array(ds.coords["station"].values)

        def _get(col):
            key = f"static_{col}"
            return (ds[key].values.astype(float)
                    if key in ds and ds[key].dtype.kind == "f"
                    else np.full(len(stations), np.nan))

        lat, lon, area = _get("gauge_lat"), _get("gauge_lon"), _get("area")

        # ── A. Exact dedup: hysets vs camels-US ─────────────────────────────
        hy_mask = np.array([s.startswith("hysets_") for s in stations])
        ca_mask = np.array([s.startswith("camels_") for s in stations])
        hy_idx = np.where(hy_mask)[0]
        ca_idx = np.where(ca_mask)[0]
        drop_exact = set()
        if len(hy_idx) and len(ca_idx):
            dlat = np.abs(lat[hy_idx][:, None] - lat[ca_idx][None, :])
            dlon = np.abs(lon[hy_idx][:, None] - lon[ca_idx][None, :])
            ar_hy = area[hy_idx][:, None]
            ar_ca = area[ca_idx][None, :]
            with np.errstate(invalid="ignore", divide="ignore"):
                dar_rel = np.abs(ar_hy - ar_ca) / np.maximum(np.abs(ar_hy), 1e-9)
            match = (dlat < 1e-4) & (dlon < 1e-4) & (dar_rel < 0.05)
            drop_exact = {stations[hy_idx[i]] for i in np.where(match.any(axis=1))[0]}

        # ── B. Fuzzy dedup: GRDC vs Caravan (5 km + 5% area) ───────────────
        keep_mask = ~np.isin(stations, list(drop_exact))
        grdc_idx = np.where(
            np.array([s.startswith("GRDC_") for s in stations]) & keep_mask)[0]
        cara_idx = np.where(
            (~np.array([s.startswith("GRDC_") for s in stations])) & keep_mask)[0]

        drop_grdc = set()
        CHUNK = 500
        for start in range(0, len(grdc_idx), CHUNK):
            chunk = grdc_idx[start: start + CHUNK]
            valid = ~(np.isnan(lat[chunk]) | np.isnan(area[chunk]) | (area[chunk] == 0))
            if not valid.any():
                continue
            cv = chunk[valid]
            dist = haversine_matrix(lat[cv], lon[cv], lat[cara_idx], lon[cara_idx])
            ar_g = area[cv][:, None]
            ar_c = area[cara_idx][None, :]
            with np.errstate(invalid="ignore", divide="ignore"):
                area_rel = np.abs(ar_g - ar_c) / np.maximum(np.abs(ar_g), 1e-9)
            match = (dist < DIST_THRESH_KM) & (area_rel < AREA_REL_THRESH)
            drop_grdc.update(stations[cv[np.where(match.any(axis=1))[0]]].tolist())
        log.info(f"  Fuzzy GRDC duplicates removed: {len(drop_grdc)}")

        keep = [s for s in stations if s not in drop_exact and s not in drop_grdc]
        ds_out = ds.sel(station=keep)

        # FIX: write counts to attrs so the report can read them correctly
        ds_out.attrs["dedup_dropped_exact"] = len(drop_exact)
        ds_out.attrs["dedup_dropped_fuzzy"] = len(drop_grdc)
        ds_out.attrs["n_stations_after_dedup"] = len(keep)

        enc = {v: {"zlib": True, "complevel": 4}
               for v in ds_out.data_vars if ds_out[v].dtype.kind == "f"}
        ds_out.to_netcdf(OUT_NC_DEDUP, format="NETCDF4", encoding=enc)
        log.info(f"  Saved → {OUT_NC_DEDUP}  ({len(keep)} stations remain)")
        log.info("  Step 2.2 Complete.")
        return ds_out


class Step23_FilterYears:
    """Filter stations by minimum valid AMS years (35)."""

    def run(self, ds: xr.Dataset):
        log.info("=== Step 2.3 – Year-based Filtering (>=35 years) ===")
        valid_counts = (~np.isnan(ds["ann_max_streamflow"].values)).sum(axis=1)
        stations = list(ds.coords["station"].values)
        yr_series = pd.Series(valid_counts, index=stations)
        log.info(f"  Year counts – min:{yr_series.min()}  "
                 f"median:{yr_series.median():.0f}  max:{yr_series.max()}")

        keep = yr_series[yr_series >= MIN_YEARS_35].index.tolist()
        log.info(f"  ≥{MIN_YEARS_35} years: {len(keep)} stations")
        ds35 = ds.sel(station=keep)

        enc = {v: {"zlib": True, "complevel": 4}
               for v in ds35.data_vars if ds35[v].dtype.kind == "f"}
        ds35.to_netcdf(OUT_NC_35, format="NETCDF4", encoding=enc)

        log.info("  Step 2.3 Complete.")
        return ds35


class Step24_FeatureEngineering:
    """
    Features are already engineered in Step 2.1 (Read & Reduce).
    This step saves the finalised 35yr feature dataset.
    OUT_NC_FEAT (alias) points to the 35yr version.
    """

    def run(self, ds35: xr.Dataset) -> xr.Dataset:
        log.info("=== Step 2.4 – Feature Engineering (Pass-through + Save) ===")

        # Save 35-year dataset
        enc35 = {v: {"zlib": True, "complevel": 4}
                 for v in ds35.data_vars if ds35[v].dtype.kind == "f"}
        ds35.to_netcdf(OUT_NC_FEAT_35, format="NETCDF4", encoding=enc35)
        log.info(f"  Saved 35yr → {OUT_NC_FEAT_35}  ({ds35.dims['station']} stations)")

        # Alias: 4_Cara-GRDC.nc = 35yr (default for downstream steps)
        ds35.to_netcdf(OUT_NC_FEAT, format="NETCDF4", encoding=enc35)
        log.info(f"  Alias saved → {OUT_NC_FEAT}")

        log.info("  Step 2.4 Complete.")
        return ds35


class Step25_CorrelationAnalysis:
    CMAP_DIV = "RdBu_r"
    FIG_DPI = 300
    FONT_SIZE = 9

    def _build_feature_matrix(self, ds: xr.Dataset) -> pd.DataFrame:
        """
        Build a (station × feature) DataFrame of scalar predictors.

        Safely skip variables that cannot be cast to float64. This prevents
        a crash when static attributes contain station names, river names, etc.
        """
        stations = list(ds.coords["station"].values)
        exclude = {
            "ann_max_streamflow", "mean_ann_max_sf",
            "static__source", "static__region",
        }
        feat = {}
        skipped_str, skipped_nan = 0, 0

        for v in ds.data_vars:
            if v in exclude or v.startswith("static__"):
                continue
            if ds[v].dims != ("station",):
                continue

            # ── try numeric conversion; skip string/mixed columns ──────
            raw = ds[v].values
            try:
                arr = raw.astype(float)
            except (ValueError, TypeError):
                skipped_str += 1
                continue

            n_valid = np.sum(~np.isnan(arr))
            if n_valid <= 10:
                skipped_nan += 1
                continue

            feat[v] = arr

        if skipped_str > 0:
            log.info(f"  Skipped {skipped_str} non-numeric variables (string columns).")
        if skipped_nan > 0:
            log.info(f"  Skipped {skipped_nan} near-empty variables (<10 valid values).")

        return pd.DataFrame(feat, index=stations)

    @staticmethod
    def _matrix_spearman(X: np.ndarray, y: np.ndarray):
        """Vectorised Spearman correlation of each column in X against y."""
        n, p = X.shape
        Xr = np.full_like(X, np.nan)
        for j in range(p):
            col = X[:, j]
            valid = ~np.isnan(col)
            if valid.sum() > 2:
                Xr[valid, j] = rankdata(col[valid])

        y_valid = ~np.isnan(y)
        yr_full = np.full(n, np.nan)
        yr_full[y_valid] = rankdata(y[y_valid])

        rhos, pvals = np.full(p, np.nan), np.full(p, np.nan)
        for j in range(p):
            mask = ~(np.isnan(Xr[:, j]) | np.isnan(yr_full))
            nj = mask.sum()
            if nj < 4:
                continue
            r = np.corrcoef(Xr[mask, j], yr_full[mask])[0, 1]
            rhos[j] = r
            t_stat = r * np.sqrt((nj - 2) / max(1 - r ** 2, 1e-12))
            pvals[j] = 2 * scipy_t.sf(abs(t_stat), df=nj - 2)
        return rhos, pvals

    def _classify_features(self, feature_names):
        cats = {
            "Precipitation": [],
            "Temperature": [],
            "Evapotranspiration": [],
            "Snow / Soil": [],
            "Radiation": [],
            "Static (catchment)": [],
            "Other dynamic": [],
        }
        for f in feature_names:
            fl = f.lower()
            if "prec" in fl or "precipitation" in fl:
                cats["Precipitation"].append(f)
            elif "temp" in fl or "temperature" in fl:
                cats["Temperature"].append(f)
            elif "evaporation" in fl or "evap" in fl:
                cats["Evapotranspiration"].append(f)
            elif "snow" in fl or "soil_water" in fl:
                cats["Snow / Soil"].append(f)
            elif "radiation" in fl:
                cats["Radiation"].append(f)
            elif f.startswith("static_"):
                cats["Static (catchment)"].append(f)
            else:
                cats["Other dynamic"].append(f)
        return {k: v for k, v in cats.items() if v}

    def run(self, ds_feat: xr.Dataset):
        log.info("=== Step 2.5 – Correlation Analysis (Spearman) ===")
        feature_df = self._build_feature_matrix(ds_feat)
        target = ds_feat["mean_ann_max_sf"].values.astype(float)
        target_s = pd.Series(target, index=list(ds_feat.coords["station"].values))

        common_idx = feature_df.index[~np.isnan(target_s.reindex(feature_df.index))]
        feature_df = feature_df.loc[common_idx]
        target_al = target_s.reindex(common_idx).values
        log.info(f"  Feature matrix: {feature_df.shape[0]} stations × {feature_df.shape[1]} features")

        rhos, pvals = self._matrix_spearman(feature_df.values, target_al)

        corr_df = pd.DataFrame({
            "rho": rhos,
            "pval": pvals,
            "n": [(~np.isnan(feature_df.values[:, j]) & ~np.isnan(target_al)).sum()
                  for j in range(feature_df.shape[1])],
        }, index=feature_df.columns).dropna(subset=["rho"]).sort_values("rho", ascending=False)

        corr_df.to_csv(S1_FIGURE_DIR / "correlation_table.csv")
        log.info(f"  Correlation table → {S1_FIGURE_DIR / 'correlation_table.csv'}")

        cats = self._classify_features(list(corr_df.index))
        self._plot_overview(corr_df)
        self._plot_category_subplots(corr_df, cats)

        top_feats = (corr_df.head(10).index.tolist() + corr_df.tail(10).index.tolist())
        top_feats = [f for f in top_feats if f in feature_df.columns]
        self._plot_scatter_top(feature_df[top_feats], target_al, top_feats, corr_df)

        log.info("  Step 2.5 Complete.")
        return corr_df

    # ── Plotting helpers ─────────────────────────────────────────────────────

    def _setup_style(self):
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": self.FONT_SIZE,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        })

    def _plot_overview(self, corr_df: pd.DataFrame):
        self._setup_style()
        n = len(corr_df)
        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.22)), dpi=self.FIG_DPI)
        colors = ["#C0392B" if r > 0 else "#2980B9" for r in corr_df["rho"]]
        ax.barh(range(n), corr_df["rho"].values, color=colors, edgecolor="none", height=0.7)
        for i, (_, row) in enumerate(corr_df.iterrows()):
            m = ("***" if row["pval"] < 0.001 else
                 "**" if row["pval"] < 0.01 else
                 "*" if row["pval"] < 0.05 else "")
            if m:
                xpos = row["rho"] + (0.01 if row["rho"] >= 0 else -0.01)
                ha = "left" if row["rho"] >= 0 else "right"
                ax.text(xpos, i, m, va="center", ha=ha, fontsize=7, color="#333333")
        ax.set_yticks(range(n))
        ax.set_yticklabels(corr_df.index, fontsize=6.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Spearman Correlation Coefficient (ρ)")
        ax.set_title("Feature Correlations with Mean Annual Maximum Streamflow",
                     fontweight="bold", pad=10)
        ax.set_xlim(-1.05, 1.05)
        ax.legend(
            handles=[Patch(facecolor="#C0392B", label="Positive"),
                     Patch(facecolor="#2980B9", label="Negative")],
            loc="lower right", fontsize=8, framealpha=0.8,
        )
        fig.tight_layout()
        out = S1_FIGURE_DIR / "Fig1_Overview_Correlation.png"
        fig.savefig(out, dpi=self.FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Fig1 → {out}")

    def _plot_category_subplots(self, corr_df: pd.DataFrame, cats: dict):
        self._setup_style()
        n_cats = len(cats)
        ncols = 2
        nrows = int(np.ceil(n_cats / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(14, max(4, nrows * 3.5)),
                                 dpi=self.FIG_DPI)
        axes = np.array(axes).flatten()
        for ai, (cat_name, feats) in enumerate(cats.items()):
            ax = axes[ai]
            sub = corr_df.loc[[f for f in feats if f in corr_df.index]]
            if sub.empty:
                ax.set_visible(False)
                continue
            sub = sub.sort_values("rho")
            ax.barh(range(len(sub)), sub["rho"].values,
                    color=["#C0392B" if r > 0 else "#2980B9" for r in sub["rho"]],
                    height=0.7)
            ax.set_yticks(range(len(sub)))
            ax.set_yticklabels(
                [f.replace("static_", "").replace("ltmean_", "").replace("_", " ")[:40]
                 for f in sub.index],
                fontsize=6,
            )
            ax.axvline(0, color="black", linewidth=0.7)
            ax.set_title(cat_name, fontweight="bold")
            ax.set_xlabel("ρ")
            ax.set_xlim(-1.05, 1.05)
        for aj in range(n_cats, len(axes)):
            axes[aj].set_visible(False)
        fig.suptitle("Category-wise Spearman Correlations with Mean Annual Max Streamflow",
                     fontsize=12, fontweight="bold", y=1.01)
        fig.tight_layout()
        out = S1_FIGURE_DIR / "Fig2_Category_Correlation.png"
        fig.savefig(out, dpi=self.FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Fig2 → {out}")

    def _plot_scatter_top(self, feat_df, target, top_feats, corr_df):
        self._setup_style()
        n = len(top_feats)
        ncols = 5
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.2, nrows * 3.0),
                                 dpi=self.FIG_DPI)
        axes = np.array(axes).flatten()
        for i, feat in enumerate(top_feats):
            ax = axes[i]
            x = feat_df[feat].values.astype(float)
            mask = ~(np.isnan(x) | np.isnan(target))
            rho = corr_df.loc[feat, "rho"]
            pval = corr_df.loc[feat, "pval"]
            color = "#C0392B" if rho > 0 else "#2980B9"
            ax.scatter(x[mask], target[mask], s=3, alpha=0.4, color=color, linewidths=0)
            if mask.sum() > 5:
                z = np.polyfit(x[mask], target[mask], 1)
                xr_a = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                ax.plot(xr_a, np.poly1d(z)(xr_a), color=color, linewidth=1.2, alpha=0.9)
            ax.set_xlabel(feat.replace("_", " ")[:30], fontsize=6.5)
            ax.set_ylabel("Mean AMS (m³/s · area⁻¹)", fontsize=6.5)
            star = ("***" if pval < 0.001 else
                    "**" if pval < 0.01 else
                    "*" if pval < 0.05 else "n.s.")
            ax.set_title(f"ρ={rho:.2f} {star}", fontsize=8)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Scatter Plots: Top Correlated Features vs. Mean Annual Max Streamflow",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        out = S1_FIGURE_DIR / "Fig3_Scatter_Top_Features.png"
        fig.savefig(out, dpi=self.FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Fig3 → {out}")


class Step26_Report:
    def run(self, stats: dict):
        log.info("=== Step 2.6 – Generating Report ===")
        lines = [
            "# Caravan + GRDC Data Pre-processing Report",
            f"\n**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
            "## 1. Data Sources",
            f"- **Caravan:** `{CARAVAN_ROOT}`",
            f"- **GRDC:**    `{GRDC_ROOT}`", "",
            "## 2. Processing Summary",
            f"| Step | Description | Count |",
            f"|------|-------------|-------|",
            f"| 2.1  | Total stations extracted     | **{stats.get('n_merged', '–')}** |",
            f"| 2.2  | Exact duplicates removed     | **{stats.get('dedup_exact', '–')}** |",
            f"| 2.2  | Fuzzy GRDC duplicates removed| **{stats.get('dedup_fuzzy', '–')}** |",
            f"| 2.3  | Stations ≥35 years           | **{stats.get('n_35yr', '–')}** |", "",
            "## 3. Feature Engineering",
            f"- **AMS:** annual maximum streamflow per station (year-indexed matrix)",
            f"- **Rolling precipitation windows:** {ROLL_WINDOWS} days – annual max mean",
            f"- **Rolling temperature windows:**   {ROLL_WINDOWS} days – annual mean & max",
            f"- **Long-term scalar means:**  all dynamic variables",
            f"- **Static catchment attributes:** from Caravan / GRDC attribute files", "",
            "## 4. Output Files",
            f"| File | Description |",
            f"|------|-------------|",
            f"| `1_Cara-GRDC-Merge.nc`   | All stations merged               |",
            f"| `2_Cara-GRDC-rem-Dup.nc` | After deduplication               |",
            f"| `3_Cara-GRDC-35.nc`      | ≥35 yr AMS records                |",
            f"| `4_Cara-GRDC-35.nc`      | Final feature dataset (35yr)      |",
            f"| `4_Cara-GRDC.nc`         | Alias of 35yr (default)           |", "",
        ]

        # Top correlation results
        if stats.get("corr_df") is not None:
            corr_df = stats["corr_df"]
            lines += ["## 5. Top Feature Correlations with Mean Annual Max Streamflow", ""]
            for title, sub in [("Top 10 Positive", corr_df.head(10)),
                                ("Top 10 Negative", corr_df.tail(10))]:
                lines += [f"### {title}",
                          "| Feature | ρ | p-value | n |",
                          "|---------|---|---------|---|"]
                for idx, row in sub.iterrows():
                    lines.append(
                        f"| {idx.replace('_',' ')} | {row['rho']:.3f} "
                        f"| {row['pval']:.3e} | {int(row['n'])} |"
                    )
                lines.append("")

        S1_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        log.info(f"  Report → {S1_REPORT_PATH}")


def run_stage1():
    """Entry point for Stage 1 (was 00_GRDC-Caravan-Process.py::main)."""
    t0 = time.time()
    stats = {}

    log.info(f"Data output root: {S1_DATA_DIR}")
    log.info(f"Figure output root: {S1_FIGURE_DIR}")
    log.info("=" * 65)
    log.info(f"  Caravan+GRDC Pipeline  |  Workers={N_WORKERS}")
    log.info("=" * 65)

    loader = DataLoader()

    # 2.1 Extract & Merge
    ds_merge = Step21_ExtractAndMerge().run(loader.load_static(), loader)
    stats["n_merged"] = int(ds_merge.dims["station"])

    # 2.2 Deduplicate
    ds_dedup = Step22_Deduplicate().run(ds_merge)
    stats["dedup_exact"] = ds_dedup.attrs.get("dedup_dropped_exact", 0)
    stats["dedup_fuzzy"] = ds_dedup.attrs.get("dedup_dropped_fuzzy", 0)

    # 2.3 Filter years
    ds35 = Step23_FilterYears().run(ds_dedup)
    stats["n_35yr"] = int(ds35.dims["station"])

    # 2.4 Feature Engineering (save 35yr only)
    ds_feat = Step24_FeatureEngineering().run(ds35)

    # 2.5 Correlation analysis
    stats["corr_df"] = Step25_CorrelationAnalysis().run(ds_feat)

    # 2.6 Report
    Step26_Report().run(stats)

    elapsed = time.time() - t0
    log.info("=" * 65)
    log.info(f"  Stage 1 complete.  Total time: {elapsed / 60:.2f} min")
    log.info("=" * 65)

    return {"nc_feat_35": OUT_NC_FEAT_35, "nc_feat_alias": OUT_NC_FEAT, "stats": stats}


# ================================================================================
# Stage 2: At-site GEV fitting (delegates to src/gev_fit_common.py)
# (was 01_GEV-Fit.py)
# ================================================================================

# ---- Stage 2 configuration (was configs/base.yaml: gev_fit.*) ----
S2_OUT_CSV_DIR = stage_dir(DATA_PROCEED, "01_GEV-Fit")
S2_OUT_FIG_DIR = stage_dir(FIGURE_ROOT, "01_GEV-Fit")


def run_stage2():
    """Entry point for Stage 2 (was 01_GEV-Fit.py::main)."""
    cfg = GEVFitConfig(
        nc_file=OUT_NC_FEAT_35,  # Stage 1's 4_Cara-GRDC-35.nc
        out_csv=S2_OUT_CSV_DIR,
        out_fig=S2_OUT_FIG_DIR,
        return_periods=[2, 5, 10, 20, 50, 100],
        min_years=5,
        n_sample=24,
        seed=42,
        xi_clip_min=-0.8,
        xi_clip_max=0.5,
    )
    configure_plot_style()

    t0 = time.time()
    log.info("=" * 60)
    log.info("  Stage 2  -  GEV MLE fitting for all stations")
    log.info("=" * 60)

    df = fit_all_stations(cfg, log)

    csv_path = cfg.out_csv / cfg.csv_name
    df.to_csv(csv_path, index=False)
    log.info(f"  Parameters saved -> {csv_path}")

    plot_freq_curves(df, cfg, log)
    plot_param_distributions(df, cfg, log)
    plot_ks_distribution(df, cfg, log)

    elapsed = time.time() - t0
    log.info(f"  Stage 2 done in {elapsed:.1f} s  |  Figures -> {cfg.out_fig}")

    return {"gev_csv": csv_path}


# ================================================================================
# Stage 3: Quality-filter stations by GEV shape parameter and KS test
# (was 02_Data-Clean.py)
# ================================================================================

# ---- Stage 3 configuration (was configs/base.yaml: data_clean.*) ----
# NOTE: explicitly point at Stage 1's actual output location -- the original
# 02_Data-Clean.py had an ambiguous fallback default that omitted the
# "00_GRDC-Caravan-Process" subfolder; here the location is known exactly.
S3_NC_PATH = S1_DATA_DIR / "4_Cara-GRDC-35.nc"
S3_GEV_CSV = S2_OUT_CSV_DIR / "gev_station_params.csv"

S3_OUT_DATA = stage_dir(DATA_PROCEED, "02_Data-Clean")
S3_OUT_FIG = stage_dir(FIGURE_ROOT, "02_Data-Clean")

# --- Filtering thresholds ---
XI_MIN = -0.5
XI_MAX = 0.5
KS_P_MIN = 0.05  # Criterion 2: remove KS p-value < this value

LAT_VAR = "static_gauge_lat"
LON_VAR = "static_gauge_lon"

RETURN_PERIODS = [2, 5, 10, 20, 50, 100]

FILTER_COLORS = {
    "raw": "#90A4AE",
    "after_xi": "#42A5F5",
    "after_ks": "#1B5E20",
    "removed_xi": "#EF5350",
    "removed_ks": "#FF8F00",
}


# ---- Data loading ----
def load_data():
    log.info("Loading GEV CSV ...")
    gev = pd.read_csv(S3_GEV_CSV)
    log.info(f"  GEV CSV: {len(gev)} stations, columns: {gev.columns.tolist()}")

    log.info("Loading lat/lon from NC ...")
    ds = xr.open_dataset(S3_NC_PATH)
    stations_nc = ds.coords["station"].values.astype(str)
    lat_all = ds[LAT_VAR].values
    lon_all = ds[LON_VAR].values
    ds.close()

    nc_map = {s: (lat_all[i], lon_all[i]) for i, s in enumerate(stations_nc)}
    gev["lat"] = gev["station_id"].map(lambda s: nc_map.get(s, (np.nan, np.nan))[0])
    gev["lon"] = gev["station_id"].map(lambda s: nc_map.get(s, (np.nan, np.nan))[1])

    missing_ll = gev["lat"].isna().sum()
    if missing_ll:
        log.warning(f"  {missing_ll} stations missing lat/lon — retained but unmapped")

    log.info(f"  Lat valid: {gev['lat'].notna().sum()}  "
             f"xi range: [{gev['xi'].min():.3f}, {gev['xi'].max():.3f}]  "
             f"KS p range: [{gev['ks_pvalue'].min():.4g}, {gev['ks_pvalue'].max():.4f}]")
    return gev


# ---- Filtering ----
def apply_filters(gev_raw: pd.DataFrame):
    log.info("=" * 60)
    log.info("Applying quality filters")

    n0 = len(gev_raw)
    log.info(f"  Start          : {n0:6d} stations")

    # ---------- Criterion 1: ξ outside [XI_MIN, XI_MAX] ----------
    mask_xi_bad = (gev_raw["xi"] < XI_MIN) | (gev_raw["xi"] > XI_MAX)
    gev_after_xi = gev_raw[~mask_xi_bad].copy()
    n1 = len(gev_after_xi)
    removed_xi = n0 - n1
    log.info(f"  After {XI_MIN}≤ξ≤{XI_MAX}  : {n1:6d} stations  "
             f"({removed_xi} removed, {removed_xi/n0*100:.1f}%)")

    # ---------- Criterion 2: KS p-value < KS_P_MIN ----------
    mask_ks_bad = gev_after_xi["ks_pvalue"] < KS_P_MIN
    gev_clean = gev_after_xi[~mask_ks_bad].copy()
    n2 = len(gev_clean)
    removed_ks = n1 - n2
    log.info(f"  After KS p≥{KS_P_MIN}  : {n2:6d} stations  "
             f"({removed_ks} removed, {removed_ks/n1*100:.1f}%)")
    log.info(f"  Total removed  : {n0-n2:6d} stations  "
             f"({(n0-n2)/n0*100:.1f}%)")
    log.info("=" * 60)

    # Build full mask log for each raw station
    reason = pd.Series("kept", index=gev_raw.index)
    # xi filter applied first, then ks on remainder
    reason[mask_xi_bad] = "xi_extreme"
    # stations that passed xi but failed ks
    idx_ks_bad = gev_after_xi[mask_ks_bad].index
    reason[idx_ks_bad] = "ks_fail"

    return gev_raw, gev_after_xi, gev_clean, reason, dict(
        n_raw=n0, removed_xi=removed_xi, removed_ks=removed_ks,
        n_final=n2,
        pct_xi=removed_xi / n0 * 100,
        pct_ks=removed_ks / n1 * 100,
        pct_total=(n0 - n2) / n0 * 100,
    )


# ---- Statistics ----
def compute_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
    cols = ["mu", "sigma", "xi", "ks_pvalue", "n_years"]
    cols = [c for c in cols if c in df.columns]
    stats = df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats.insert(0, "dataset", label)
    return stats


def print_stats(gev_raw, gev_clean, counts):
    log.info("\n--- Descriptive statistics (raw vs cleaned) ---")
    for col in ["mu", "sigma", "xi", "ks_pvalue"]:
        r = gev_raw[col]
        c = gev_clean[col]
        log.info(f"  {col:<12}  raw  mean={r.mean():.4g}  "
                 f"std={r.std():.4g}  [p5={r.quantile(.05):.4g}, "
                 f"p95={r.quantile(.95):.4g}]")
        log.info(f"  {col:<12}  clean mean={c.mean():.4g}  "
                 f"std={c.std():.4g}  [p5={c.quantile(.05):.4g}, "
                 f"p95={c.quantile(.95):.4g}]")


# ---- Figures (save one-by-one for clarity) ----
def _save(fig, fname, msg=""):
    fig.savefig(S3_OUT_FIG / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  {fname} saved  {msg}")


# ---- Figure 01: filter funnel bar chart ----
def fig_filter_funnel(counts):
    labels = [
        "Raw\n(all)",
        f"After {XI_MIN}≤ξ≤{XI_MAX}\n(C1 pass)",
        "After KS p≥0.05\n(C2 pass = final)",
    ]
    values = [counts["n_raw"],
              counts["n_raw"] - counts["removed_xi"],
              counts["n_final"]]
    colors = [FILTER_COLORS["raw"],
              FILTER_COLORS["after_xi"],
              FILTER_COLORS["after_ks"]]
    removed = [0, counts["removed_xi"], counts["removed_ks"]]
    removed_lbl = ["", f"−{counts['removed_xi']:,}\n({counts['pct_xi']:.1f}%)",
                   f"−{counts['removed_ks']:,}\n({counts['pct_ks']:.1f}%)"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", lw=1.5)
    for bar, v, rl in zip(bars, values, removed_lbl):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 60, f"{v:,}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
        if rl:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2, rl,
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
    ax.set_ylabel("Number of stations", fontsize=12)
    ax.set_title("Station Filtering Funnel", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "01_filter_funnel.png",
          f"({counts['n_raw']}→{counts['n_final']} stations)")


# ---- Figure 02: ξ distribution  ----
def fig_xi_distribution(gev_raw, gev_after_xi, gev_clean):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    x_left = min(-0.6, float(gev_raw["xi"].min()), XI_MIN - 0.05)
    x_right = max(0.6, float(gev_raw["xi"].max()), XI_MAX + 0.05)
    bins = np.linspace(x_left, x_right, 56)
    kw = dict(edgecolor="white", linewidth=0.4)

    for ax, df, title, color in [
        (axes[0], gev_raw, f"Raw  (n={len(gev_raw):,})", FILTER_COLORS["raw"]),
        (axes[1], gev_after_xi, f"After {XI_MIN}≤ξ≤{XI_MAX}  (n={len(gev_after_xi):,})", FILTER_COLORS["after_xi"]),
        (axes[2], gev_clean, f"Final cleaned  (n={len(gev_clean):,})", FILTER_COLORS["after_ks"]),
    ]:
        ax.hist(df["xi"], bins=bins, color=color, **kw)
        ax.axvline(XI_MIN, color="#EF5350", lw=1.5, ls="--", label=f"ξ={XI_MIN}")
        ax.axvline(XI_MAX, color="#EF5350", lw=1.5, ls="--", label=f"ξ={XI_MAX}")
        ax.set_xlabel("Shape parameter ξ", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.legend(fontsize=8)
        mu_xi = df["xi"].mean()
        ax.text(0.05, 0.97, f"mean={mu_xi:.3f}\nstd={df['xi'].std():.3f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", alpha=0.7, ec="none"))

    fig.suptitle("Shape Parameter ξ Distribution at Each Filtering Step",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_xi_distribution.png")


# ---- Figure 03: KS p-value histogram + CDF ----
def fig_pvalue_distribution(gev_raw, gev_clean):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: raw p-value histogram
    ax = axes[0]
    pv = gev_raw["ks_pvalue"]
    ax.hist(pv, bins=40, color=FILTER_COLORS["raw"], edgecolor="white", lw=0.4)
    ax.axvline(KS_P_MIN, color="#EF5350", lw=2, ls="--",
               label=f"threshold={KS_P_MIN}")
    ax.set_xlabel("KS p-value", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Raw KS p-value  (n={len(gev_raw):,})", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    n_fail = (pv < KS_P_MIN).sum()
    ax.text(0.97, 0.97,
            f"p<{KS_P_MIN}: {n_fail} ({n_fail/len(pv)*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#EF5350", bbox=dict(fc="white", alpha=0.7, ec="none"))

    # Panel 2: log-scale p-value histogram (highlights the failing tail)
    ax = axes[1]
    log_pv = -np.log10(pv.clip(lower=1e-300))
    thresh_log = -np.log10(KS_P_MIN)
    ax.hist(log_pv, bins=50, color=FILTER_COLORS["raw"], edgecolor="white", lw=0.4)
    ax.axvline(thresh_log, color="#EF5350", lw=2, ls="--",
               label=f"−log₁₀({KS_P_MIN})={thresh_log:.1f}")
    ax.set_xlabel("−log₁₀(KS p-value)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("KS statistic (log scale)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 3: ECDF of cleaned p-values
    ax = axes[2]
    pv_clean = np.sort(gev_clean["ks_pvalue"].values)
    ecdf = np.arange(1, len(pv_clean) + 1) / len(pv_clean)
    ax.plot(pv_clean, ecdf, color=FILTER_COLORS["after_ks"], lw=1.8)
    ax.axvline(KS_P_MIN, color="#EF5350", lw=1.5, ls="--",
               label=f"threshold={KS_P_MIN}")
    ax.set_xlabel("KS p-value", fontsize=10)
    ax.set_ylabel("ECDF", fontsize=10)
    ax.set_title(f"ECDF of KS p-value  (cleaned, n={len(gev_clean):,})",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("KS Goodness-of-Fit p-value Analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_pvalue_distribution.png")


# ---- Figure 04: parameter histograms before vs after ----
def fig_param_histograms(gev_raw, gev_clean):
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    pairs = [
        ("mu", "Location μ", np.log10, "log₁₀(μ)", None, None),
        ("sigma", "Scale σ", np.log10, "log₁₀(σ)", None, None),
        ("xi", "Shape ξ", None, "ξ", XI_MIN, XI_MAX),
    ]

    for row, (col, colname, xform, xlabel, xlo, xhi) in enumerate(pairs):
        for panel, (df, lbl, color) in enumerate([
            (gev_raw, f"Raw  (n={len(gev_raw):,})", FILTER_COLORS["raw"]),
            (gev_clean, f"Cleaned  (n={len(gev_clean):,})", FILTER_COLORS["after_ks"]),
        ]):
            ax = axes[row, panel]
            vals = df[col].copy()
            if xform is not None:
                vals = xform(vals.clip(lower=1e-9))
            ax.hist(vals, bins=50, color=color, edgecolor="white", lw=0.3,
                    density=True, alpha=0.85)
            if xlo is not None:
                ax.axvline(xlo, color="#EF5350", lw=1.2, ls="--")
                ax.axvline(xhi, color="#EF5350", lw=1.2, ls="--")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title(f"{colname}  |  {lbl}", fontsize=9, fontweight="bold")
            ax.text(0.98, 0.97,
                    f"mean={df[col].mean():.3g}\nstd={df[col].std():.3g}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(fc="white", alpha=0.7, ec="none"))
            ax.grid(True, alpha=0.25)

    fig.suptitle("GEV Parameter Distributions  —  Raw vs Cleaned",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_param_histograms.png")


# ---- Figure 05/06/07: world scatter maps ----
def _world_scatter(df, val_col, cmap, vmin, vmax, title, threshold_lines=None,
                   cbar_label=""):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_facecolor("#EAF2F8")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)

    # simple coastline reference  (just axes, no geopandas needed)
    ax.axhline(0, color="#B0BEC5", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0, color="#B0BEC5", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)

    sc = ax.scatter(df["lon"], df["lat"], c=df[val_col],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    s=12, alpha=0.75, linewidths=0, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label(cbar_label, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    return fig


def fig_world_maps(gev_raw, gev_clean):
    # Fig 05: raw stations, colour = xi
    fig = _world_scatter(
        gev_raw.dropna(subset=["lat"]), "xi", "RdBu_r",
        XI_MIN, XI_MAX,
        f"All raw stations coloured by ξ  (n={len(gev_raw):,})",
        cbar_label="Shape ξ",
    )
    _save(fig, "05_world_map_all.png")

    # Fig 06: cleaned stations, colour = xi
    fig = _world_scatter(
        gev_clean.dropna(subset=["lat"]), "xi", "RdBu_r",
        XI_MIN, XI_MAX,
        f"Cleaned stations coloured by ξ  (n={len(gev_clean):,})",
        cbar_label="Shape ξ",
    )
    _save(fig, "06_world_map_cleaned.png")

    # Fig 07: raw, colour = -log10(ks_pvalue)
    gev_raw2 = gev_raw.copy()
    gev_raw2["neg_log_p"] = -np.log10(gev_raw2["ks_pvalue"].clip(lower=1e-300))
    fig = _world_scatter(
        gev_raw2.dropna(subset=["lat"]), "neg_log_p", "hot_r",
        0, 5,
        f"Raw stations coloured by −log₁₀(KS p-value)  (n={len(gev_raw):,})",
        cbar_label="−log₁₀(p)",
    )
    _save(fig, "07_world_map_pvalue.png")


# ---- Figure 08: ξ vs –log10(p) scatter with thresholds ----
def fig_xi_pvalue_scatter(gev_raw, reason):
    fig, ax = plt.subplots(figsize=(10, 7))

    xi_vals = gev_raw["xi"].values
    neg_logp = -np.log10(gev_raw["ks_pvalue"].clip(lower=1e-300).values)

    color_map = {
        "kept": ("#1B5E20", "Kept", 12, 0.5),
        "xi_extreme": ("#EF5350", f"ξ<{XI_MIN} or ξ>{XI_MAX}", 8, 0.6),
        "ks_fail": ("#FF8F00", "KS fail", 8, 0.6),
    }
    for cat, (c, lbl, s, alpha) in color_map.items():
        mask = (reason == cat).values
        ax.scatter(xi_vals[mask], neg_logp[mask],
                   s=s, c=c, alpha=alpha, linewidths=0, label=lbl, zorder=2)

    # Threshold lines
    ax.axvline(XI_MIN, color="#EF5350", lw=1.5, ls="--",
               label=f"ξ threshold: [{XI_MIN}, {XI_MAX}]")
    ax.axvline(XI_MAX, color="#EF5350", lw=1.5, ls="--", label="_nolegend_")
    ax.axhline(-np.log10(KS_P_MIN), color="#FF8F00", lw=1.5, ls="--",
               label=f"−log₁₀({KS_P_MIN}) = {-np.log10(KS_P_MIN):.1f}")

    ax.set_xlabel("Shape ξ", fontsize=12)
    ax.set_ylabel("−log₁₀(KS p-value)", fontsize=12)
    ax.set_title("ξ vs KS Rejection Severity  —  Two-Criterion View",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # Annotate quadrants
    y_max = neg_logp[np.isfinite(neg_logp)].max() * 0.95
    thresh_log = -np.log10(KS_P_MIN)
    ax.text(XI_MIN + 0.01, thresh_log + 0.15, "KS fail region",
            fontsize=8, color="#FF8F00", alpha=0.9)
    ax.text(XI_MAX + 0.005, y_max * 0.5, "ξ extreme\nregion",
            fontsize=8, color="#EF5350", rotation=90, va="center")

    plt.tight_layout()
    _save(fig, "08_xi_pvalue_scatter.png")


# ---- Figure 09: n_years distribution comparison ----
def fig_nyears(gev_raw, gev_clean):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(5, gev_raw["n_years"].max() + 5, 5)
    ax.hist(gev_raw["n_years"], bins=bins, alpha=0.6,
            label=f"Raw ({len(gev_raw):,})",
            color=FILTER_COLORS["raw"], edgecolor="white", lw=0.4)
    ax.hist(gev_clean["n_years"], bins=bins, alpha=0.8,
            label=f"Cleaned ({len(gev_clean):,})",
            color=FILTER_COLORS["after_ks"], edgecolor="white", lw=0.4)
    ax.set_xlabel("Record length (years)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Record Length Distribution  —  Raw vs Cleaned",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "09_nyears_distribution.png")


# ---- Figure 10: quantile ratio Q100/Q2 (extreme extrapolation spread) ----
def fig_q_ratio(gev_raw, gev_clean):
    """Q100/Q2 ratio captures how heavily-tailed the fit is.
       Unreliable stations often show Q100/Q2 >> 10."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, lbl, color in [
        (axes[0], gev_raw, f"Raw  (n={len(gev_raw):,})", FILTER_COLORS["raw"]),
        (axes[1], gev_clean, f"Cleaned  (n={len(gev_clean):,})", FILTER_COLORS["after_ks"]),
    ]:
        if "Q100" in df.columns and "Q2" in df.columns:
            ratio = (df["Q100"] / df["Q2"].clip(lower=1e-9)).clip(upper=50)
            ax.hist(ratio, bins=60, color=color, edgecolor="white", lw=0.3)
            ax.set_xlabel("Q100 / Q2  (extrapolation ratio)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"Extrapolation Ratio  |  {lbl}",
                         fontsize=10, fontweight="bold")
            med = ratio.median()
            ax.axvline(med, color="black", lw=1.5, ls="--",
                       label=f"median={med:.2f}")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle("Q100/Q2 Extrapolation Ratio  —  Raw vs Cleaned",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "10_q_ratio.png")


# ---- Save data outputs ----
def save_outputs(gev_raw, gev_clean, counts, reason):
    log.info("\nSaving outputs ...")

    # Cleaned GEV CSV
    gev_clean.to_csv(S3_OUT_DATA / "gev_cleaned.csv", index=False)
    log.info(f"  gev_cleaned.csv  ({len(gev_clean)} stations)")

    # Filter summary table
    summary_rows = [
        {"step": "raw", "n_stations": counts["n_raw"],
         "removed": 0, "pct_removed": 0.0, "criterion": "—"},
        {"step": "after_xi", "n_stations": counts["n_raw"] - counts["removed_xi"],
         "removed": counts["removed_xi"],
         "pct_removed": counts["pct_xi"],
         "criterion": f"xi < {XI_MIN} or xi > {XI_MAX}"},
        {"step": "final", "n_stations": counts["n_final"],
         "removed": counts["removed_ks"],
         "pct_removed": counts["pct_ks"],
         "criterion": f"KS p-value < {KS_P_MIN}"},
    ]
    pd.DataFrame(summary_rows).to_csv(
        S3_OUT_DATA / "filter_summary.csv", index=False)
    log.info("  filter_summary.csv")

    # Descriptive stats before/after
    stats_before = compute_stats(gev_raw, "raw")
    stats_after = compute_stats(gev_clean, "cleaned")
    stats_before.to_csv(S3_OUT_DATA / "stats_before.csv")
    stats_after.to_csv(S3_OUT_DATA / "stats_after.csv")
    log.info("  stats_before.csv  stats_after.csv")

    # Removal reason tag for each raw station
    reason_df = gev_raw[["station_id"]].copy()
    reason_df["removal_reason"] = reason.values
    reason_df.to_csv(S3_OUT_DATA / "removal_reasons.csv", index=False)
    log.info("  removal_reasons.csv")


# ---- Filter NC file ----
def filter_nc(gev_clean: pd.DataFrame) -> Path:
    """Subset the NC file to only the quality-passed stations.

    Saves  S3_OUT_DATA / '4_Cara-GRDC-35_cleaned.nc'
    xarray isel() handles both 1-D (station,) static features and
    2-D (station, year) AMS data simultaneously.
    """
    log.info("Filtering NC file to quality-passed stations ...")
    kept_ids = set(gev_clean["station_id"].astype(str))

    ds = xr.open_dataset(S3_NC_PATH)
    stations_nc = ds.coords["station"].values.astype(str)
    n_orig = len(stations_nc)

    keep_idx = [i for i, s in enumerate(stations_nc) if s in kept_ids]
    n_kept = len(keep_idx)
    log.info(f"  NC stations: {n_orig:,} original  →  {n_kept:,} kept  "
             f"({n_orig - n_kept:,} removed)")

    ds_filtered = ds.isel(station=keep_idx)
    out_nc = S3_OUT_DATA / "4_Cara-GRDC-35_cleaned.nc"
    ds_filtered.to_netcdf(out_nc)
    ds.close()

    size_mb = out_nc.stat().st_size / 1e6
    log.info(f"  Saved → {out_nc}  ({size_mb:.1f} MB)")
    return out_nc


# ---- Report ----
def print_report(gev_raw, gev_clean, counts):
    sep = "=" * 65
    lines = [
        sep,
        "  Stage 3  —  GEV Station Quality Filtering Report",
        sep,
        f"  Input  : {counts['n_raw']:6,} stations",
        "",
        f"  Criterion 1  ξ < {XI_MIN} or ξ > {XI_MAX}",
        f"    Removed  : {counts['removed_xi']:6,}  ({counts['pct_xi']:.1f}%)",
        f"    Remaining: {counts['n_raw'] - counts['removed_xi']:6,}",
        "",
        "  Criterion 2  KS p-value < 0.05",
        f"    Removed  : {counts['removed_ks']:6,}  ({counts['pct_ks']:.1f}%)",
        f"    Remaining: {counts['n_final']:6,}",
        "",
        f"  Total removed: {counts['n_raw'] - counts['n_final']:6,}  "
        f"({counts['pct_total']:.1f}%)",
        f"  Final kept   : {counts['n_final']:6,}  "
        f"({100 - counts['pct_total']:.1f}%)",
        "",
        "  Cleaned dataset statistics",
        "-" * 65,
    ]
    for col, label in [("xi", "Shape ξ"),
                        ("mu", "Location μ"),
                        ("sigma", "Scale σ"),
                        ("ks_pvalue", "KS p-value"),
                        ("n_years", "Record length (yr)")]:
        if col not in gev_clean.columns:
            continue
        c = gev_clean[col]
        lines.append(f"  {label:<20}  mean={c.mean():.4g}  "
                     f"std={c.std():.4g}  "
                     f"[p5={c.quantile(.05):.4g}, "
                     f"p95={c.quantile(.95):.4g}]")
    lines.append(sep)
    text = "\n".join(lines)
    print("\n" + text)
    (S3_OUT_DATA / "report.txt").write_text(text, encoding="utf-8")
    log.info("  report.txt saved")


def run_stage3():
    """Entry point for Stage 3 (was 02_Data-Clean.py::main)."""
    t0 = time.time()

    # 1. Load
    gev_raw = load_data()

    # 2. Filter
    gev_raw, gev_after_xi, gev_clean, reason, counts = apply_filters(gev_raw)

    # 3. Stats
    print_stats(gev_raw, gev_clean, counts)

    # 4. Figures
    log.info("\nGenerating figures ...")
    fig_filter_funnel(counts)
    fig_xi_distribution(gev_raw, gev_after_xi, gev_clean)
    fig_pvalue_distribution(gev_raw, gev_clean)
    fig_param_histograms(gev_raw, gev_clean)
    fig_world_maps(gev_raw, gev_clean)
    fig_xi_pvalue_scatter(gev_raw, reason)
    fig_nyears(gev_raw, gev_clean)
    fig_q_ratio(gev_raw, gev_clean)

    # 5. Save data outputs
    save_outputs(gev_raw, gev_clean, counts, reason)

    # 6. Filter NC file
    cleaned_nc = filter_nc(gev_clean)

    # 7. Report
    print_report(gev_raw, gev_clean, counts)

    elapsed = time.time() - t0
    log.info(f"\nStage 3 done in {elapsed:.1f}s")
    log.info(f"  Figures  -> {S3_OUT_FIG}")
    log.info(f"  Data     -> {S3_OUT_DATA}")

    return {"gev_cleaned_csv": S3_OUT_DATA / "gev_cleaned.csv", "nc_cleaned": cleaned_nc}


# ================================================================================
# MAIN PIPELINE
# ================================================================================
def main():
    stage1_outputs = run_stage1()
    stage2_outputs = run_stage2()
    stage3_outputs = run_stage3()
    return stage1_outputs, stage2_outputs, stage3_outputs


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
