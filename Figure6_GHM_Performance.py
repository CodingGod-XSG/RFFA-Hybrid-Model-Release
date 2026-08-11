# -*- coding: utf-8 -*-
"""
Reproduces Figure 6 -- GHM (Global Hydrological Model) performance under PUR and
feature importance for GEV-NN and the hybrid model. Panels: (a) spatial
distribution of GHM Q100 NSE across PUR holdout regions with climate-zone inset
bars; (b) GHM NSE across return periods Q2-Q100; (c) top-10 features by mean
|SHAP value| for GEV-NN vs the hybrid model.

Layout:
Row 1: (a) GHM PUR Map (fixed US NA bug) & Signed Climate Inset (Red Arid)
       (b) GHM NSE Curve with Uncertainty Bands (Zoomed)
Row 2: (c) Split SHAP Importance (Y-limit 0.35, shifted rightwards)

Key Feature: Absolute mathematically locked labels with precise user-defined offsets.
"""
from __future__ import annotations

import glob
import logging
import re
import sys
import textwrap
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import DATA_RAW, DATA_PROCEED, FIGURE_ROOT, stage_dir

warnings.filterwarnings("ignore")
matplotlib.rcParams["axes.formatter.limits"] = (-3, 4)

# -----------------------------------------------------------------------------
# Config & IO
# -----------------------------------------------------------------------------
TAG = "Figure6_GHM_Performance"
OUT_DATA = stage_dir(DATA_PROCEED, TAG)
OUT_FIG = stage_dir(FIGURE_ROOT, TAG)

BASIN_CSV = DATA_PROCEED / "05_PUR_Basin_Select" / "station_basin_assignment.csv"
HYDRO_ROOT = DATA_RAW / "Hydrosheds"
HYBAS_LEVEL = 2
CLIMATE_SHP = DATA_RAW / "ClimateZone" / "ClimateZone5ClassMerge.shp"
NC_PATH = DATA_PROCEED / "02_Data-Clean" / "4_Cara-GRDC-35_cleaned.nc"

# GHM benchmark predictions (ALL stations and PUR-only subset)
HYDRO_ALL_CSV = DATA_PROCEED / "04_GHM_Benchmark" / "merged_obs_sim_pur.csv"
HYDRO_PUR_CSV = DATA_PROCEED / "04_GHM_Benchmark" / "pur_selected_obs_sim_pur.csv"

SHAP_DIR = DATA_PROCEED / "15_SHAP"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DATA / "log.txt", mode="w", encoding="utf-8"),
    ],
)
LOG = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants & Colors
# -----------------------------------------------------------------------------
RETURN_PERIODS = [2, 5, 10, 20, 50, 100]
CLIMATE_ORDER = ["Tropical", "Arid", "Temperate", "Cold"]

C_GEV_NN = "#E5E5E5"      # Light grey
C_HYBRID = "#22C55E"      # Bright green
C_ARID_TXT = "#D32F2F"    # Red color for Arid text in parentheses
MAP_CMAP = "RdYlGn"
MAP_VMIN, MAP_VMAX = 0.0, 1.0

# -----------------------------------------------------------------------------
# Style Setup
# -----------------------------------------------------------------------------
def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 24, "axes.titlesize": 28, "axes.labelsize": 24,
        "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.2, "grid.linestyle": "--", "figure.dpi": 120,
        "savefig.bbox": "tight"
    })

# -----------------------------------------------------------------------------
# Data Logic
# -----------------------------------------------------------------------------
def _nse(qt: np.ndarray, qp: np.ndarray) -> float:
    ok = np.isfinite(qt) & np.isfinite(qp) & (qt > 0) & (qp > 0)
    if ok.sum() < 3: return np.nan
    t, p = qt[ok], qp[ok]
    ss = float(np.sum((t - t.mean()) ** 2))
    return float(1 - np.sum((t - p) ** 2) / ss) if ss > 0 else np.nan

def _load_nc_coords() -> pd.DataFrame:
    if not NC_PATH.exists(): return pd.DataFrame(columns=["station_id", "lat", "lon"])
    with nc.Dataset(NC_PATH) as ds:
        stns = np.array(ds.variables["station"][:]).astype(str)
        lat = np.ma.filled(ds.variables["lat"][:], np.nan) if "lat" in ds.variables else np.full(len(stns), np.nan)
        lon = np.ma.filled(ds.variables["lon"][:], np.nan) if "lon" in ds.variables else np.full(len(stns), np.nan)
    return pd.DataFrame({"station_id": stns, "lat": lat, "lon": lon}).dropna(subset=["lat", "lon"]).drop_duplicates("station_id")

def _attach_coords(df: pd.DataFrame, nc_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["lat", "lon"]:
        if col not in out.columns: out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if nc_df.empty: return out
    out = out.merge(nc_df.rename(columns={"lat": "_lat", "lon": "_lon"}), on="station_id", how="left")
    out["lat"] = out["lat"].where(np.isfinite(out["lat"]), out["_lat"])
    out["lon"] = out["lon"].where(np.isfinite(out["lon"]), out["_lon"])
    return out.drop(columns=["_lat", "_lon"], errors="ignore")

def _load_ghm_curve_with_std() -> pd.DataFrame:
    if not HYDRO_ALL_CSV.exists(): return pd.DataFrame()
    df = pd.read_csv(HYDRO_ALL_CSV)
    rows = []
    np.random.seed(42)
    for t in RETURN_PERIODS:
        c_obs, c_sim = f"Q{t}_obs", f"Q{t}_sim"
        if c_obs in df.columns and c_sim in df.columns:
            sub = df[[c_obs, c_sim]].dropna()
            qt, qp = sub[c_obs].values, sub[c_sim].values
            val = _nse(qt, qp)

            # Simulate uncertainty via Bootstrap method
            boot_vals = []
            idx = np.arange(len(qt))
            for _ in range(100):
                b_idx = np.random.choice(idx, size=len(idx), replace=True)
                b_nse = _nse(qt[b_idx], qp[b_idx])
                if np.isfinite(b_nse): boot_vals.append(b_nse)
            std = float(np.std(boot_vals)) if boot_vals else 0.05
            rows.append({"return_period": t, "NSE": val, "NSE_std": std})
    return pd.DataFrame(rows)

def _assign_climate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["climate_zone"] = "Unknown"
    if out.empty or not CLIMATE_SHP.exists(): return out
    cz = gpd.read_file(CLIMATE_SHP)
    if cz.crs is None: cz = cz.set_crs("EPSG:4326")
    elif str(cz.crs) != "EPSG:4326": cz = cz.to_crs("EPSG:4326")

    pts = out.dropna(subset=["lat", "lon"]).copy()
    if pts.empty: return out
    gpts = gpd.GeoDataFrame(pts, geometry=[Point(xy) for xy in zip(pts["lon"], pts["lat"])], crs="EPSG:4326")
    joined = gpd.sjoin(gpts, cz[["Name", "geometry"]], how="left", predicate="within")
    mapped = joined[["station_id", "Name"]].drop_duplicates("station_id").rename(columns={"Name": "climate_zone"})
    out = out.drop(columns=["climate_zone"], errors="ignore").merge(mapped, on="station_id", how="left")
    out["climate_zone"] = out["climate_zone"].fillna("Unknown")
    return out

def _ghm_region_nse(station_df: pd.DataFrame, basin_assign: pd.DataFrame) -> pd.DataFrame:
    if station_df.empty or "basin_label" not in station_df.columns:
        return pd.DataFrame(columns=["basin_label", "n_stations", "NSE_Q100", "continent", "HYBAS_ID"])

    rows = []
    for basin_label, grp in station_df.groupby("basin_label"):
        basin_label = str(basin_label).strip()
        if not basin_label or basin_label.lower() == "nan": continue
        qt, qp = grp["Q100_true"].to_numpy(float), grp["Q100_pred"].to_numpy(float)
        rows.append({"basin_label": basin_label, "n_stations": int(grp["station_id"].nunique()), "NSE_Q100": _nse(qt, qp)})

    out = pd.DataFrame(rows)
    if not out.empty and not basin_assign.empty and {"basin_label", "continent", "HYBAS_ID"}.issubset(set(basin_assign.columns)):
        meta = basin_assign[["basin_label", "continent", "HYBAS_ID"]].drop_duplicates("basin_label")
        out = out.merge(meta, on="basin_label", how="left")
    return out

def _climate_nse_dict(station_df: pd.DataFrame) -> dict:
    if station_df.empty: return {z: np.nan for z in CLIMATE_ORDER}
    st = _assign_climate(station_df)
    out = {}
    for zone in CLIMATE_ORDER:
        grp = st[st["climate_zone"] == zone]
        out[zone] = _nse(grp["Q100_true"].to_numpy(float), grp["Q100_pred"].to_numpy(float)) if len(grp) >= 3 else np.nan
    return out

def _discover_shps(root: Path, level: int) -> dict:
    lvl = f"{level:02d}"
    out = {}
    for folder in sorted(root.glob("hybas_*_lev01-12_v1c")):
        code = folder.name.split("_")[1].lower()
        shp = folder / f"hybas_{code}_lev{lvl}_v1c.shp"
        if shp.exists(): out[code.upper()] = shp
    return out

def _load_world_basemap() -> gpd.GeoDataFrame:
    shp_map = _discover_shps(HYDRO_ROOT, HYBAS_LEVEL)
    frames = []
    for _, shp in sorted(shp_map.items()):
        try:
            gdf = gpd.read_file(shp)
            if gdf.crs is None: gdf = gdf.set_crs("EPSG:4326")
            elif str(gdf.crs) != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            frames.append(gdf[["geometry"]].copy())
        except: pass
    if not frames: return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")

def _load_pur_basin_polys(basin_assign: pd.DataFrame) -> gpd.GeoDataFrame:
    shp_map = _discover_shps(HYDRO_ROOT, HYBAS_LEVEL)
    rows = []
    for cont, sub in basin_assign.groupby("continent", observed=True):
        shp = shp_map.get(str(cont).upper())
        if not shp: continue
        gdf = gpd.read_file(shp)
        gdf["HYBAS_ID"] = pd.to_numeric(gdf["HYBAS_ID"], errors="coerce").fillna(0).astype("int64")
        keep = set(sub["HYBAS_ID"].astype("int64").tolist())
        sub_gdf = gdf[gdf["HYBAS_ID"].isin(keep)].copy()
        if sub_gdf.empty: continue
        if sub_gdf.crs is None: sub_gdf = sub_gdf.set_crs("EPSG:4326")
        elif str(sub_gdf.crs) != "EPSG:4326": sub_gdf = sub_gdf.to_crs("EPSG:4326")
        sub_gdf["continent"] = str(cont).upper()
        rows.append(sub_gdf[["continent", "HYBAS_ID", "geometry"]])
    if not rows: raise RuntimeError("No HydroSHEDS polygons loaded")

    out = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    out = out.drop_duplicates(subset=["continent", "HYBAS_ID"])
    bl = basin_assign[["continent", "HYBAS_ID", "basin_label"]].drop_duplicates()
    bl["continent"] = bl["continent"].astype(str).str.upper()
    bl["HYBAS_ID"] = bl["HYBAS_ID"].astype("int64")
    out = out.merge(bl, on=["continent", "HYBAS_ID"], how="left")
    out["basin_label"] = out["basin_label"].fillna(out["continent"] + "_" + out["HYBAS_ID"].astype(str))
    return out

# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------
def _draw_map_polished(ax, world_gdf, perf_gdf, pts, title: str, climate_nse: dict):
    if not world_gdf.empty:
        world_gdf.plot(ax=ax, facecolor="#F0F0F0", edgecolor="#222222", linewidth=0.4, zorder=1)

    if not perf_gdf.empty and "NSE_Q100" in perf_gdf.columns:
        norm = mcolors.Normalize(vmin=MAP_VMIN, vmax=MAP_VMAX)
        perf_gdf.plot(ax=ax, column="NSE_Q100", cmap=MAP_CMAP, norm=norm,
                      edgecolor="#111111", linewidth=0.8, alpha=0.95, zorder=3)
        '''
        b = perf_gdf.geometry.bounds
        for minx, miny, maxx, maxy in b.itertuples(index=False, name=None):
            if not np.isfinite((minx, miny, maxx, maxy)).all(): continue
            w = max(float(maxx - minx), 0.0)
            h = max(float(maxy - miny), 0.0)
            if w <= 0 or h <= 0: continue
            ax.add_patch(mpatches.Rectangle(
                (float(minx), float(miny)), w, h,
                fill=False, edgecolor="#1B1B1B", linewidth=0.90,
                alpha=0.85, zorder=4
            ))
        '''
    ax.set_xlim(-145, 180); ax.set_ylim(-60, 85); ax.set_axis_off()
    ax.text(-120, -45, title, fontsize=34, fontweight="bold")

    # Climate Inset
    if climate_nse:
        ins = ax.inset_axes([0.80, 0.58, 0.20, 0.28])
        zones = CLIMATE_ORDER[::-1]
        vals = [climate_nse.get(z, np.nan) for z in zones]
        abs_vals = [abs(v) if np.isfinite(v) else 0 for v in vals]

        y_pos = np.arange(len(zones))
        ins.barh(y_pos, abs_vals, height=0.6, color=C_HYBRID, edgecolor="white", alpha=0.85)
        ins.set_yticks(y_pos); ins.set_yticklabels(zones, fontsize=16)
        ins.set_xticks([]); ins.spines[['top', 'right', 'bottom']].set_visible(False)
        ins.set_facecolor((1, 1, 1, 0.8))

        max_abs = max(abs_vals) if any(abs_vals) else 1
        ins.set_xlim(0, max_abs * 1.6)

        # Red color specifically for Arid text
        for i, (v, zone) in enumerate(zip(vals, zones)):
            if np.isfinite(v):
                sign = "+" if v >= 0 else ""
                txt_color = C_ARID_TXT if zone == "Arid" else "#111111"
                ins.text(abs(v) + 0.02, i, f"({sign}{v:.2f})", va='center', fontsize=15, fontweight="bold", color=txt_color)

def _draw_ghm_curve_polished(ax, df: pd.DataFrame):
    ax.axvspan(3.5, 5.5, color="#E6F2FF", alpha=0.6, zorder=0)
    ax.text(4.5, 0.97, "Extreme\ntail", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=18, color="#2C4A73", fontstyle="italic")

    if not df.empty:
        x = np.arange(len(RETURN_PERIODS))
        y, e = df["NSE"].values, df["NSE_std"].values

        xs = np.linspace(0, 5, 200)
        ys = PchipInterpolator(x, y)(xs) if len(x) >= 3 else np.interp(xs, x, y)
        es = PchipInterpolator(x, e)(xs) if len(x) >= 3 else np.interp(xs, x, e)
        es = np.maximum(es, 0.005)

        ax.fill_between(xs, ys-es, ys+es, color=C_HYBRID, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(xs, ys, color=C_HYBRID, lw=3.5, zorder=4)
        ax.scatter(x, y, color=C_HYBRID, s=100, edgecolors="white", linewidths=1.5, zorder=5, label="GHM")
        ax.set_ylim(min(y)-0.05, max(y)+0.08)

    ax.set_xticks(np.arange(len(RETURN_PERIODS)))
    ax.set_xticklabels([f"Q{t}" for t in RETURN_PERIODS])
    ax.set_ylabel("NSE", fontsize=24)
    ax.legend(loc="lower left", fontsize=18)

def _load_hybrid_pur_curve() -> pd.DataFrame:
    """Load GEV-Flow (Hybrid Model) PUR predictions and compute per-return-period NSE (mean±std across seeds)."""
    tag_dir, fn_prefix, ftag = "11_GEV_NN", "GEV_NN_ST", "+flow"
    src = DATA_PROCEED / tag_dir
    pat = str(src / f"predictions*{fn_prefix}_PUR_*_s*_{ftag}.csv")
    files = sorted(glob.glob(pat))
    if not files:
        LOG.warning("No Hybrid Model PUR files found: %s", pat)
        return pd.DataFrame()

    def _seed_from_local(n: str) -> str:
        m = re.search(r"_s(\d+)(?:_split\d+)?_", str(n))
        return m.group(1) if m else "unk"

    frames = []
    for fp in files:
        d = pd.read_csv(fp)
        d["seed"] = _seed_from_local(Path(fp).name)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw["station_id"] = raw["station_id"].astype(str).str.strip()

    rows = []
    for seed, g in raw.groupby("seed", sort=True):
        qt_cols = [f"Q{t}_true" for t in RETURN_PERIODS]
        qp_cols = [f"Q{t}_pred" for t in RETURN_PERIODS]
        fc = [c for c in qt_cols if c in g.columns]
        pc = [c for c in qp_cols if c in g.columns]
        if not fc or not pc:
            continue
        agg = g.groupby("station_id")
        wide = pd.concat([agg[fc].first(), agg[pc].median()], axis=1).reset_index()
        for t in RETURN_PERIODS:
            ct, cp = f"Q{t}_true", f"Q{t}_pred"
            if ct not in wide.columns or cp not in wide.columns:
                continue
            qt = wide[ct].to_numpy(float)
            qp = wide[cp].to_numpy(float)
            rows.append({"return_period": t, "seed": seed, "NSE": _nse(qt, qp)})

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    result = (df.groupby("return_period")["NSE"]
              .agg(["mean", "std"]).reset_index()
              .rename(columns={"mean": "NSE", "std": "NSE_std"}))
    return result.sort_values("return_period").reset_index(drop=True)


def _draw_hybrid_pur_curve_polished(ax, df_ghm: pd.DataFrame, df_hybrid: pd.DataFrame):
    """Panel (b): GHM vs Hybrid Model PUR NSE curves overlaid for comparison."""
    ax.axvspan(3.5, 5.5, color="#E6F2FF", alpha=0.6, zorder=0)
    ax.text(4.5, 0.97, "Extreme\ntail", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=18, color="#2C4A73", fontstyle="italic")

    all_y = []
    x = np.arange(len(RETURN_PERIODS))

    def _plot_one(df, color, label, lw, dash=None):
        if df.empty:
            return
        y = np.array([
            float(df[df["return_period"] == t]["NSE"].iloc[0])
            if not df[df["return_period"] == t].empty else np.nan
            for t in RETURN_PERIODS
        ], dtype=float)
        e = np.array([
            float(df[df["return_period"] == t]["NSE_std"].iloc[0])
            if not df[df["return_period"] == t].empty else 0.0
            for t in RETURN_PERIODS
        ], dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < 2:
            return
        xv = x[ok].astype(float)
        yv = y[ok].astype(float)
        ev = np.nan_to_num(e[ok].astype(float), nan=0.0)
        xs = np.linspace(float(xv.min()), float(xv.max()), 200)
        ys = PchipInterpolator(xv, yv)(xs) if len(xv) >= 3 else np.interp(xs, xv, yv)
        es = PchipInterpolator(xv, ev)(xs) if len(xv) >= 3 else np.interp(xs, xv, ev)
        es = np.maximum(es, 0.005)
        ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.15, linewidth=0, zorder=2)
        ln, = ax.plot(xs, ys, color=color, lw=lw, zorder=4, label=label)
        if dash:
            ln.set_dashes(dash)
        ax.scatter(xv, yv, color=color, s=90, edgecolors="white", linewidths=1.5, zorder=5)
        all_y.extend(yv.tolist())

    # GHM curve (solid)
    _plot_one(df_ghm, C_HYBRID, "GHM", lw=3.5)
    # Hybrid Model PUR curve (dashed)
    _plot_one(df_hybrid, "#1E40AF", "Hybrid Model (PUR)", lw=2.8, dash=(6, 3))

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{t}" for t in RETURN_PERIODS])
    ax.set_ylabel("NSE", fontsize=24)
    if all_y:
        ax.set_ylim(max(0.0, float(np.nanmin(all_y)) - 0.05),
                    min(1.0, float(np.nanmax(all_y)) + 0.08))
    ax.legend(loc="lower left", fontsize=18)


def _draw_split_shap_polished(ax, shap_gev: pd.DataFrame, shap_hyb: pd.DataFrame):
    if shap_gev.empty and shap_hyb.empty:
        ax.text(0.5, 0.5, "No SHAP data", transform=ax.transAxes, ha="center", fontsize=22)
        return

    d1 = shap_gev.sort_values("mean_abs_shap", ascending=False).head(10) if not shap_gev.empty else pd.DataFrame()
    d2 = shap_hyb.sort_values("mean_abs_shap", ascending=False).head(10) if not shap_hyb.empty else pd.DataFrame()

    n1, n2 = len(d1), len(d2)
    gap = 1.4

    x1 = np.arange(n1)
    x2 = np.arange(n2) + n1 + gap

    all_x = np.concatenate([x1, x2])
    all_labels = list(d1["feature"]) + list(d2["feature"])

    if n1 > 0:
        ax.bar(x1, d1["mean_abs_shap"], color=C_GEV_NN, edgecolor="#111111", linewidth=1.5, alpha=0.95, label="GEV-NN")
    if n2 > 0:
        ax.bar(x2, d2["mean_abs_shap"], color=C_HYBRID, edgecolor="#111111", linewidth=1.5, alpha=0.90, label="Hybrid Model")

    # Fontsize 16 for better readability
    ax.set_xticks(all_x)
    ax.set_xticklabels([textwrap.fill(str(l), 12) for l in all_labels], rotation=35, ha="right", fontsize=16)

    if n1 > 0 and n2 > 0:
        ax.axvline(n1 + gap/2 - 0.5, color="#CCCCCC", linestyle="--", lw=1.5, zorder=0)

    ax.set_ylabel("Mean |SHAP value|", fontsize=24)

    # Strictly fix Y-axis range
    ax.set_ylim(0, 0.35)

    # Legend right-aligned
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=22)

    # Clamping xlim so the rightmost bar aligns flush with the legend
    if len(all_x) > 0:
        ax.set_xlim(all_x[0] - 0.8, all_x[-1] + 0.6)

    # Shift axes computationally rightward
    pos = ax.get_position()
    shift = 0.025
    ax.set_position([pos.x0 + shift, pos.y0, pos.width - shift, pos.height])

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    _setup_style()
    LOG.info("Start %s", TAG)

    nc_df = _load_nc_coords() if NC_PATH.exists() else pd.DataFrame()
    world_gdf = _load_world_basemap()

    # keep_default_na=False to save North America ("NA") from becoming NaN
    basin_assign = pd.read_csv(BASIN_CSV, keep_default_na=False) if BASIN_CSV.exists() else pd.DataFrame(columns=["station_id", "continent", "HYBAS_ID", "basin_label"])
    if not basin_assign.empty:
        basin_assign["station_id"] = basin_assign["station_id"].astype(str).str.strip()
        basin_assign["continent"] = basin_assign["continent"].replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
        basin_assign["HYBAS_ID"] = pd.to_numeric(basin_assign["HYBAS_ID"], errors="coerce")
        basin_assign = basin_assign.dropna(subset=["station_id", "continent", "HYBAS_ID"]).copy()
        basin_assign["HYBAS_ID"] = basin_assign["HYBAS_ID"].astype("int64")
        basin_assign["continent"] = basin_assign["continent"].astype(str).str.strip().str.upper()

    ghm_fp = HYDRO_PUR_CSV if HYDRO_PUR_CSV.exists() else HYDRO_ALL_CSV
    ghm_data = pd.read_csv(ghm_fp) if ghm_fp.exists() else pd.DataFrame()
    if not ghm_data.empty:
        ghm_data["station_id"] = ghm_data["station_id"].astype(str).str.strip()
        ghm_data = _attach_coords(ghm_data.rename(columns={"Q100_obs":"Q100_true", "Q100_sim":"Q100_pred"}), nc_df)

    ghm_region = _ghm_region_nse(ghm_data, basin_assign)
    ghm_climate = _climate_nse_dict(ghm_data)

    try: pur_polys = _load_pur_basin_polys(basin_assign)
    except: pur_polys = gpd.GeoDataFrame(columns=["basin_label", "geometry"], geometry="geometry", crs="EPSG:4326")
    map_gdf = pur_polys.merge(ghm_region[["basin_label", "NSE_Q100"]], on="basin_label", how="right") if not pur_polys.empty else pur_polys.copy()

    df_curve = _load_ghm_curve_with_std()

    fp_base = SHAP_DIR / "shap_importance_GEV_NN_base.csv"
    fp_flow = SHAP_DIR / "shap_importance_GEV_NN_flow.csv"
    shap_gev = pd.read_csv(fp_base) if fp_base.exists() else pd.DataFrame(columns=["feature", "mean_abs_shap"])
    shap_hyb = pd.read_csv(fp_flow) if fp_flow.exists() else pd.DataFrame(columns=["feature", "mean_abs_shap"])

    # Figure Layout
    fig = plt.figure(figsize=(24, 16))

    # Use tightly controlled margins to calculate absolute coordinate positions
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[4.0, 1.6],
                           hspace=0.15, wspace=0.15, left=0.06, right=0.96, top=0.92, bottom=0.10)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_shap = fig.add_subplot(gs[1, :])

    # Draw Panels
    ghm_pts = ghm_data[["station_id", "lat", "lon"]].drop_duplicates("station_id") if not ghm_data.empty else pd.DataFrame()
    _draw_map_polished(ax_map, world_gdf, map_gdf, ghm_pts, "GHM", ghm_climate)

    sm = plt.cm.ScalarMappable(cmap=MAP_CMAP, norm=mcolors.Normalize(MAP_VMIN, MAP_VMAX))
    sm.set_array([])
    cbar_ax = ax_map.inset_axes([0.47, 0.05, 0.28, 0.035])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_ticks(np.linspace(MAP_VMIN, MAP_VMAX, 6))
    cb.set_label("Q100 NSE", fontsize=22, labelpad=8)
    cb.ax.tick_params(labelsize=18, length=5, width=1.2, color="#444444")
    cb.outline.set_linewidth(1.2)
    cb.outline.set_edgecolor("#444444")
    cbar_ax.set_facecolor((1, 1, 1, 0.84))

    _draw_ghm_curve_polished(ax_curve, df_curve)
    _draw_split_shap_polished(ax_shap, shap_gev, shap_hyb)

    # -------------------------------------------------------------------------
    # Absolute Label Placement
    # -------------------------------------------------------------------------
    # Draw canvas first to finalize dynamic bbox calculations
    fig.canvas.draw()

    # Get physical coordinates of the plotted axes to act as anchors
    pos_a = ax_map.get_position()
    pos_b = ax_curve.get_position()
    pos_c = ax_shap.get_position()

    # Displacement offsets corresponding to approximately ~1 label height down and ~2 label widths right
    shift_down = 0.035
    shift_right = 0.05

    # Define locked horizontal & vertical reference lines
    label_x_left  = 0.02 + shift_right
    label_x_right = pos_b.x0 - 0.035
    label_y_top   = max(pos_a.y1, pos_b.y1) + 0.015 - shift_down
    label_y_bot   = pos_c.y1 + 0.025

    # Apply mathematically perfect labels
    fig.text(label_x_left, label_y_top, "a", fontsize=42, fontweight="bold", va="bottom", ha="left")
    fig.text(label_x_right, label_y_top, "b", fontsize=42, fontweight="bold", va="bottom", ha="left")
    fig.text(label_x_left, label_y_bot, "c", fontsize=42, fontweight="bold", va="bottom", ha="left")

    # Save original figure
    for fmt in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"{TAG}.{fmt}")
        LOG.info("Saved figure: %s", OUT_FIG / f"{TAG}.{fmt}")
    plt.close(fig)

    # ── Second figure: same layout, panel (b) → Hybrid Model PUR NSE curve ──
    LOG.info("Generating second figure with Hybrid Model PUR curve in panel (b) …")
    df_hybrid_pur = _load_hybrid_pur_curve()
    if df_hybrid_pur.empty:
        LOG.warning("No Hybrid Model PUR data found — skipping second figure")
    else:
        fig2 = plt.figure(figsize=(24, 16))
        gs2 = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[4.0, 1.6],
                                hspace=0.15, wspace=0.15, left=0.06, right=0.96,
                                top=0.92, bottom=0.10)
        ax2_map   = fig2.add_subplot(gs2[0, 0])
        ax2_curve = fig2.add_subplot(gs2[0, 1])
        ax2_shap  = fig2.add_subplot(gs2[1, :])

        # Panel a: identical GHM map
        _draw_map_polished(ax2_map, world_gdf, map_gdf, ghm_pts, "GHM", ghm_climate)
        sm2 = plt.cm.ScalarMappable(cmap=MAP_CMAP, norm=mcolors.Normalize(MAP_VMIN, MAP_VMAX))
        sm2.set_array([])
        cbar_ax2 = ax2_map.inset_axes([0.47, 0.05, 0.28, 0.035])
        cb2 = fig2.colorbar(sm2, cax=cbar_ax2, orientation="horizontal")
        cb2.set_ticks(np.linspace(MAP_VMIN, MAP_VMAX, 6))
        cb2.set_label("Q100 NSE", fontsize=22, labelpad=8)
        cb2.ax.tick_params(labelsize=18, length=5, width=1.2, color="#444444")
        cb2.outline.set_linewidth(1.2)
        cb2.outline.set_edgecolor("#444444")
        cbar_ax2.set_facecolor((1, 1, 1, 0.84))

        # Panel b: GHM vs Hybrid Model PUR NSE curves (comparison)
        _draw_hybrid_pur_curve_polished(ax2_curve, df_curve, df_hybrid_pur)

        # Panel c: identical SHAP
        _draw_split_shap_polished(ax2_shap, shap_gev, shap_hyb)

        # Labels
        fig2.canvas.draw()
        pos2_a = ax2_map.get_position()
        pos2_b = ax2_curve.get_position()
        pos2_c = ax2_shap.get_position()
        shift_down, shift_right = 0.035, 0.05
        lx_left  = 0.02 + shift_right
        lx_right = pos2_b.x0 - 0.035
        ly_top   = max(pos2_a.y1, pos2_b.y1) + 0.015 - shift_down
        ly_bot   = pos2_c.y1 + 0.025
        fig2.text(lx_left,  ly_top, "a", fontsize=42, fontweight="bold", va="bottom", ha="left")
        fig2.text(lx_right, ly_top, "b", fontsize=42, fontweight="bold", va="bottom", ha="left")
        fig2.text(lx_left,  ly_bot, "c", fontsize=42, fontweight="bold", va="bottom", ha="left")

        for fmt in ("png", "pdf"):
            fp2 = OUT_FIG / f"{TAG}_hybrid_pur_b.{fmt}"
            fig2.savefig(fp2)
            LOG.info("Saved second figure: %s", fp2)
        plt.close(fig2)

if __name__ == "__main__":
    main()
