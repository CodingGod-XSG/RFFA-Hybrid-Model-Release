# -*- coding: utf-8 -*-
"""Figure3_ML_Performance.py

Reproduces Figure 3 - Model performance under PUB and PUR across return
periods and regions. Panels: (a,b) spatial distribution of Q100 NSE under
PUR for XGBoost and ANN with climate-zone inset bar charts; (c) NSE across
return periods Q2-Q100 under PUB/PUR for XGBoost/ANN; (d) PUB-PUR gap
(delta NSE) by return period.

Fixes vs v1:
  - Map aspect ratio corrected (Robinson-style equirectangular)
  - PUR region coverage diagnostic + relaxed merge
  - CDF built from raw station-level abs_rel_err (not seed-level rRMSE)
  - Climate zone uses Q100 abs_rel_err, y-axis clipped to 200%,
    annotates n per group
  - Shared colorbar for both maps (single strip, right of maps)
  - ΔNSE bar panel added (PUR−PUB gap by model) replacing empty slot
  - Natural Earth coastlines as fallback world background
"""
from __future__ import annotations

import glob
import logging
import math
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import DATA_RAW, DATA_PROCEED, FIGURE_ROOT, stage_dir

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
from scipy import stats
from shapely.geometry import Point

warnings.filterwarnings("ignore")
matplotlib.rcParams["axes.formatter.limits"] = (-3, 4)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

PROC     = DATA_PROCEED
FIG_ROOT = FIGURE_ROOT

TAG      = "Figure3_ML_Performance"
OUT_DATA = stage_dir(DATA_PROCEED, TAG)
OUT_FIG  = stage_dir(FIGURE_ROOT, TAG)

BASIN_CSV   = DATA_PROCEED / "05_PUR_Basin_Select" / "station_basin_assignment.csv"
HYDRO_ROOT  = DATA_RAW / "Hydrosheds"
HYBAS_LEVEL = 2
CLIMATE_SHP = DATA_RAW / "ClimateZone" / "ClimateZone5ClassMerge.shp"
NC_PATH     = DATA_PROCEED / "02_Data-Clean" / "4_Cara-GRDC-35_cleaned.nc"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DATA / "log.txt", mode="w", encoding="utf-8"),
    ],
)
LOG = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RETURN_PERIODS = [2, 5, 10, 20, 50, 100]
Q_TAIL = 100

MODEL_META = {
    "RF":      ("06_RF",       "RF"),
    "ANN":     ("09_ANN",      "ANN_Single"),
    "XGBoost": ("08_XGBoost",  "XGBoost"),
    "SVM":     ("07_SVM",      "SVM"),
}

CLIMATE_ORDER = ["Tropical", "Arid", "Temperate", "Cold"]

C_PUB = "#2166AC"
C_PUR = "#B2182B"
C_RF  = "#6B7280"
C_ANN = "#E69F00"
C_XGB = "#2E7D32"   # dark green for XGBoost
C_SVM = "#6A1B9A"   # purple for SVM

MODEL_COLORS: dict[str, str] = {
    "RF":      C_RF,
    "ANN":     C_ANN,
    "XGBoost": C_XGB,
    "SVM":     C_SVM,
}

# Per-model map label positions (longer names shift left so panel letter aligns)
_MAP_TITLE_X: dict[str, float] = {
    "RF":      -115.2,
    "ANN":     -115.2,
    "SVM":     -115.2,
    "XGBoost": -130.0,
}
_MAP_LETTER_X: dict[str, float] = {
    "RF":      -125.0,
    "ANN":     -125.0,
    "SVM":     -125.0,
    "XGBoost": -140.0,
}

MAP_CMAP = "RdYlGn"
MAP_VMIN, MAP_VMAX = 0.0, 1.0

ERR_CMAP = "RdYlGn_r"   # for PUB error scatter (low error = green)
ERR_VMIN, ERR_VMAX = 0.0, 100.0  # % abs_rel_error

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
def _setup_style() -> None:
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          26,
        "axes.titlesize":     28,
        "axes.labelsize":     26,
        "xtick.labelsize":    22,
        "ytick.labelsize":    22,
        "legend.fontsize":    20,
        "legend.frameon":     False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.color":         "#DDDDDD",
        "grid.linestyle":     "--",
        "grid.linewidth":     0.5,
        "figure.dpi":         120,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.10,
    })


def _letter(ax, txt: str, x: float = -0.07, y: float = 1.03) -> None:
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=36, fontweight="bold", va="top", ha="left", zorder=10)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def _nse(qt: np.ndarray, qp: np.ndarray) -> float:
    ok = np.isfinite(qt) & np.isfinite(qp) & (qt > 0) & (qp > 0)
    if ok.sum() < 3:
        return np.nan
    t, p = qt[ok], qp[ok]
    ss = float(np.sum((t - t.mean()) ** 2))
    return float(1 - np.sum((t - p) ** 2) / ss) if ss > 0 else np.nan


def _ecdf(v: np.ndarray):
    x = np.sort(v)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────
def _seed_from(name: str) -> str:
    m = re.search(r"_s(\d+)(?:_split\d+)?_", str(name))
    return m.group(1) if m else "unk"


def _pur_region_from(name: str, fn_prefix: str, ftag: str) -> str:
    pat = re.compile(
        rf"^predictions_{re.escape(fn_prefix)}_PUR_(?P<r>.+?)_s\d+(?:_split\d+)?_{re.escape(ftag)}\.csv$"
    )
    m = pat.match(str(name))
    if m:
        return m.group("r")
    text = str(name)
    i = text.find("PUR_")
    j = re.search(r"_s\d+", text)
    return text[i + 4 : j.start()] if (i >= 0 and j and j.start() > i + 4) else ""


def _filter_pur(raw: pd.DataFrame, fn_prefix: str, ftag: str) -> pd.DataFrame:
    """Drop bottom 30% PUR regions by station count (same as 20_ModelComparison)."""
    if raw.empty or "source_file" not in raw.columns:
        return raw
    df = raw.copy()
    df["_r"] = df["source_file"].map(lambda n: _pur_region_from(n, fn_prefix, ftag))
    df = df[df["_r"] != ""]
    if df.empty:
        return raw.iloc[0:0]
    rc = df.groupby("_r", observed=True)["station_id"].nunique().sort_values(ascending=False)
    n_drop = int(math.ceil(len(rc) * 0.30))
    # rc is sorted descending; keep the first (largest) 70% regions.
    keep_n = max(len(rc) - n_drop, 0)
    keep = rc.index[:keep_n].tolist()
    LOG.info("  PUR filter: %d total regions → keep %d, drop %d",
             len(rc), len(keep), n_drop)
    return df[df["_r"].isin(keep)].drop(columns=["_r"])


def _load_raw(model: str, exp: str) -> pd.DataFrame:
    tag_dir, fn_prefix = MODEL_META[model]
    ftag = "base"
    src = PROC / tag_dir
    pat = (str(src / f"predictions_{fn_prefix}_PUB_PUB_s*_{ftag}.csv") if exp == "PUB"
           else str(src / f"predictions_{fn_prefix}_PUR_*_s*_{ftag}.csv"))
    files = sorted(glob.glob(pat))
    if not files:
        LOG.warning("  No files: %s", pat)
        return pd.DataFrame()
    frames = []
    for fp in files:
        d = pd.read_csv(fp)
        d["source_file"] = Path(fp).name
        d["seed"] = _seed_from(Path(fp).name)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw["station_id"] = raw["station_id"].astype(str).str.strip()
    if exp == "PUR":
        LOG.info("  PUR filter: skipped (keep all regions; already filtered in training)")
    return raw


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Station-level wide table: true cols = first, pred cols = median over seeds."""
    if raw.empty:
        return pd.DataFrame()
    qt = [f"Q{t}_true" for t in RETURN_PERIODS]
    qp = [f"Q{t}_pred" for t in RETURN_PERIODS]
    need = ["station_id"] + qt + qp
    miss = [c for c in need if c not in raw.columns]
    if miss:
        LOG.warning("  Missing columns: %s", miss)
        return pd.DataFrame()
    first_cols = [c for c in ["lat", "lon"] + qt if c in raw.columns]
    g = raw.groupby("station_id")
    return pd.concat([g[first_cols].first(), g[qp].median()], axis=1).reset_index()


def _by_seed(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if raw.empty or "seed" not in raw.columns:
        return {}
    return {str(s): _aggregate(d) for s, d in raw.groupby("seed", sort=True)}


def _load_nc_coords() -> pd.DataFrame:
    if not NC_PATH.exists():
        return pd.DataFrame(columns=["station_id", "lat", "lon"])
    lat_kw = ["static_gauge_lat", "gauge_lat", "lat"]
    lon_kw = ["static_gauge_lon", "gauge_lon", "lon"]
    with nc.Dataset(NC_PATH) as ds:
        stns = np.array(ds.variables["station"][:]).astype(str)
        lv = next((k for k in lat_kw if k in ds.variables), None)
        lov = next((k for k in lon_kw if k in ds.variables), None)
        if not lv or not lov:
            return pd.DataFrame(columns=["station_id", "lat", "lon"])
        lat = np.ma.filled(ds.variables[lv][:], np.nan).astype(float)
        lon = np.ma.filled(ds.variables[lov][:], np.nan).astype(float)
    df = pd.DataFrame({"station_id": stns.astype(str), "lat": lat, "lon": lon})
    return df.dropna(subset=["lat", "lon"]).drop_duplicates("station_id")


def _attach_coords(df: pd.DataFrame, nc_df: pd.DataFrame, basin_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["lat", "lon"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if not nc_df.empty:
        out = out.merge(nc_df.rename(columns={"lat": "_lat", "lon": "_lon"}),
                        on="station_id", how="left")
        for col, _col in [("lat", "_lat"), ("lon", "_lon")]:
            out[col] = out[col].where(np.isfinite(out[col]), out[_col])
        out = out.drop(columns=["_lat", "_lon"], errors="ignore")
    if not basin_df.empty:
        bsub = basin_df[["station_id", "lat", "lon"]].rename(
            columns={"lat": "_blat", "lon": "_blon"})
        out = out.merge(bsub, on="station_id", how="left")
        for col, _col in [("lat", "_blat"), ("lon", "_blon")]:
            out[col] = out[col].where(np.isfinite(out[col]), out[_col])
        out = out.drop(columns=["_blat", "_blon"], errors="ignore")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# GeoData helpers
# ─────────────────────────────────────────────────────────────────────────────
def _discover_shps(root: Path, level: int) -> dict[str, Path]:
    lvl = f"{level:02d}"
    out = {}
    for folder in sorted(root.glob("hybas_*_lev01-12_v1c")):
        code = folder.name.split("_")[1].lower()
        shp = folder / f"hybas_{code}_lev{lvl}_v1c.shp"
        if shp.exists():
            out[code.upper()] = shp
    return out


def _load_world_basemap() -> gpd.GeoDataFrame:
    """Full HydroSHEDS world basemap (all continents, all basins at level)."""
    shp_map = _discover_shps(HYDRO_ROOT, HYBAS_LEVEL)
    if not shp_map:
        LOG.warning("No HydroSHEDS shapefiles found — using empty basemap")
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    frames = []
    for cont, shp in sorted(shp_map.items()):
        try:
            gdf = gpd.read_file(shp)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            elif str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            frames.append(gdf[["geometry"]].copy())
            LOG.info("  World basemap loaded: %s (%d polygons)", cont, len(gdf))
        except Exception as e:
            LOG.warning("  Failed loading %s: %s", shp, e)
    if not frames:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                            geometry="geometry", crs="EPSG:4326")


def _load_pur_basin_polys(basin_assign: pd.DataFrame) -> gpd.GeoDataFrame:
    """Load HydroSHEDS polygons only for basins in assignment CSV."""
    shp_map = _discover_shps(HYDRO_ROOT, HYBAS_LEVEL)
    rows = []
    for cont, sub in basin_assign.groupby("continent", observed=True):
        shp = shp_map.get(str(cont).upper())
        if not shp:
            LOG.warning("  Missing shapefile for continent: %s", cont)
            continue
        gdf = gpd.read_file(shp)
        gdf["HYBAS_ID"] = pd.to_numeric(gdf["HYBAS_ID"], errors="coerce").astype("Int64")
        gdf = gdf.dropna(subset=["HYBAS_ID"])
        gdf["HYBAS_ID"] = gdf["HYBAS_ID"].astype("int64")
        keep = set(sub["HYBAS_ID"].astype("int64").tolist())
        sub_gdf = gdf[gdf["HYBAS_ID"].isin(keep)].copy()
        if sub_gdf.empty:
            continue
        if sub_gdf.crs is None:
            sub_gdf = sub_gdf.set_crs("EPSG:4326")
        elif str(sub_gdf.crs) != "EPSG:4326":
            sub_gdf = sub_gdf.to_crs("EPSG:4326")
        sub_gdf["continent"] = str(cont).upper()
        rows.append(sub_gdf[["continent", "HYBAS_ID", "geometry"]])
    if not rows:
        raise RuntimeError("No HydroSHEDS polygons loaded")
    out = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True),
                           geometry="geometry", crs="EPSG:4326")
    out = out.drop_duplicates(subset=["continent", "HYBAS_ID"])
    bl = basin_assign[["continent", "HYBAS_ID", "basin_label"]].drop_duplicates()
    bl["continent"] = bl["continent"].str.upper()
    bl["HYBAS_ID"] = bl["HYBAS_ID"].astype("int64")
    out = out.merge(bl, on=["continent", "HYBAS_ID"], how="left")
    out["basin_label"] = out["basin_label"].fillna(
        out["continent"] + "_" + out["HYBAS_ID"].astype(str))
    rp = out.geometry.representative_point()
    out["lon_centroid"] = rp.x
    out["lat_centroid"] = rp.y
    LOG.info("  PUR basin polygons: %d loaded", len(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Computations
# ─────────────────────────────────────────────────────────────────────────────
def _nse_by_climate(model: str, nc_coords: pd.DataFrame) -> dict:
    """Compute Q100 NSE per climate zone for PUR experiment."""
    raw = _load_raw(model, "PUR")
    if raw.empty: return {}
    wide = _aggregate(raw)
    if wide.empty: return {}
    wide = _attach_coords(wide, nc_coords, pd.DataFrame(columns=["station_id"]))
    wide = _assign_climate(wide)
    
    out = {}
    for zone in CLIMATE_ORDER:
        grp = wide[wide["climate_zone"] == zone]
        if len(grp) < 3:
            out[zone] = np.nan
            continue
        qt = grp[f"Q{Q_TAIL}_true"].to_numpy(float)
        qp = grp[f"Q{Q_TAIL}_pred"].to_numpy(float)
        out[zone] = _nse(qt, qp)
    return out


def _pur_region_nse(model: str, basin_assign: pd.DataFrame,
                    nc_coords: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Compute Q100 NSE for each PUR region (region parsed from PUR filenames)."""
    _, fn_prefix = MODEL_META[model]
    ftag = "base"
    raw = _load_raw(model, "PUR")
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["station_id", "lat", "lon"]), set()

    raw = raw.copy()
    raw["pur_region"] = raw["source_file"].map(
        lambda n: _pur_region_from(n, fn_prefix, ftag)
    )
    raw["pur_region"] = raw["pur_region"].astype(str).str.strip()
    raw = raw[raw["pur_region"] != ""]
    if raw.empty:
        LOG.warning("  No PUR region labels parsed for %s", model)
        return pd.DataFrame(), pd.DataFrame(columns=["station_id", "lat", "lon"]), set()

    pur_regions = set(raw["pur_region"].unique().tolist())

    # Station points for map scatter.
    wide_all = _aggregate(raw)
    wide_all = _attach_coords(wide_all, nc_coords, basin_assign)
    pts = wide_all[["station_id", "lat", "lon"]].drop_duplicates("station_id")

    rows = []
    for region, g in raw.groupby("pur_region", observed=True):
        w = _aggregate(g)
        if w.empty:
            continue
        qt = w[f"Q{Q_TAIL}_true"].to_numpy(float)
        qp = w[f"Q{Q_TAIL}_pred"].to_numpy(float)
        rows.append({
            "basin_label": str(region),
            "n_stations": int(w["station_id"].nunique()),
            "NSE_Q100": _nse(qt, qp),
        })

    region_df = pd.DataFrame(rows)
    basin_meta = basin_assign[["basin_label", "continent", "HYBAS_ID"]].drop_duplicates("basin_label")
    if not region_df.empty:
        region_df = region_df.merge(basin_meta, on="basin_label", how="left")
        miss = int(region_df["continent"].isna().sum())
        if miss > 0:
            LOG.warning("  %s PUR regions without basin geometry mapping: %d", model, miss)

    LOG.info("  %s PUR regions in files: %d | with NSE: %d",
             model, len(pur_regions), len(region_df))
    return region_df, pts, pur_regions


def _seed_nse_curve(model: str) -> pd.DataFrame:
    rows = []
    for exp in ["PUB", "PUR"]:
        raw = _load_raw(model, exp)
        for seed, w in _by_seed(raw).items():
            for t in RETURN_PERIODS:
                qt = w[f"Q{t}_true"].to_numpy(float)
                qp = w[f"Q{t}_pred"].to_numpy(float)
                rows.append({"model": model, "experiment": exp, "seed": seed,
                             "return_period": t, "NSE": _nse(qt, qp)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.groupby(["model", "experiment", "return_period"], as_index=False)["NSE"]
            .agg(["mean", "std"]).reset_index()
            .rename(columns={"mean": "NSE_mean", "std": "NSE_std"}))


def _station_abs_rel(model: str, basin_assign: pd.DataFrame,
                     nc_coords: pd.DataFrame) -> pd.DataFrame:
    """Per-station Q100 absolute relative error (PUB+PUR combined)."""
    rows = []
    for exp in ["PUB", "PUR"]:
        raw = _load_raw(model, exp)
        if raw.empty:
            continue
        wide = _aggregate(raw)
        if wide.empty:
            continue
        wide = _attach_coords(wide, nc_coords, basin_assign)
        qt = wide[f"Q{Q_TAIL}_true"].to_numpy(float)
        qp = wide[f"Q{Q_TAIL}_pred"].to_numpy(float)
        ok = np.isfinite(qt) & np.isfinite(qp) & (qt > 0)
        err = np.where(ok, np.abs(qp - qt) / np.maximum(qt, 1e-8) * 100.0, np.nan)
        err = np.clip(err, 0, 400)
        tmp = wide[["station_id", "lat", "lon"]].copy()
        tmp["abs_rel_err_q100"] = err
        tmp["experiment"] = exp
        tmp["model"] = model
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    merged = pd.concat(rows, ignore_index=True)
    # Take median across PUB/PUR per station to get one value per station
    out = (merged.groupby("station_id", as_index=False)
           .agg(lat=("lat", "first"), lon=("lon", "first"),
                abs_rel_err_q100=("abs_rel_err_q100", "median"),
                model=("model", "first")))
    return out


def _assign_climate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["climate_zone"] = "Unknown"
    if not CLIMATE_SHP.exists():
        LOG.warning("Climate SHP not found: %s", CLIMATE_SHP)
        return out
    cz = gpd.read_file(CLIMATE_SHP)
    if "Name" not in cz.columns:
        LOG.warning("Climate SHP has no 'Name' column")
        return out
    if cz.crs is None:
        cz = cz.set_crs("EPSG:4326")
    elif str(cz.crs) != "EPSG:4326":
        cz = cz.to_crs("EPSG:4326")
    pts = out.dropna(subset=["lat", "lon"]).copy()
    gpts = gpd.GeoDataFrame(pts, geometry=[Point(xy) for xy in zip(pts["lon"], pts["lat"])],
                            crs="EPSG:4326")
    joined = gpd.sjoin(gpts, cz[["Name", "geometry"]], how="left", predicate="within")
    mapped = (joined[["station_id", "Name"]]
              .drop_duplicates("station_id")
              .rename(columns={"Name": "climate_zone"}))
    mapped["climate_zone"] = mapped["climate_zone"].fillna("Unknown")
    out = out.drop(columns=["climate_zone"], errors="ignore").merge(mapped, on="station_id", how="left")
    out["climate_zone"] = out["climate_zone"].fillna("Unknown")
    return out


def _cdf_data() -> pd.DataFrame:
    """Station-level Q100 abs_rel_err per model × experiment — for CDF."""
    rows = []
    for model in ["RF", "ANN"]:
        tag_dir, fn_prefix = MODEL_META[model]
        for exp in ["PUB", "PUR"]:
            raw = _load_raw(model, exp)
            if raw.empty:
                continue
            wide = _aggregate(raw)
            if wide.empty:
                continue
            qt = wide[f"Q{Q_TAIL}_true"].to_numpy(float)
            qp = wide[f"Q{Q_TAIL}_pred"].to_numpy(float)
            ok = np.isfinite(qt) & np.isfinite(qp) & (qt > 0)
            err = np.abs(qp[ok] - qt[ok]) / np.maximum(qt[ok], 1e-8) * 100.0
            err = np.clip(err, 0, 400)
            for v in err:
                rows.append({"model": model, "experiment": exp, "abs_rel_err": float(v)})
    return pd.DataFrame(rows)


def _pub_station_error(model: str, nc_coords: pd.DataFrame,
                       basin_assign: pd.DataFrame) -> pd.DataFrame:
    """Per-station Q100 absolute relative error for PUB only."""
    raw = _load_raw(model, "PUB")
    if raw.empty:
        return pd.DataFrame()
    wide = _aggregate(raw)
    if wide.empty:
        return pd.DataFrame()
    wide = _attach_coords(wide, nc_coords, basin_assign)
    qt  = wide[f"Q{Q_TAIL}_true"].to_numpy(float)
    qp  = wide[f"Q{Q_TAIL}_pred"].to_numpy(float)
    ok  = np.isfinite(qt) & np.isfinite(qp) & (qt > 0)
    err = np.where(ok, np.abs(qp - qt) / np.maximum(qt, 1e-8) * 100.0, np.nan)
    out = wide[["station_id", "lat", "lon"]].copy()
    out["abs_rel_err_q100"] = np.clip(err, 0, 300)
    LOG.info("  %s PUB station errors: %d stations", model, int(ok.sum()))
    return out.dropna(subset=["lat", "lon"])


def _pub_basin_nse(model: str, basin_assign: pd.DataFrame,
                   nc_coords: pd.DataFrame,
                   common_regions: set = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Q100 NSE per PUR basin using PUB test stations."""
    raw = _load_raw(model, "PUB")
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["station_id", "lat", "lon"])
    wide = _aggregate(raw)
    if wide.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["station_id", "lat", "lon"])
    wide = _attach_coords(wide, nc_coords, basin_assign)
    pts  = wide[["station_id", "lat", "lon"]].drop_duplicates("station_id")

    bsub  = basin_assign[["station_id", "basin_label", "continent", "HYBAS_ID"]].drop_duplicates("station_id")
    if common_regions:
        bsub = bsub[bsub["basin_label"].isin(common_regions)]
    wide2 = wide.merge(bsub, on="station_id", how="left").dropna(subset=["basin_label"])

    rows = []
    for basin, g in wide2.groupby("basin_label", observed=True):
        qt = g[f"Q{Q_TAIL}_true"].to_numpy(float)
        qp = g[f"Q{Q_TAIL}_pred"].to_numpy(float)
        nse = _nse(qt, qp)
        rows.append({"basin_label": str(basin),
                     "n_stations":  int(g["station_id"].nunique()),
                     "NSE_Q100":    nse})

    basin_df = pd.DataFrame(rows)
    if not basin_df.empty:
        meta = basin_assign[["basin_label", "continent", "HYBAS_ID"]].drop_duplicates("basin_label")
        basin_df = basin_df.merge(meta, on="basin_label", how="left")
    LOG.info("  %s PUB basin NSE: %d basins", model, len(basin_df))
    return basin_df, pts


def _pub_nse_by_climate(model: str, nc_coords: pd.DataFrame) -> dict:
    """Q100 NSE per climate zone for PUB stations."""
    raw = _load_raw(model, "PUB")
    if raw.empty:
        return {}
    wide = _aggregate(raw)
    if wide.empty:
        return {}
    wide = _attach_coords(wide, nc_coords, pd.DataFrame(columns=["station_id"]))
    wide = _assign_climate(wide)
    out = {}
    for zone in CLIMATE_ORDER:
        grp = wide[wide["climate_zone"] == zone]
        if len(grp) < 3:
            out[zone] = np.nan
            continue
        qt = grp[f"Q{Q_TAIL}_true"].to_numpy(float)
        qp = grp[f"Q{Q_TAIL}_pred"].to_numpy(float)
        out[zone] = _nse(qt, qp)
    return out


def _pub_err_by_climate(model: str, nc_coords: pd.DataFrame) -> dict:
    """Median Q100 abs_rel_err per climate zone for PUB stations."""
    raw = _load_raw(model, "PUB")
    if raw.empty:
        return {}
    wide = _aggregate(raw)
    if wide.empty:
        return {}
    wide = _attach_coords(wide, nc_coords, pd.DataFrame(columns=["station_id"]))
    wide = _assign_climate(wide)
    qt  = wide[f"Q{Q_TAIL}_true"].to_numpy(float)
    qp  = wide[f"Q{Q_TAIL}_pred"].to_numpy(float)
    ok  = np.isfinite(qt) & np.isfinite(qp) & (qt > 0)
    err = np.where(ok, np.abs(qp - qt) / np.maximum(qt, 1e-8) * 100.0, np.nan)
    wide["_err"] = np.clip(err, 0, 300)
    out = {}
    for zone in CLIMATE_ORDER:
        grp = wide[wide["climate_zone"] == zone]["_err"].dropna()
        out[zone] = float(np.nanmedian(grp)) if len(grp) >= 3 else np.nan
    return out


def _delta_nse_data(line_df: pd.DataFrame,
                    models: list[str] | None = None) -> pd.DataFrame:
    """ΔNSE = PUB_mean − PUR_mean per model × return_period."""
    if models is None:
        models = ["RF", "ANN"]
    rows = []
    for model in models:
        for t in RETURN_PERIODS:
            pub = line_df[(line_df["model"] == model) &
                          (line_df["experiment"] == "PUB") &
                          (line_df["return_period"] == t)]["NSE_mean"]
            pur = line_df[(line_df["model"] == model) &
                          (line_df["experiment"] == "PUR") &
                          (line_df["return_period"] == t)]["NSE_mean"]
            if pub.empty or pur.empty:
                continue
            rows.append({"model": model, "return_period": t,
                         "delta_NSE": float(pub.iloc[0]) - float(pur.iloc[0])})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plot sub-functions
# ─────────────────────────────────────────────────────────────────────────────
def _draw_map(ax, world_gdf: gpd.GeoDataFrame,
              perf_gdf: gpd.GeoDataFrame,
              pts: pd.DataFrame,
              title: str, letter: str,
              climate_nse: dict = None, bar_color: str = None,
              letter_x: float = -125.0, title_x: float = -115.2) -> None:
    """Draw choropleth map on ax. Returns ScalarMappable for shared colorbar."""
    # World basemap
    if not world_gdf.empty:
        world_gdf.plot(ax=ax, facecolor="#e9e9e9", edgecolor="#151515",
                       linewidth=0.30, alpha=1.0, zorder=1)

    # PUR basins colored by NSE
    if not perf_gdf.empty and "NSE_Q100" in perf_gdf.columns:
        norm = mcolors.Normalize(vmin=MAP_VMIN, vmax=MAP_VMAX)
        perf_gdf.plot(ax=ax, column="NSE_Q100", cmap=MAP_CMAP,
                      norm=norm, edgecolor="#101010", linewidth=1.05,
                      alpha=0.98, zorder=3,
                      missing_kwds={"color": "#d0d0d0", "edgecolor": "#000000"})
        '''
        # Add per-region rectangular boundary boxes to improve visual grouping.
        b = perf_gdf.geometry.bounds
        for minx, miny, maxx, maxy in b.itertuples(index=False, name=None):
            if not np.isfinite([minx, miny, maxx, maxy]).all():
                continue
            w = max(float(maxx - minx), 0.0)
            h = max(float(maxy - miny), 0.0)
            if w <= 0 or h <= 0:
                continue
            ax.add_patch(mpatches.Rectangle(
                (float(minx), float(miny)), w, h,
                fill=False, edgecolor="#1B1B1B", linewidth=0.90,
                alpha=0.85, zorder=4
            ))
        '''
    else:
        ax.text(0.5, 0.5, "No prediction data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=20, color="#888888")

    ax.set_xlim(-145, 180)
    ax.set_ylim(-60, 85)
    ax.grid(False)
    ax.set_axis_off()

    ax.text(title_x, -45, title, fontsize=32, fontweight="bold", ha="left")
    ax.text(letter_x, 1.03, letter, transform=ax.get_xaxis_transform(),
            fontsize=36, fontweight="bold", va="top", ha="left", zorder=10)

    if climate_nse is not None and bar_color is not None:
        # Put horizontal bar chart above Australia/Pacific
        ins = ax.inset_axes([0.82, 0.60, 0.18, 0.25])
        zones = CLIMATE_ORDER[::-1]
        vals = [climate_nse.get(z, np.nan) for z in zones]
        y = np.arange(len(zones))
        # Always draw bars rightward (use abs); annotate negatives in red with minus sign
        plot_vals = [abs(v) if np.isfinite(v) else 0.0 for v in vals]
        ins.barh(y, plot_vals, height=0.6, color=bar_color, alpha=0.85, edgecolor="white")
        ins.set_yticks(y)
        ins.set_yticklabels(zones, fontsize=16)
        ins.set_xticks([])
        ins.set_xlim(0, 0.65)
        ins.spines['top'].set_visible(False)
        ins.spines['right'].set_visible(False)
        ins.spines['bottom'].set_visible(False)
        ins.set_facecolor((1, 1, 1, 0.85))
        for i, v in enumerate(vals):
            if np.isfinite(v):
                txt_color = "#CC0000" if v < 0 else "black"
                ins.text(abs(v) + 0.02, i, f"{v:.2f}",
                         va='center', ha='left', fontsize=14, color=txt_color)


def _draw_nse_curve(ax, line_df: pd.DataFrame, model: str, letter: str) -> None:
    ax.axvspan(3.5, 5.5, color="#DDE8F5", alpha=0.55, zorder=0)
    ax.text(4.5, 0.98, "Extreme\ntail",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=18,
            color="#3A5B8C", fontstyle="italic", linespacing=1.3)

    x = np.arange(len(RETURN_PERIODS))
    all_y = []
    for exp, color, label in [("PUB", C_PUB, "PUB"), ("PUR", C_PUR, "PUR")]:
        sub = line_df[(line_df["model"] == model) & (line_df["experiment"] == exp)]
        if sub.empty:
            continue
        y = np.array([sub[sub["return_period"] == t]["NSE_mean"].iloc[0]
                      if not sub[sub["return_period"] == t].empty else np.nan
                      for t in RETURN_PERIODS], dtype=float)
        e = np.nan_to_num(
            np.array([sub[sub["return_period"] == t]["NSE_std"].iloc[0]
                      if not sub[sub["return_period"] == t].empty else np.nan
                      for t in RETURN_PERIODS], dtype=float), nan=0)
        ok = np.isfinite(y)
        if ok.sum() < 2:
            continue
        ax.plot(x[ok], y[ok], color=color, lw=3.2, marker="o",
                ms=10, mec="white", mew=1.6, label=label, zorder=4)
        ax.fill_between(x[ok], y[ok] - e[ok], y[ok] + e[ok],
                        color=color, alpha=0.12, linewidth=0, zorder=3)
        all_y.extend(y[ok].tolist())

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{t}" for t in RETURN_PERIODS], fontsize=20)
    ax.set_ylabel("NSE", fontsize=24)
    ax.axhline(0, color="#BBBBBB", lw=0.9, ls=":")
    if all_y:
        lo = max(0.0, float(np.nanmin(all_y)) - 0.05)
        hi = min(1.0, float(np.nanmax(all_y)) + 0.07)
        ax.set_ylim(lo, hi)
    ax.legend(ncol=1, loc="upper right", handlelength=1.5, fontsize=20)
    ax.set_title(f"{model} — NSE by Return Period", fontweight="bold", fontsize=26)
    _letter(ax, letter)


def _draw_nse_combined(ax, line_df: pd.DataFrame,
                       primary_model: str = "RF",
                       letter: str = "c") -> None:
    """NSE curves: primary_model (solid circles) vs ANN (dashed squares)."""
    ax.axvspan(3.5, 5.5, color="#DDE8F5", alpha=0.55, zorder=0)
    ax.text(4.5, 0.98, "Extreme\ntail",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=18,
            color="#3A5B8C", fontstyle="italic", linespacing=1.3)

    x = np.arange(len(RETURN_PERIODS))
    all_y = []

    def _smooth_line_with_band(xv: np.ndarray, yv: np.ndarray, ev: np.ndarray):
        xs = np.linspace(float(xv.min()), float(xv.max()), 240)
        if len(xv) >= 3:
            ys = PchipInterpolator(xv, yv)(xs)
            es = PchipInterpolator(xv, ev)(xs)
        else:
            ys = np.interp(xs, xv, yv)
            es = np.interp(xs, xv, ev)
        es = np.maximum(es, 0.010)
        return xs, ys, es

    pm = primary_model
    specs = [
        (pm,    "PUB", C_PUB, 3.2, "o", None,   f"{pm}–PUB"),
        (pm,    "PUR", C_PUR, 3.2, "o", None,   f"{pm}–PUR"),
        ("ANN", "PUB", C_PUB, 2.8, "s", (6, 3), "ANN–PUB"),
        ("ANN", "PUR", C_PUR, 2.8, "s", (6, 3), "ANN–PUR"),
    ]
    for mdl, exp, color, lw, marker, dash, label in specs:
        sub = line_df[(line_df["model"] == mdl) & (line_df["experiment"] == exp)]
        if sub.empty:
            continue
        y = np.array([sub[sub["return_period"] == t]["NSE_mean"].iloc[0]
                      if not sub[sub["return_period"] == t].empty else np.nan
                      for t in RETURN_PERIODS], dtype=float)
        e = np.nan_to_num(
            np.array([sub[sub["return_period"] == t]["NSE_std"].iloc[0]
                      if not sub[sub["return_period"] == t].empty else np.nan
                      for t in RETURN_PERIODS], dtype=float), nan=0)
        ok = np.isfinite(y)
        if ok.sum() < 2:
            continue
        xv = x[ok].astype(float)
        yv = y[ok].astype(float)
        ev = np.nan_to_num(e[ok].astype(float), nan=0.0)
        xs, ys, es = _smooth_line_with_band(xv, yv, ev)

        ln, = ax.plot(xs, ys, color=color, lw=lw, marker=marker,
                      markevery=40, ms=8, mec="white", mew=1.2,
                      label=label, zorder=4)
        if dash:
            ln.set_dashes(dash)
        ax.fill_between(xs, ys - es, ys + es,
                        color=color, alpha=0.16 if mdl == "RF" else 0.12,
                        linewidth=0, zorder=2)
        all_y.extend(ys.tolist())

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{t}" for t in RETURN_PERIODS], fontsize=20)
    ax.set_ylabel("NSE", fontsize=24)
    ax.axhline(0, color="#BBBBBB", lw=0.9, ls=":")
    if all_y:
        lo = max(0.0, float(np.nanmin(all_y)) - 0.05)
        hi = min(1.0, float(np.nanmax(all_y)) + 0.07)
        ax.set_ylim(lo, hi)
    ax.legend(ncol=2, loc="lower left", handlelength=1.8,
              columnspacing=0.8, fontsize=18)
    _letter(ax, letter)


def _draw_climate_box(ax, df: pd.DataFrame) -> None:
    """Grouped boxplot of Q100 abs_rel_err by climate zone."""
    if df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        _letter(ax, "e")
        return

    valid = df[df["climate_zone"].isin(CLIMATE_ORDER)].copy()
    valid["abs_rel_err_q100"] = valid["abs_rel_err_q100"].clip(0, 200)

    cen = np.arange(len(CLIMATE_ORDER)) * 1.4
    w = 0.50
    for mi, (mdl, color) in enumerate([("RF", C_RF), ("ANN", C_ANN)]):
        pos_offset = -w / 2 if mi == 0 else w / 2
        data, positions, ns = [], [], []
        for zi, zone in enumerate(CLIMATE_ORDER):
            vals = (valid[(valid["climate_zone"] == zone) & (valid["model"] == mdl)]
                    ["abs_rel_err_q100"].dropna().to_numpy(float))
            if len(vals) >= 5:
                data.append(vals)
                positions.append(float(cen[zi]) + pos_offset)
                ns.append(len(vals))
        if not data:
            continue
        bp = ax.boxplot(data, positions=positions, widths=0.42,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="#111111", lw=1.8),
                        whiskerprops=dict(lw=1.2),
                        boxprops=dict(lw=1.2),
                        capprops=dict(lw=1.2))
        for b in bp["boxes"]:
            b.set_facecolor(color)
            b.set_alpha(0.85)
        # Annotate n
        for pos, n in zip(positions, ns):
            ax.text(pos, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 200,
                    f"n={n}", ha="center", va="bottom", fontsize=8, color="#555555")

    ax.set_xticks(cen)
    ax.set_xticklabels(CLIMATE_ORDER, rotation=15, ha="right", fontsize=12)
    ax.set_ylabel("Q100 Abs. Relative Error (%)", fontsize=13)
    ax.set_ylim(0, 210)
    ax.set_title("Climate-Zone Q100 Error", fontweight="bold", fontsize=15)
    ax.legend(handles=[mpatches.Patch(facecolor=C_RF, label="RF"),
                       mpatches.Patch(facecolor=C_ANN, label="ANN")],
              loc="upper right", fontsize=12)
    _letter(ax, "e")


def _draw_cdf(ax, cdf_df: pd.DataFrame) -> None:
    """Station-level Q100 absolute relative error CDF, 4 lines."""
    specs = [
        ("RF",  "PUB", C_PUB, 2.4, None,   "RF–PUB"),
        ("RF",  "PUR", C_PUR, 2.4, None,   "RF–PUR"),
        ("ANN", "PUB", C_PUB, 2.0, (6, 3), "ANN–PUB"),
        ("ANN", "PUR", C_PUR, 2.0, (6, 3), "ANN–PUR"),
    ]
    has = False
    for mdl, exp, color, lw, dash, label in specs:
        sub = cdf_df[(cdf_df["model"] == mdl) & (cdf_df["experiment"] == exp)]
        if sub.empty:
            continue
        vals = sub["abs_rel_err"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        has = True
        x, y = _ecdf(vals)
        med = float(np.nanmedian(vals))
        ln, = ax.plot(x, y, color=color, lw=lw, label=f"{label} (med={med:.0f}%)")
        if dash:
            ln.set_dashes(dash)

    for v in [50, 100]:
        ax.axvline(v, color="#CCCCCC", lw=0.9, ls="--", zorder=1)
        ax.text(v + 3, 0.02, f"{v}%", fontsize=9, color="#AAAAAA", va="bottom")

    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Absolute Relative Error (%)", fontsize=13)
    ax.set_ylabel("Cumulative Probability", fontsize=13)
    ax.set_title("Q100 Station Error CDF", fontweight="bold", fontsize=15)
    if has:
        ax.legend(loc="lower right", fontsize=11, handlelength=2.0)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="#888888")
    _letter(ax, "f")


def _draw_pub_scatter_map(ax, world_gdf: gpd.GeoDataFrame,
                          err_df: pd.DataFrame,
                          title: str, letter: str,
                          climate_err: dict = None,
                          bar_color: str = None,
                          letter_x: float = -125.0,
                          title_x: float = -115.2) -> None:
    """Scatter map: PUB stations coloured by Q100 abs_rel_error (%)."""
    if not world_gdf.empty:
        world_gdf.plot(ax=ax, facecolor="#e9e9e9", edgecolor="#151515",
                       linewidth=0.30, alpha=1.0, zorder=1)

    if not err_df.empty:
        sp   = err_df.dropna(subset=["lat", "lon", "abs_rel_err_q100"])
        norm = mcolors.Normalize(vmin=ERR_VMIN, vmax=ERR_VMAX)
        ax.scatter(sp["lon"], sp["lat"], s=20,
                   c=sp["abs_rel_err_q100"], cmap=ERR_CMAP,
                   norm=norm, alpha=0.75, linewidths=0,
                   zorder=4, rasterized=True)
    else:
        ax.text(0.5, 0.5, "No prediction data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=20, color="#888888")

    ax.set_xlim(-145, 180)
    ax.set_ylim(-60, 85)
    ax.grid(False)
    ax.set_axis_off()
    ax.text(title_x, -45, title, fontsize=32, fontweight="bold", ha="left")
    ax.text(letter_x, 1.03, letter, transform=ax.get_xaxis_transform(),
            fontsize=36, fontweight="bold", va="top", ha="left", zorder=10)

    if climate_err is not None and bar_color is not None:
        ins   = ax.inset_axes([0.82, 0.60, 0.18, 0.25])
        zones = CLIMATE_ORDER[::-1]
        vals  = [climate_err.get(z, np.nan) for z in zones]
        y     = np.arange(len(zones))
        ins.barh(y, vals, height=0.6, color=bar_color, alpha=0.85, edgecolor="white")
        ins.set_yticks(y)
        ins.set_yticklabels(zones, fontsize=16)
        ins.set_xticks([])
        ins.set_xlim(0, 120)
        ins.spines["top"].set_visible(False)
        ins.spines["right"].set_visible(False)
        ins.spines["bottom"].set_visible(False)
        ins.set_facecolor((1, 1, 1, 0.85))
        ins.set_title("Med. Err%", fontsize=14, pad=3)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ins.text(v + 3, i, f"{v:.0f}%", va="center", ha="left", fontsize=14)


def _add_colorbar(fig, ax_list, cmap, vmin, vmax, label, fontsize=22) -> None:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin, vmax))
    sm.set_array([])
    for ax_m in ax_list:
        cbar_ax = ax_m.inset_axes([0.47, 0.05, 0.28, 0.035])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cb.set_ticks(np.linspace(vmin, vmax, 6))
        cb.set_label(label, fontsize=fontsize, labelpad=8)
        cb.ax.tick_params(labelsize=18, length=5, width=1.2, color="#444444")
        cb.outline.set_linewidth(1.2)
        cb.outline.set_edgecolor("#444444")
        cbar_ax.set_facecolor((1, 1, 1, 0.84))


def _draw_delta_nse(ax, delta_df: pd.DataFrame,
                    primary_model: str = "RF",
                    letter: str = "d") -> None:
    """ΔNSE (PUB−PUR) bar chart by return period, grouped by model."""
    if delta_df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        _letter(ax, letter, y=1.10)
        return

    x = np.arange(len(RETURN_PERIODS))
    w = 0.38
    for mi, (mdl, color) in enumerate([(primary_model, C_RF), ("ANN", C_ANN)]):
        sub = delta_df[delta_df["model"] == mdl]
        vals = [float(sub[sub["return_period"] == t]["delta_NSE"].iloc[0])
                if not sub[sub["return_period"] == t].empty else np.nan
                for t in RETURN_PERIODS]
        offset = -w / 2 if mi == 0 else w / 2
        bars = ax.bar(x + offset, vals, width=w, color=color,
                      alpha=0.85, label=mdl, edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=16)

    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="-")
    ax.axvspan(3.5, 5.5, color="#DDE8F5", alpha=0.45, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{t}" for t in RETURN_PERIODS], fontsize=20)
    ax.set_ylabel("ΔNSE (PUB − PUR)", fontsize=24)
    ax.set_ylim(top=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 1.05), fontsize=20, borderaxespad=0.)
    _letter(ax, letter, y=1.15)


# ─────────────────────────────────────────────────────────────────────────────
# Shared figure helpers (used by main)
# ─────────────────────────────────────────────────────────────────────────────
def _make_map_gdf(region_metric: pd.DataFrame, region_labels: set[str],
                  model: str, pur_polys: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    empty = gpd.GeoDataFrame(
        columns=list(pur_polys.columns) + ["NSE_Q100", "n_stations"],
        geometry="geometry", crs="EPSG:4326")
    if not region_labels:
        return empty
    base = pur_polys[pur_polys["basin_label"].isin(region_labels)].copy()
    if base.empty:
        LOG.warning("%s map: no polygons matched PUR region labels", model)
        return empty
    if region_metric.empty:
        base["NSE_Q100"] = np.nan
        base["n_stations"] = np.nan
        return base
    out = base.merge(region_metric[["basin_label", "n_stations", "NSE_Q100"]],
                     on="basin_label", how="left")
    missing = int(out["NSE_Q100"].isna().sum())
    if missing > 0:
        LOG.warning("%s map: %d PUR regions missing valid NSE", model, missing)
    return out


def _make_pub_map_gdf(basin_metric: pd.DataFrame, model: str,
                      pur_polys: gpd.GeoDataFrame,
                      common_set: set[str]) -> gpd.GeoDataFrame:
    empty = gpd.GeoDataFrame(
        columns=list(pur_polys.columns) + ["NSE_Q100", "n_stations"],
        geometry="geometry", crs="EPSG:4326")
    if basin_metric.empty:
        return empty
    filtered = basin_metric[basin_metric["basin_label"].isin(common_set)].copy()
    no_data  = common_set - set(filtered["basin_label"].tolist())
    if no_data:
        LOG.info("%s PUB choropleth: %d PUR basins have no PUB stations — shown as grey",
                 model, len(no_data))
    base = pur_polys[pur_polys["basin_label"].isin(common_set)].copy()
    if base.empty:
        LOG.warning("%s PUB choropleth: no polygons for common_set", model)
        return empty
    out = base.merge(filtered[["basin_label", "n_stations", "NSE_Q100"]],
                     on="basin_label", how="left")
    LOG.info("%s PUB choropleth: %d basins total, %d with NSE, %d grey",
             model, len(out), int(out["NSE_Q100"].notna().sum()),
             int(out["NSE_Q100"].isna().sum()))
    return out


def _build_2x2_fig():
    fig = plt.figure(figsize=(24, 13))
    gs  = gridspec.GridSpec(
        2, 2, height_ratios=[1.0, 1.0], width_ratios=[4.0, 1.55],
        hspace=0.15, wspace=0.05,
        left=0.01, right=0.97, top=0.97, bottom=0.05,
    )
    return fig, fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), \
                fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _setup_style()
    LOG.info("=" * 60)
    LOG.info("%s - start", TAG)

    # ── Load static data ──────────────────────────────────────────────────────
    # Keep_default_na=False is critical here: continent code "NA" means
    # North America, not missing value.
    basin_assign = pd.read_csv(BASIN_CSV, keep_default_na=False)
    basin_assign["station_id"] = basin_assign["station_id"].astype(str).str.strip()

    # Drop invalid basin assignments early to avoid fake continent "NAN"
    # and shapefile lookup failures.
    basin_assign["continent"] = basin_assign["continent"].where(
        basin_assign["continent"].notna(), np.nan
    )
    basin_assign["continent"] = basin_assign["continent"].astype(str).str.strip().str.upper()
    basin_assign["continent"] = basin_assign["continent"].replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    basin_assign["HYBAS_ID"] = pd.to_numeric(basin_assign["HYBAS_ID"], errors="coerce")

    n_before = len(basin_assign)
    basin_assign = basin_assign.dropna(subset=["station_id", "continent", "HYBAS_ID"]).copy()
    basin_assign["HYBAS_ID"] = basin_assign["HYBAS_ID"].astype("int64")
    basin_assign = basin_assign.drop_duplicates("station_id")
    n_drop_invalid = n_before - len(basin_assign)
    if n_drop_invalid > 0:
        LOG.info("Dropped %d stations with invalid basin assignment (continent/HYBAS_ID)", n_drop_invalid)

    LOG.info("Basin assignment: %d stations", len(basin_assign))

    nc_coords = _load_nc_coords()
    world_gdf = _load_world_basemap()
    pur_polys = _load_pur_basin_polys(basin_assign)

    # ── Per-model data (PUR) ─────────────────────────────────────────────────
    model_basin:   dict[str, pd.DataFrame]      = {}
    model_pts:     dict[str, pd.DataFrame]      = {}
    model_regions: dict[str, set[str]]          = {}
    model_climate: dict[str, dict]              = {}
    model_line_df: dict[str, pd.DataFrame]      = {}
    model_pub_err: dict[str, pd.DataFrame]      = {}
    model_pub_basin: dict[str, pd.DataFrame]    = {}
    model_pub_pts:   dict[str, pd.DataFrame]    = {}
    model_pub_nse_climate: dict[str, dict]      = {}
    model_pub_err_climate: dict[str, dict]      = {}

    for mdl in MODEL_META:
        LOG.info("Loading data for %s …", mdl)
        basin, pts, regions = _pur_region_nse(mdl, basin_assign, nc_coords)
        model_basin[mdl]   = basin
        model_pts[mdl]     = pts
        model_regions[mdl] = regions
        model_climate[mdl] = _nse_by_climate(mdl, nc_coords)
        model_line_df[mdl] = _seed_nse_curve(mdl)

        LOG.info("  %s PUB analysis …", mdl)
        model_pub_err[mdl]         = _pub_station_error(mdl, nc_coords, basin_assign)
        model_pub_nse_climate[mdl] = _pub_nse_by_climate(mdl, nc_coords)
        model_pub_err_climate[mdl] = _pub_err_by_climate(mdl, nc_coords)
        pub_basin, pub_pts         = _pub_basin_nse(mdl, basin_assign, nc_coords)
        model_pub_basin[mdl]       = pub_basin
        model_pub_pts[mdl]         = pub_pts

    ann_regions = model_regions["ANN"]

    # ── Save NSE curve tables ─────────────────────────────────────────────────
    all_line_df = pd.concat(list(model_line_df.values()), ignore_index=True)
    all_line_df.to_csv(OUT_DATA / "step1_nse_curves.csv", index=False)

    for stale_name in ["step1_cdf_data.csv", "step1_climate_data.csv"]:
        stale_fp = OUT_DATA / stale_name
        if stale_fp.exists():
            stale_fp.unlink()
            LOG.info("Removed stale artifact: %s", stale_fp)

    # ── Generate figures for each comparison pair ─────────────────────────────
    # Primary models to compare against ANN.
    for primary in ["RF", "XGBoost", "SVM"]:
        if primary not in MODEL_META:
            LOG.warning("Skipping %s: not in MODEL_META", primary)
            continue

        pm_color  = C_RF   # all primary models use same gray for style consistency
        pm_lx     = _MAP_LETTER_X[primary]
        pm_tx     = _MAP_TITLE_X[primary]
        ann_lx    = _MAP_LETTER_X["ANN"]
        ann_tx    = _MAP_TITLE_X["ANN"]
        tag       = primary.lower()   # "rf" / "xgboost" / "svm"

        LOG.info("─── Generating figures: %s vs ANN ───", primary)

        # PUR common regions (intersection of primary model & ANN)
        pm_regions  = model_regions[primary]
        common_set  = pm_regions & ann_regions
        pm_only     = sorted(pm_regions - common_set)
        ann_only    = sorted(ann_regions - common_set)
        if pm_only or ann_only:
            LOG.warning("  Region match: %s-only=%d, ANN-only=%d removed",
                        primary, len(pm_only), len(ann_only))

        # Filter per-model basin tables to common regions
        pm_basin = (model_basin[primary][model_basin[primary]["basin_label"].isin(common_set)].copy()
                    if not model_basin[primary].empty else model_basin[primary])
        ann_basin = (model_basin["ANN"][model_basin["ANN"]["basin_label"].isin(common_set)].copy()
                     if not model_basin["ANN"].empty else model_basin["ANN"])

        # ── Region-wise NSE comparison: primary vs ANN ───────────────────────
        if not pm_basin.empty and not ann_basin.empty:
            cmp_df = pm_basin[["basin_label", "NSE_Q100"]].merge(
                ann_basin[["basin_label", "NSE_Q100"]],
                on="basin_label", suffixes=(f"_{primary}", "_ANN")
            ).dropna(subset=[f"NSE_Q100_{primary}", "NSE_Q100_ANN"])
            n_total      = len(cmp_df)
            n_pm_better  = int((cmp_df[f"NSE_Q100_{primary}"] > cmp_df["NSE_Q100_ANN"]).sum())
            n_ann_better = n_total - n_pm_better
            LOG.info(
                "  Region-wise NSE (%s vs ANN, PUR): "
                "%d / %d regions where %s > ANN (%.1f%%); "
                "%d / %d regions where ANN > %s (%.1f%%)",
                primary,
                n_pm_better,  n_total, primary, 100.0 * n_pm_better  / n_total if n_total else 0.0,
                n_ann_better, n_total, primary, 100.0 * n_ann_better / n_total if n_total else 0.0,
            )
            cmp_df.to_csv(OUT_DATA / f"step1_region_nse_comparison_{tag}_vs_ann.csv",
                          index=False)

        pm_map  = _make_map_gdf(pm_basin,  common_set, primary, pur_polys)
        ann_map = _make_map_gdf(ann_basin, common_set, "ANN",   pur_polys)
        LOG.info("  %s PUR map: %d | ANN PUR map: %d", primary, len(pm_map), len(ann_map))

        # Restrict PUB basin NSE to the same common_set
        pm_pub_basin_c  = (model_pub_basin[primary][
            model_pub_basin[primary]["basin_label"].isin(common_set)].copy()
            if not model_pub_basin[primary].empty else model_pub_basin[primary])
        ann_pub_basin_c = (model_pub_basin["ANN"][
            model_pub_basin["ANN"]["basin_label"].isin(common_set)].copy()
            if not model_pub_basin["ANN"].empty else model_pub_basin["ANN"])

        pm_pub_map  = _make_pub_map_gdf(pm_pub_basin_c,  primary, pur_polys, common_set)
        ann_pub_map = _make_pub_map_gdf(ann_pub_basin_c, "ANN",   pur_polys, common_set)

        # NSE curves & ΔNSE for this pair
        line_df  = pd.concat([model_line_df[primary], model_line_df["ANN"]], ignore_index=True)
        delta_df = _delta_nse_data(line_df, models=[primary, "ANN"])

        # Save per-comparison tables
        tag_lower = primary.lower()
        if not pm_basin.empty:
            pm_basin.to_csv(OUT_DATA / f"step1_basin_nse_{tag_lower}.csv", index=False)
        if not ann_basin.empty:
            ann_basin.to_csv(OUT_DATA / f"step1_basin_nse_ann_for_{tag_lower}.csv", index=False)
        if not pm_pub_basin_c.empty:
            pm_pub_basin_c.to_csv(OUT_DATA / f"step1_pub_basin_nse_{tag_lower}.csv", index=False)
        if not ann_pub_basin_c.empty:
            ann_pub_basin_c.to_csv(OUT_DATA / f"step1_pub_basin_nse_ann_for_{tag_lower}.csv", index=False)
        if not model_pub_err[primary].empty:
            model_pub_err[primary].to_csv(
                OUT_DATA / f"step1_pub_station_err_{tag_lower}.csv", index=False)
        if not model_pub_err["ANN"].empty:
            model_pub_err["ANN"].to_csv(
                OUT_DATA / f"step1_pub_station_err_ann_for_{tag_lower}.csv", index=False)
        delta_df.to_csv(OUT_DATA / f"step1_delta_nse_{tag_lower}.csv", index=False)

        # ── Figure A: PUR choropleth ───────────────────────────────────────────
        LOG.info("  Drawing PUR choropleth …")
        figA, axA_pm, axA_nse, axA_ann, axA_delta = _build_2x2_fig()
        _draw_map(axA_pm,  world_gdf, pm_map,  model_pts[primary],
                  primary, "a", model_climate[primary], pm_color,
                  letter_x=pm_lx, title_x=pm_tx)
        _draw_map(axA_ann, world_gdf, ann_map, model_pts["ANN"],
                  "ANN", "b", model_climate["ANN"], C_ANN,
                  letter_x=ann_lx, title_x=ann_tx)
        _draw_nse_combined(axA_nse,   line_df, primary_model=primary, letter="c")
        _draw_delta_nse(axA_delta,    delta_df, primary_model=primary, letter="d")
        _add_colorbar(figA, [axA_pm, axA_ann], MAP_CMAP, MAP_VMIN, MAP_VMAX, "Q100 NSE")
        fpA = OUT_FIG / f"fig_step1_ml_limits_{tag_lower}.png"
        figA.savefig(fpA, dpi=300)
        plt.close(figA)
        LOG.info("  Saved: %s", fpA)

        # ── Figure B: PUB scatter (per-station error) ──────────────────────────
        LOG.info("  Drawing PUB scatter …")
        figB, axB_pm, axB_nse, axB_ann, axB_delta = _build_2x2_fig()
        _draw_pub_scatter_map(axB_pm,  world_gdf, model_pub_err[primary],
                              f"{primary}  (PUB)", "a",
                              model_pub_err_climate[primary], pm_color,
                              letter_x=pm_lx, title_x=pm_tx)
        _draw_pub_scatter_map(axB_ann, world_gdf, model_pub_err["ANN"],
                              "ANN  (PUB)", "b",
                              model_pub_err_climate["ANN"], C_ANN,
                              letter_x=ann_lx, title_x=ann_tx)
        _draw_nse_combined(axB_nse,  line_df, primary_model=primary, letter="c")
        _draw_delta_nse(axB_delta,   delta_df, primary_model=primary, letter="d")
        _add_colorbar(figB, [axB_pm, axB_ann], ERR_CMAP, ERR_VMIN, ERR_VMAX,
                      "Q100 Abs. Rel. Error (%)", fontsize=20)
        fpB = OUT_FIG / f"fig_step1_pub_scatter_{tag_lower}.png"
        figB.savefig(fpB, dpi=300)
        plt.close(figB)
        LOG.info("  Saved: %s", fpB)

        # ── Figure C: PUB choropleth (basin-level NSE) ─────────────────────────
        LOG.info("  Drawing PUB choropleth …")
        figC, axC_pm, axC_nse, axC_ann, axC_delta = _build_2x2_fig()
        _draw_map(axC_pm,  world_gdf, pm_pub_map,  model_pub_pts[primary],
                  f"{primary}  (PUB)", "a",
                  model_pub_nse_climate[primary], pm_color,
                  letter_x=pm_lx, title_x=pm_tx)
        _draw_map(axC_ann, world_gdf, ann_pub_map, model_pub_pts["ANN"],
                  "ANN  (PUB)", "b",
                  model_pub_nse_climate["ANN"], C_ANN,
                  letter_x=ann_lx, title_x=ann_tx)
        _draw_nse_combined(axC_nse,  line_df, primary_model=primary, letter="c")
        _draw_delta_nse(axC_delta,   delta_df, primary_model=primary, letter="d")
        _add_colorbar(figC, [axC_pm, axC_ann], MAP_CMAP, MAP_VMIN, MAP_VMAX, "Q100 NSE")
        fpC = OUT_FIG / f"fig_step1_pub_choropleth_{tag_lower}.png"
        figC.savefig(fpC, dpi=300)
        plt.close(figC)
        LOG.info("  Saved: %s", fpC)

    LOG.info("Done. Data → %s", OUT_DATA)


if __name__ == "__main__":
    main()