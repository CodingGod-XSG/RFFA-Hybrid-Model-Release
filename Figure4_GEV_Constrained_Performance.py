# -*- coding: utf-8 -*-
"""Figure4_GEV_Constrained_Performance.py

Reproduces Figure 4 - Physical consistency and predictive performance of
ANN and GEV-constrained models. Panels: (a) spatial distribution of ANN
monotonicity violation rate across PUR holdout regions; (b) per-pair
monotonicity violation rate for adjacent quantile pairs under PUB/PUR;
(c) ANN monotonicity violation rate by Koeppen climate zone; (d) mean NSE
across Q50/Q100 under PUR for all models, and NSE heatmap across return
periods/models under PUB/PUR.

REWRITTEN VERSION:
  - a) Global map of ANN Violation Rate in PUR (no title, boundary boxes, styled colorbar).
  - b) Bar chart of ANN Violation Rate (PUB vs PUR) by Return Period (value labels, academic style).
  - c) Bar chart of ANN Violation Rate (PUB vs PUR) by Climate Zone (value labels, academic style).
  - d) Heatmap of NSE comparing models (Columns) and Return Periods / Experiments (Rows).
      Reordered: PUB block (top), PUR block (bottom). Diverging RwB colormap, black boxes.
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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from shapely.geometry import Point

warnings.filterwarnings("ignore")
matplotlib.rcParams["axes.formatter.limits"] = (-3, 4)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

PROC     = DATA_PROCEED
FIG_ROOT = FIGURE_ROOT

TAG      = "Figure4_GEV_Constrained_Performance"
OUT_DATA = stage_dir(DATA_PROCEED, TAG)
OUT_FIG  = stage_dir(FIGURE_ROOT, TAG)

BASIN_CSV = DATA_PROCEED / "05_PUR_Basin_Select" / "station_basin_assignment.csv"
RETAINED_BASIN_CSV = DATA_PROCEED / "05_PUR_Basin_Select" / "pur_retained_basins.csv"
HYDRO_ROOT  = DATA_RAW / "Hydrosheds"
HYBAS_LEVEL = 2
PUR_KEEP_N  = 15
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
ADJ_PAIRS = list(zip(RETURN_PERIODS[:-1], RETURN_PERIODS[1:]))

MODELS_ALL = [
    ("06_RF", "RF", "RF"),
    ("09_ANN", "ANN", "ANN_Single"),
    ("10_ANN_Joint", "ANN-Joint", "ANN_Direct"),
    ("11_GEV_NN", "GEV-NN", "GEV_NN_ST"),
    ("12_GEV_NN_MSE", "GEV-NN-MSE", "GEV_NN_MSEOnly"),
    ("13_GEV_NN_NLL", "GEV-NN-NLL", "GEV_NN_NLL"),
]

# Variants with XGBoost / SVM replacing RF in position 0
MODELS_XGB = [
    ("08_XGBoost", "XGBoost", "XGBoost"),
    ("09_ANN", "ANN", "ANN_Single"),
    ("10_ANN_Joint", "ANN-Joint", "ANN_Direct"),
    ("11_GEV_NN", "GEV-NN", "GEV_NN_ST"),
    ("12_GEV_NN_MSE", "GEV-NN-MSE", "GEV_NN_MSEOnly"),
    ("13_GEV_NN_NLL", "GEV-NN-NLL", "GEV_NN_NLL"),
]
MODELS_SVM = [
    ("07_SVM", "SVM", "SVM"),
    ("09_ANN", "ANN", "ANN_Single"),
    ("10_ANN_Joint", "ANN-Joint", "ANN_Direct"),
    ("11_GEV_NN", "GEV-NN", "GEV_NN_ST"),
    ("12_GEV_NN_MSE", "GEV-NN-MSE", "GEV_NN_MSEOnly"),
    ("13_GEV_NN_NLL", "GEV-NN-NLL", "GEV_NN_NLL"),
]

CLIMATE_ORDER = ["Tropical", "Arid", "Temperate", "Cold"]

C_PUB = "#6FAED6"
C_PUR = "#E8A4A4"

MAP_CMAP = "RdYlGn_r"
MAP_VMIN, MAP_VMAX = 0.0, 1.0

# Heatmap NSE colormap: diverging from red (lower) to blue (higher)
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "deep_rwb", ["#B2182B", "#F7F7F7", "#2166AC"], N=256
)

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
        "figure.dpi":         120,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })

def _letter(ax, txt: str, x: float = -0.07, y: float = 1.05) -> None:
    ax.text(x, y, txt, transform=ax.transAxes,
            fontsize=36, fontweight="bold", va="top", ha="left", zorder=10)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Prep
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
    if raw.empty or "source_file" not in raw.columns:
        return raw
    df = raw.copy()
    df["_r"] = df["source_file"].map(lambda n: _pur_region_from(n, fn_prefix, ftag))
    df = df[df["_r"] != ""]
    if df.empty:
        return raw.iloc[0:0]
    rc = df.groupby("_r", observed=True)["station_id"].nunique().sort_values(ascending=False)
    n_drop = int(math.ceil(len(rc) * 0.30))
    keep_n = max(len(rc) - n_drop, 0)
    keep = rc.index[:keep_n].tolist()
    return df[df["_r"].isin(keep)].drop(columns=["_r"])

def _load_raw(tag_dir: str, fn_prefix: str, exp: str) -> pd.DataFrame:
    ftag = "base"
    src = PROC / tag_dir
    pat = (str(src / f"predictions_{fn_prefix}_PUB_PUB_s*_{ftag}.csv") if exp == "PUB"
           else str(src / f"predictions_{fn_prefix}_PUR_*_s*_{ftag}.csv"))
    files = sorted(glob.glob(pat))
    if not files:
        return pd.DataFrame()
    frames = []
    for fp in files:
        d = pd.read_csv(fp)
        d["source_file"] = Path(fp).name
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw["station_id"] = raw["station_id"].astype(str).str.strip()
    return raw

def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    
    qt = [f"Q{t}_true" for t in RETURN_PERIODS]
    qp = [f"Q{t}_pred" for t in RETURN_PERIODS]
    missing_p = [c for c in qp if c not in raw.columns]
    missing_t = [c for c in qt if c not in raw.columns]
    
    if missing_p or missing_t:
        return pd.DataFrame()
    
    first_cols = [c for c in ["lat", "lon", "source_file"] if c in raw.columns] + qt
    g = raw.groupby("station_id")
    return pd.concat([g[first_cols].first(), g[qp].median()], axis=1).reset_index()

def _nse(qt: np.ndarray, qp: np.ndarray) -> float:
    ok = np.isfinite(qt) & np.isfinite(qp) & (qt > 0) & (qp > 0)
    if ok.sum() < 3:
        return np.nan
    t, p = qt[ok], qp[ok]
    ss = float(np.sum((t - t.mean()) ** 2))
    return float(1 - np.sum((t - p) ** 2) / ss) if ss > 0 else np.nan

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
    if not basin_df.empty and "lat" in basin_df.columns:
        bsub = basin_df[["station_id", "lat", "lon"]].rename(columns={"lat": "_blat", "lon": "_blon"})
        out = out.merge(bsub, on="station_id", how="left")
        for col, _col in [("lat", "_blat"), ("lon", "_blon")]:
            out[col] = out[col].where(np.isfinite(out[col]), out[_col])
        out = out.drop(columns=["_blat", "_blon"], errors="ignore")
    return out

def _load_target_pur_labels(basin_assign: pd.DataFrame) -> list[str]:
    labels: list[str] = []

    if RETAINED_BASIN_CSV.exists():
        try:
            keep_df = pd.read_csv(RETAINED_BASIN_CSV, keep_default_na=False)
            if "basin_label" in keep_df.columns:
                labels = keep_df["basin_label"].astype(str).str.strip().tolist()
        except Exception as exc:
            LOG.warning("Failed reading retained PUR basin list: %s", exc)

    if not labels and (not basin_assign.empty) and ("basin_label" in basin_assign.columns):
        cnt = (
            basin_assign.groupby("basin_label", observed=True)["station_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        labels = cnt.head(max(PUR_KEEP_N, 1)).index.astype(str).tolist()

    seen = set()
    labels = [x for x in labels if x and not (x in seen or seen.add(x))]
    if labels:
        LOG.info("PUR target regions loaded: %d", len(labels))
    else:
        LOG.warning("No PUR target region labels loaded; map will use available labels only.")

    return labels

def _calc_violations(agg_df: pd.DataFrame) -> pd.DataFrame:
    out = agg_df.copy()
    out["violated_any"] = False
    for t1, t2 in ADJ_PAIRS:
        c1, c2 = f"Q{t1}_pred", f"Q{t2}_pred"
        viol = out[c1] > out[c2]
        out[f"viol_Q{t1}_Q{t2}"] = viol
        out["violated_any"] = out["violated_any"] | viol
    return out

def _assign_climate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["climate_zone"] = "Unknown"
    if not CLIMATE_SHP.exists():
        return out
    cz = gpd.read_file(CLIMATE_SHP)
    if "Name" not in cz.columns:
        return out
    if cz.crs is None:
        cz = cz.set_crs("EPSG:4326")
    elif str(cz.crs) != "EPSG:4326":
        cz = cz.to_crs("EPSG:4326")
    
    pts = out.dropna(subset=["lat", "lon"]).copy()
    if pts.empty:
        return out
    gpts = gpd.GeoDataFrame(pts, geometry=[Point(xy) for xy in zip(pts["lon"], pts["lat"])], crs="EPSG:4326")
    joined = gpd.sjoin(gpts, cz[["Name", "geometry"]], how="left", predicate="within")
    mapped = joined[["station_id", "Name"]].drop_duplicates("station_id").rename(columns={"Name": "climate_zone"})
    mapped["climate_zone"] = mapped["climate_zone"].fillna("Unknown")
    
    out = out.drop(columns=["climate_zone"], errors="ignore").merge(mapped, on="station_id", how="left")
    out["climate_zone"] = out["climate_zone"].fillna("Unknown")
    return out

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
    shp_map = _discover_shps(HYDRO_ROOT, HYBAS_LEVEL)
    if not shp_map:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    frames = []
    for cont, shp in sorted(shp_map.items()):
        try:
            gdf = gpd.read_file(shp)
            if gdf.crs is None: gdf = gdf.set_crs("EPSG:4326")
            elif str(gdf.crs) != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            frames.append(gdf[["geometry"]].copy())
        except:
            pass
    if not frames:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")

def _load_pur_basin_polys(basin_assign: pd.DataFrame) -> gpd.GeoDataFrame:
    shp_map = _discover_shps(HYDRO_ROOT, HYBAS_LEVEL)
    rows = []
    for cont, sub in basin_assign.groupby("continent", observed=True):
        shp = shp_map.get(str(cont).upper())
        if not shp:
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
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    out = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    out = out.drop_duplicates(subset=["continent", "HYBAS_ID"])
    bl = basin_assign[["continent", "HYBAS_ID", "basin_label"]].drop_duplicates()
    bl["continent"] = bl["continent"].str.upper()
    bl["HYBAS_ID"] = bl["HYBAS_ID"].astype("int64")
    out = out.merge(bl, on=["continent", "HYBAS_ID"], how="left")
    out["basin_label"] = out["basin_label"].fillna(out["continent"] + "_" + out["HYBAS_ID"].astype(str))
    rp = out.geometry.representative_point()
    out["lon_centroid"] = rp.x
    out["lat_centroid"] = rp.y
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Plotting sub-functions
# ─────────────────────────────────────────────────────────────────────────────
def _ann_pair_vr_table(st_df: pd.DataFrame) -> pd.DataFrame:
    ann_df = st_df[st_df["model"] == "ANN"].copy()
    rows = []
    for t1, t2 in ADJ_PAIRS:
        col = f"viol_Q{t1}_Q{t2}"
        pub = ann_df[ann_df["experiment"] == "PUB"][col]
        pur = ann_df[ann_df["experiment"] == "PUR"][col]
        rows.append({
            "pair": f"Q{t1}->Q{t2}",
            "vr_pub": float(pub.mean()) if not pub.empty else np.nan,
            "vr_pur": float(pur.mean()) if not pur.empty else np.nan,
        })
    return pd.DataFrame(rows)


def _ann_climate_vr_table(st_df: pd.DataFrame) -> pd.DataFrame:
    ann_df = st_df[st_df["model"] == "ANN"].copy()
    rows = []
    for zone in CLIMATE_ORDER:
        sub = ann_df[ann_df["climate_zone"] == zone]
        pub = sub[sub["experiment"] == "PUB"]["violated_any"]
        pur = sub[sub["experiment"] == "PUR"]["violated_any"]
        rows.append({
            "climate_zone": zone,
            "vr_pub": float(pub.mean()) if not pub.empty else np.nan,
            "vr_pur": float(pur.mean()) if not pur.empty else np.nan,
            "n_pub": int(pub.notna().sum()),
            "n_pur": int(pur.notna().sum()),
        })
    return pd.DataFrame(rows)


def _nse_heatmap_table(st_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for exp in ["PUB", "PUR"]:
        for t in RETURN_PERIODS:
            row = {"experiment": exp, "return_period": int(t)}
            qt_col = f"Q{t}_true"
            qp_col = f"Q{t}_pred"
            for mdl in models:
                sub = st_df[(st_df["model"] == mdl) & (st_df["experiment"] == exp)]
                if sub.empty or qt_col not in sub.columns or qp_col not in sub.columns:
                    row[mdl] = np.nan
                else:
                    row[mdl] = _nse(sub[qt_col].to_numpy(float), sub[qp_col].to_numpy(float))
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_map_a(
    ax,
    st_df: pd.DataFrame,
    basin_assign: pd.DataFrame,
    nc_coords: pd.DataFrame,
    pur_labels: list[str] | None = None,
) -> pd.DataFrame:
    del nc_coords
    world_gdf = _load_world_basemap()
    if not world_gdf.empty:
        world_gdf.plot(ax=ax, facecolor="#e9e9e9", edgecolor="#151515",
                       linewidth=0.30, alpha=1.0, zorder=1)

    basin_assign_plot = basin_assign.copy()
    if pur_labels and (not basin_assign.empty) and ("basin_label" in basin_assign.columns):
        basin_assign_plot = basin_assign[basin_assign["basin_label"].isin(pur_labels)].copy()

    region_tbl = pd.DataFrame(columns=["basin_label", "n_stations", "violation_rate"])
    ann_pur = st_df[(st_df["model"] == "ANN") & (st_df["experiment"] == "PUR")].copy()
    basin_meta = pd.DataFrame()
    if not basin_assign_plot.empty and {"station_id", "basin_label"}.issubset(set(basin_assign_plot.columns)):
        basin_meta = basin_assign_plot[["station_id", "basin_label"]].copy()
        basin_meta["station_id"] = basin_meta["station_id"].astype(str).str.strip()
        basin_meta["basin_label"] = basin_meta["basin_label"].astype(str).str.strip()
        basin_meta = basin_meta[(basin_meta["station_id"] != "") & (basin_meta["basin_label"] != "")]
        basin_meta = basin_meta.drop_duplicates("station_id")

    if not ann_pur.empty:
        ann_join = ann_pur.copy()
        if not basin_meta.empty:
            ann_join = ann_join.merge(basin_meta, on="station_id", how="left")
        if "basin_label" not in ann_join.columns:
            ann_join["basin_label"] = np.nan

        if "source_file" in ann_join.columns:
            fallback_lbl = ann_join["source_file"].map(lambda n: _pur_region_from(n, "09_ANN", "base"))
            ann_join["basin_label"] = ann_join["basin_label"].fillna(fallback_lbl)

        ann_join["basin_label"] = ann_join["basin_label"].astype(str).str.strip()
        if pur_labels:
            ann_join = ann_join[ann_join["basin_label"].isin(pur_labels)].copy()
        ann_join = ann_join[(ann_join["basin_label"] != "") & (ann_join["basin_label"] != "nan")].copy()
        if not ann_join.empty:
            grouped = (
                ann_join.groupby("basin_label", observed=True)
                .agg(n_stations=("station_id", "nunique"),
                     violation_rate=("violated_any", "mean"))
                .reset_index()
            )
            if pur_labels:
                region_tbl = pd.DataFrame({"basin_label": pur_labels}).merge(grouped, on="basin_label", how="left")
                region_tbl["n_stations"] = region_tbl["n_stations"].fillna(0).astype(int)
            else:
                region_tbl = grouped
    if region_tbl.empty and pur_labels:
        region_tbl = pd.DataFrame({
            "basin_label": pur_labels,
            "n_stations": np.zeros(len(pur_labels), dtype=int),
            "violation_rate": np.nan,
        })

    basin_plot = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    if not basin_assign_plot.empty:
        basin_gdf = _load_pur_basin_polys(basin_assign_plot)
        if not basin_gdf.empty:
            basin_plot = basin_gdf.copy()
            if not region_tbl.empty:
                basin_plot = basin_plot.merge(region_tbl, on="basin_label", how="left")
            else:
                basin_plot["n_stations"] = np.nan
                basin_plot["violation_rate"] = np.nan

    if not basin_plot.empty:
        vals = basin_plot["violation_rate"].to_numpy(float)
        finite = vals[np.isfinite(vals)]
        vmax = MAP_VMAX if finite.size == 0 else min(1.0, max(0.1, float(np.nanpercentile(finite, 92))))
        norm = mcolors.Normalize(vmin=MAP_VMIN, vmax=vmax)

        basin_plot.plot(
            ax=ax,
            column="violation_rate",
            cmap=MAP_CMAP,
            norm=norm,
            edgecolor="#101010",
            linewidth=1.05,
            alpha=0.98,
            zorder=3,
            missing_kwds={"color": "#d0d0d0", "edgecolor": "#000000"},
        )
        cbar_ax = ax.inset_axes([0.47, 0.05, 0.28, 0.035])
        sm = plt.cm.ScalarMappable(cmap=MAP_CMAP, norm=norm)
        sm.set_array([])
        cb = ax.figure.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cb.set_ticks(np.linspace(MAP_VMIN, vmax, 6))
        cb.set_label("Violation Rate (%)", fontsize=24, labelpad=10)
        cb.ax.tick_params(labelsize=20, length=4, width=1.1, color="#444444")
        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
        cb.outline.set_linewidth(1.1)
        cb.outline.set_edgecolor("#444444")
        cbar_ax.set_facecolor((1, 1, 1, 0.84))

    if not basin_plot.empty:
        bb = basin_plot.geometry.bounds
        minx = float(np.nanmin(bb["minx"].to_numpy(float)))
        maxx = float(np.nanmax(bb["maxx"].to_numpy(float)))
        miny = float(np.nanmin(bb["miny"].to_numpy(float)))
        maxy = float(np.nanmax(bb["maxy"].to_numpy(float)))
        
        ax.set_anchor('W')
        x_left_adj = minx + max(0, (-125 - minx) / 2.0)
        x_left = x_left_adj - 1.0
        x_right = maxx + 3.0
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(miny - 3.0, maxy + 4.0)
    else:
        ax.set_anchor('W')
        ax.set_xlim(-145, 160)
        ax.set_ylim(-60, 85)
    ax.grid(False)
    ax.set_axis_off()
    ax.text(-0.06, 1.03, "a", transform=ax.transAxes,
            fontsize=36, fontweight="bold", va="top", ha="left", zorder=10)
            
    ax.text(0.02, 0.20, "ANN", transform=ax.transAxes,
            fontsize=36, fontweight="bold", va="center", ha="left", zorder=10)
            
    return region_tbl


def _plot_bar_b(ax, st_df: pd.DataFrame) -> pd.DataFrame:
    pair_tbl = _ann_pair_vr_table(st_df)
    if pair_tbl.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        _letter(ax, "b", x=-0.06, y=1.16)
        return pair_tbl

    y = np.arange(len(pair_tbl))
    h = 0.36
    pub = pair_tbl["vr_pub"].to_numpy(float)
    pur = pair_tbl["vr_pur"].to_numpy(float)

    b_pub = ax.barh(y - h / 2, pub, height=h, color=C_PUB,
                    alpha=0.90, edgecolor="white", linewidth=0.8, label="PUB", zorder=3)
    b_pur = ax.barh(y + h / 2, pur, height=h, color=C_PUR,
                    alpha=0.90, edgecolor="white", linewidth=0.8, label="PUR", zorder=3)

    finite = np.r_[pub[np.isfinite(pub)], pur[np.isfinite(pur)]]
    xmax = float(np.nanmax(finite)) if finite.size else 1.0
    ax.set_xlim(0, 0.20)
    txt_dx = 0.004

    def _pct_fmt(v: float) -> str:
        p = v * 100
        return f"{p:.1f}%" if p < 10 else f"{p:.0f}%"

    for i in range(len(y)):
        v_pub, v_pur = pub[i], pur[i]
        dy_pub, dy_pur = 0.0, 0.0
        if np.isfinite(v_pub) and np.isfinite(v_pur) and abs(v_pub - v_pur) < (xmax * 0.12):
            dy_pub = -0.07
            dy_pur = 0.07
        if np.isfinite(v_pub):
            ax.text(v_pub + txt_dx, y[i] - h / 2 + dy_pub, _pct_fmt(v_pub), va="center", ha="left", fontsize=16, color="#222222")
        if np.isfinite(v_pur):
            ax.text(v_pur + txt_dx, y[i] + h / 2 + dy_pur, _pct_fmt(v_pur), va="center", ha="left", fontsize=16, color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(pair_tbl["pair"].tolist(), fontsize=16)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.set_xticks([0, 0.05, 0.10, 0.15])
    ax.set_xlabel("Violation Rate (%)", fontsize=24)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", ncol=1, fontsize=20, frameon=False)
    _letter(ax, "b", x=-0.06, y=1.16)
    return pair_tbl


def _plot_bar_c(ax, st_df: pd.DataFrame) -> pd.DataFrame:
    climate_tbl = _ann_climate_vr_table(st_df)
    if climate_tbl.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        _letter(ax, "c", x=-0.06, y=1.16)
        return climate_tbl

    y = np.arange(len(climate_tbl))
    h = 0.36
    pub = climate_tbl["vr_pub"].to_numpy(float)
    pur = climate_tbl["vr_pur"].to_numpy(float)

    for i in range(len(climate_tbl)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f8fafc", zorder=0)

    b_pub = ax.barh(y - h / 2, pub, height=h, color=C_PUB,
                   alpha=0.90, edgecolor="white", linewidth=0.8, label="PUB", zorder=3)
    b_pur = ax.barh(y + h / 2, pur, height=h, color=C_PUR,
                   alpha=0.90, edgecolor="white", linewidth=0.8, label="PUR", zorder=3)

    finite = np.r_[pub[np.isfinite(pub)], pur[np.isfinite(pur)]]
    xmax = float(np.nanmax(finite)) if finite.size else 1.0
    ax.set_xlim(0, 0.60)
    txt_dx = 0.012

    def _pct_fmt(v: float) -> str:
        p = v * 100
        return f"{p:.1f}%" if p < 10 else f"{p:.0f}%"

    for i in range(len(y)):
        v_pub, v_pur = pub[i], pur[i]
        dy_pub, dy_pur = 0.0, 0.0
        if np.isfinite(v_pub) and np.isfinite(v_pur) and abs(v_pub - v_pur) < (xmax * 0.12):
            dy_pub = -0.07
            dy_pur = 0.07
        if np.isfinite(v_pub):
            ax.text(v_pub + txt_dx, y[i] - h / 2 + dy_pub, _pct_fmt(v_pub), va="center", ha="left", fontsize=16, color="#222222")
        if np.isfinite(v_pur):
            ax.text(v_pur + txt_dx, y[i] + h / 2 + dy_pur, _pct_fmt(v_pur), va="center", ha="left", fontsize=16, color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(climate_tbl["climate_zone"].tolist(), fontsize=16)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.set_xlabel("Violation Rate (%)", fontsize=24)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _letter(ax, "c", x=-0.06, y=1.16)
    return climate_tbl


def _plot_lollipop_b(ax, st_df: pd.DataFrame) -> pd.DataFrame:
    pair_tbl = _ann_pair_vr_table(st_df)
    if pair_tbl.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        _letter(ax, "b", x=-0.06, y=1.16)
        return pair_tbl

    y = np.arange(len(pair_tbl))
    h = 0.16  # 全局稍微增加一点棒棒糖的间距
    pub = pair_tbl["vr_pub"].to_numpy(float)
    pur = pair_tbl["vr_pur"].to_numpy(float)

    for i in range(len(y)):
        if np.isfinite(pub[i]):
            ax.plot([0, pub[i]], [y[i] - h, y[i] - h], color="#A0A0A0", lw=18, zorder=2)
            ax.scatter(pub[i], y[i] - h, color="#007BFF", s=220, edgecolors="#111111", linewidths=1.2, zorder=3, label="PUB" if i==0 else "")
        if np.isfinite(pur[i]):
            ax.plot([0, pur[i]], [y[i] + h, y[i] + h], color="#A0A0A0", lw=18, zorder=2)
            ax.scatter(pur[i], y[i] + h, color="#FF2A2A", s=220, edgecolors="#111111", linewidths=1.2, zorder=3, label="PUR" if i==0 else "")

    finite = np.r_[pub[np.isfinite(pub)], pur[np.isfinite(pur)]]
    xmax = float(np.nanmax(finite)) if finite.size else 1.0
    ax.set_xlim(0, 0.20)
    txt_dx = 0.004

    def _pct_fmt(v: float) -> str:
        p = v * 100
        return f"{p:.1f}%" if p < 10 else f"{p:.0f}%"

    for i in range(len(y)):
        v_pub, v_pur = pub[i], pur[i]
        dy_pub, dy_pur = 0.0, 0.0
        if np.isfinite(v_pub) and np.isfinite(v_pur) and abs(v_pub - v_pur) < (xmax * 0.12):
            dy_pub = -0.08
            dy_pur = 0.08
        if np.isfinite(v_pub):
            ax.text(v_pub + txt_dx, y[i] - h + dy_pub, _pct_fmt(v_pub), va="center", ha="left", fontsize=20, color="#222222")
        if np.isfinite(v_pur):
            ax.text(v_pur + txt_dx, y[i] + h + dy_pur, _pct_fmt(v_pur), va="center", ha="left", fontsize=20, color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(pair_tbl["pair"].tolist(), fontsize=16)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.set_xticks([0, 0.05, 0.10, 0.15])
    ax.set_xlabel("Violation Rate (%)", fontsize=24)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", ncol=1, fontsize=20, frameon=False)
    _letter(ax, "b", x=-0.06, y=1.16)
    return pair_tbl


def _plot_lollipop_c(ax, st_df: pd.DataFrame) -> pd.DataFrame:
    climate_tbl = _ann_climate_vr_table(st_df)
    if climate_tbl.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        _letter(ax, "c", x=-0.06, y=1.16)
        return climate_tbl

    y = np.arange(len(climate_tbl))
    h = 0.16
    pub = climate_tbl["vr_pub"].to_numpy(float)
    pur = climate_tbl["vr_pur"].to_numpy(float)

    for i in range(len(climate_tbl)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f8fafc", zorder=0)

    for i in range(len(y)):
        if np.isfinite(pub[i]):
            ax.plot([0, pub[i]], [y[i] - h, y[i] - h], color="#A0A0A0", lw=18, zorder=2)
            ax.scatter(pub[i], y[i] - h, color="#007BFF", s=220, edgecolors="#111111", linewidths=1.2, zorder=3, label="PUB" if i==0 else "")
        if np.isfinite(pur[i]):
            ax.plot([0, pur[i]], [y[i] + h, y[i] + h], color="#A0A0A0", lw=18, zorder=2)
            ax.scatter(pur[i], y[i] + h, color="#FF2A2A", s=220, edgecolors="#111111", linewidths=1.2, zorder=3, label="PUR" if i==0 else "")

    finite = np.r_[pub[np.isfinite(pub)], pur[np.isfinite(pur)]]
    xmax = float(np.nanmax(finite)) if finite.size else 1.0
    ax.set_xlim(0, 0.60)
    txt_dx = 0.012

    def _pct_fmt(v: float) -> str:
        p = v * 100
        return f"{p:.1f}%" if p < 10 else f"{p:.0f}%"

    for i in range(len(y)):
        v_pub, v_pur = pub[i], pur[i]
        dy_pub, dy_pur = 0.0, 0.0
        if np.isfinite(v_pub) and np.isfinite(v_pur) and abs(v_pub - v_pur) < (xmax * 0.12):
            dy_pub = -0.08
            dy_pur = 0.08
        if np.isfinite(v_pub):
            ax.text(v_pub + txt_dx, y[i] - h + dy_pub, _pct_fmt(v_pub), va="center", ha="left", fontsize=20, color="#222222")
        if np.isfinite(v_pur):
            ax.text(v_pur + txt_dx, y[i] + h + dy_pur, _pct_fmt(v_pur), va="center", ha="left", fontsize=20, color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(climate_tbl["climate_zone"].tolist(), fontsize=16)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.set_xlabel("Violation Rate (%)", fontsize=24)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#cbd5e1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _letter(ax, "c", x=-0.06, y=1.16)
    return climate_tbl


def _plot_heatmap_d(ax, st_df: pd.DataFrame, ax_top=None,
                    models_list: list | None = None) -> pd.DataFrame:
    models = [m[1] for m in (models_list if models_list is not None else MODELS_ALL)]
    table = _nse_heatmap_table(st_df, models)
    if table.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        _letter(ax, "d", x=-0.34, y=1.02)
        return table

    mat = table[models].to_numpy(float)
    finite = mat[np.isfinite(mat)]
    if finite.size:
        q_lo, q_hi = np.nanpercentile(finite, [15, 85])
        if q_hi <= q_lo:
            q_lo, q_hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        pad = max((q_hi - q_lo) * 0.12, 0.015)
        vmin = max(-0.2, float(q_lo - pad))
        vmax = min(1.0, float(q_hi + pad))
        if vmax <= vmin:
            vmax = min(1.0, vmin + 0.06)
        vcenter = float(np.nanmedian(finite))
        if not (vmin < vcenter < vmax):
            vcenter = 0.5 * (vmin + vmax)
    else:
        vmin, vcenter, vmax = 0.0, 0.5, 1.0

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    n_rows, n_cols = mat.shape
    x = np.arange(n_cols + 1)
    y = np.arange(n_rows + 1)
    mesh = ax.pcolormesh(
        x,
        y,
        mat,
        cmap=HEATMAP_CMAP,
        norm=norm,
        shading="flat",
        edgecolors="#111111",
        linewidth=2.2,
    )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("auto")
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(models, rotation=28, ha="right", fontsize=18)

    q_labels = [f"Q{t}" for t in RETURN_PERIODS] * 2
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(q_labels, fontsize=15)

    pub_n = len(RETURN_PERIODS)
    ax.plot([0, n_cols], [pub_n, pub_n], color="black", linewidth=2.2, linestyle="--", zorder=7)

    def _draw_group_bracket(y0: float, y1: float, label: str) -> None:
        x_outer = -0.80
        x_inner = -0.40
        ax.plot([x_inner, x_outer], [y0, y0], color="black", linewidth=2.5,
            linestyle="--", clip_on=False, zorder=7)
        ax.plot([x_outer, x_outer], [y0, y1], color="black", linewidth=2.5,
            linestyle="--", clip_on=False, zorder=7)
        ax.plot([x_outer, x_inner], [y1, y1], color="black", linewidth=2.5,
            linestyle="--", clip_on=False, zorder=7)
        ax.text(x_outer - 0.40, 0.5 * (y0 + y1), label, rotation=90,
                ha="center", va="center", fontsize=24, fontweight="bold", clip_on=False)

    _draw_group_bracket(0.0, float(pub_n), "PUB")
    _draw_group_bracket(float(pub_n), float(n_rows), "PUR")

    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = ax.inset_axes([1.025, 0.0, 0.04, 1.0])
    cbar = ax.figure.colorbar(mesh, cax=cax)
    cbar.set_label("NSE", fontsize=24)
    cbar.ax.tick_params(labelsize=20, length=4, width=1.0)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    if ax_top is not None:
        pur_start = pub_n
        q50_idx = pur_start + RETURN_PERIODS.index(50)
        q100_idx = pur_start + RETURN_PERIODS.index(100)
        row_idx = [i for i in [q50_idx, q100_idx] if i < mat.shape[0]]
        if row_idx:
            col_means = np.nanmean(mat[row_idx, :], axis=0)
        else:
            col_means = np.nanmean(mat[pub_n:, :], axis=0)
        x_left = np.arange(n_cols)
        ax_top.bar(
            x_left,
            col_means,
            width=1.0,
            align="edge",
            color="#4A90E2",
            edgecolor="black",
            linewidth=1.2,
        )
        ax_top.set_xlim(0, n_cols)
        ax_top.margins(x=0)
        finite_cm = col_means[np.isfinite(col_means)]
        if finite_cm.size == 0:
            ylo, yhi = 0.0, 1.0
        else:
            vmin_bar = float(np.nanmin(finite_cm))
            vmax_bar = float(np.nanmax(finite_cm))
            span = max(vmax_bar - vmin_bar, 0.02)
            ylo = max(0.0, vmin_bar - span * 0.45)
            yhi = min(1.0, vmax_bar + span * 0.35)
            if yhi - ylo < 0.05:
                mid = 0.5 * (ylo + yhi)
                ylo = max(0.0, mid - 0.03)
                yhi = min(1.0, mid + 0.03)
        ax_top.set_ylim(ylo, yhi + 0.05)
        
        if n_cols > 2:
            annot_y = yhi - 0.015
            ax_top.plot([2.1, n_cols - 0.1], [annot_y, annot_y], color="black", lw=1.5, ls="--", clip_on=False)
            ax_top.plot([2.1, 2.1], [annot_y - 0.02, annot_y], color="black", lw=1.5, ls="--", clip_on=False)
            ax_top.plot([n_cols - 0.1, n_cols - 0.1], [annot_y - 0.02, annot_y], color="black", lw=1.5, ls="--", clip_on=False)
            ax_top.text(0.5*(2.1 + n_cols - 0.1), annot_y + 0.01, "GEV constrained",
                        ha="center", va="bottom", fontsize=20, fontweight="bold", fontstyle="italic", clip_on=False)
        
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_top.tick_params(axis="y", labelsize=20)
        ax_top.set_ylabel("NSE", fontsize=24)
        ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax_top.grid(axis="y", linestyle="--", alpha=0.30, color="#cbd5e1")
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        _letter(ax_top, "d", x=-0.34, y=1.04)
    else:
        _letter(ax, "d", x=-0.34, y=1.02)

    table_out = table.copy()
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=True)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(models, rotation=28, ha="right", fontsize=18)
    table_out.insert(0, "row_label", table_out["experiment"] + "_Q" + table_out["return_period"].astype(str))
    return table_out


# ─────────────────────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    LOG.info("=" * 60)
    LOG.info("%s - start", TAG)
    LOG.info("=" * 60)

    _setup_style()
    nc_coords = _load_nc_coords()
    
    basin_assign = pd.DataFrame()
    if BASIN_CSV.exists():
        basin_assign = pd.read_csv(BASIN_CSV, keep_default_na=False)
        if not basin_assign.empty:
            basin_assign["station_id"] = basin_assign["station_id"].astype(str).str.strip()
            basin_assign["continent"] = basin_assign["continent"].astype(str).str.strip().str.upper()
            basin_assign["continent"] = basin_assign["continent"].replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
            basin_assign["HYBAS_ID"] = pd.to_numeric(basin_assign["HYBAS_ID"], errors="coerce")
            basin_assign = basin_assign.dropna(subset=["station_id", "continent", "HYBAS_ID"]).copy()
            basin_assign["HYBAS_ID"] = basin_assign["HYBAS_ID"].astype("int64")
            basin_assign = basin_assign.drop_duplicates("station_id")

    pur_labels = _load_target_pur_labels(basin_assign)
    if pur_labels:
        LOG.info("Panel-a PUR labels (n=%d): %s", len(pur_labels), ", ".join(pur_labels))

    # Deduplicate model entries across all three lists before loading
    _all_entries: dict[str, tuple] = {}
    for entry in MODELS_ALL + MODELS_XGB + MODELS_SVM:
        _all_entries[entry[1]] = entry  # key by display name

    frames = []
    for tag_dir, mdl, fn_prefix in _all_entries.values():
        for exp in ["PUB", "PUR"]:
            raw = _load_raw(tag_dir, fn_prefix, exp)
            if raw.empty: continue
            agg = _aggregate(raw)
            if agg.empty: continue
            agg["model"] = mdl
            agg["experiment"] = exp
            agg = _calc_violations(agg)
            frames.append(agg)
            
    if not frames:
        LOG.error("No data loaded. Check paths and formats.")
        return
        
    st_df = pd.concat(frames, ignore_index=True)
    st_df = _attach_coords(st_df, nc_coords, basin_assign)
    st_df = _assign_climate(st_df)
    
    st_df.to_csv(OUT_DATA / "constrained_station_violations.csv", index=False)
    LOG.info("Data logic completed, saved: %s", OUT_DATA / "constrained_station_violations.csv")
    
    fig = plt.figure(figsize=(24, 14.2))
    outer = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.15, 1.15, 1.0],
        height_ratios=[1.18, 1.02],
        wspace=0.20,
        hspace=0.20,
    )
    
    ax_a = fig.add_subplot(outer[0, :2])

    left_bottom = outer[1, :2].subgridspec(2, 1, height_ratios=[5, 4], hspace=0.55)
    ax_b = fig.add_subplot(left_bottom[0, 0])
    ax_c = fig.add_subplot(left_bottom[1, 0])

    right = outer[:, 2].subgridspec(2, 1, height_ratios=[0.20, 0.80], hspace=0.06)
    ax_d = fig.add_subplot(right[1, 0])
    ax_d_top = fig.add_subplot(right[0, 0], sharex=ax_d)
    
    map_tbl = _plot_map_a(ax_a, st_df, basin_assign, nc_coords, pur_labels=pur_labels)
    pair_tbl = _plot_bar_b(ax_b, st_df)
    climate_tbl = _plot_bar_c(ax_c, st_df)
    heat_tbl = _plot_heatmap_d(ax_d, st_df, ax_top=ax_d_top)

    map_tbl.to_csv(OUT_DATA / "ann_pur_region_violation_rate.csv", index=False)
    pair_tbl.to_csv(OUT_DATA / "ann_pub_pur_pair_violation_rate.csv", index=False)
    climate_tbl.to_csv(OUT_DATA / "ann_pub_pur_climate_violation_rate.csv", index=False)
    heat_tbl.to_csv(OUT_DATA / "model_pub_pur_nse_heatmap_matrix.csv", index=False)
    
    out_fig = OUT_FIG / "fig_optimized1_quantile_violation_composite.png"
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    LOG.info("Done. Saved composite figure to %s", out_fig)
    
    fig2 = plt.figure(figsize=(24, 14.2))
    outer2 = fig2.add_gridspec(
        2,
        3,
        width_ratios=[1.15, 1.15, 1.0],
        height_ratios=[1.18, 1.02],
        wspace=0.20,
        hspace=0.20, 
    )
    
    ax_a2 = fig2.add_subplot(outer2[0, :2])
    
    left_bottom2 = outer2[1, :2].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10)
    ax_b2 = fig2.add_subplot(left_bottom2[0, 0])
    ax_c2 = fig2.add_subplot(left_bottom2[0, 1])

    right2 = outer2[:, 2].subgridspec(2, 1, height_ratios=[0.20, 0.80], hspace=0.06)
    ax_d2 = fig2.add_subplot(right2[1, 0])
    ax_d_top2 = fig2.add_subplot(right2[0, 0], sharex=ax_d2)
    
    _plot_map_a(ax_a2, st_df, basin_assign, nc_coords, pur_labels=pur_labels)
    _plot_lollipop_b(ax_b2, st_df)
    _plot_lollipop_c(ax_c2, st_df)
    _plot_heatmap_d(ax_d2, st_df, ax_top=ax_d_top2)
    
    out_fig2 = OUT_FIG / "fig_optimized2_quantile_violation_composite_lollipop.png"
    fig2.savefig(out_fig2, dpi=300)
    plt.close(fig2)
    LOG.info("Done. Saved secondary lollipop composite figure to %s", out_fig2)

    # ── Extra figures: panel-d heatmap with XGBoost / SVM replacing RF ──────
    for variant_models, variant_name in [
        (MODELS_XGB, "xgboost"),
        (MODELS_SVM, "svm"),
    ]:
        fig_v = plt.figure(figsize=(24, 14.2))
        outer_v = fig_v.add_gridspec(
            2, 3,
            width_ratios=[1.15, 1.15, 1.0],
            height_ratios=[1.18, 1.02],
            wspace=0.20, hspace=0.20,
        )

        ax_a_v = fig_v.add_subplot(outer_v[0, :2])
        left_bot_v = outer_v[1, :2].subgridspec(2, 1, height_ratios=[5, 4], hspace=0.55)
        ax_b_v = fig_v.add_subplot(left_bot_v[0, 0])
        ax_c_v = fig_v.add_subplot(left_bot_v[1, 0])
        right_v = outer_v[:, 2].subgridspec(2, 1, height_ratios=[0.20, 0.80], hspace=0.06)
        ax_d_v = fig_v.add_subplot(right_v[1, 0])
        ax_d_top_v = fig_v.add_subplot(right_v[0, 0], sharex=ax_d_v)

        _plot_map_a(ax_a_v, st_df, basin_assign, nc_coords, pur_labels=pur_labels)
        _plot_bar_b(ax_b_v, st_df)
        _plot_bar_c(ax_c_v, st_df)
        _plot_heatmap_d(ax_d_v, st_df, ax_top=ax_d_top_v,
                        models_list=variant_models)

        out_fig_v = OUT_FIG / f"fig_optimized3_{variant_name}_heatmap_composite.png"
        fig_v.savefig(out_fig_v, dpi=300)
        plt.close(fig_v)
        LOG.info("Saved %s variant composite to %s", variant_name.upper(), out_fig_v)

if __name__ == "__main__":
    main()