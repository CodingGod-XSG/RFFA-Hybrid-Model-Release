# -*- coding: utf-8 -*-
"""Figure5_HybridModel_Performance.py

Reproduces Figure 5 - Performance of the hybrid model relative to GEV-NN.
Panels: (a,b) regional Q100 NSE under PUR for GEV-NN and the hybrid model;
(c) NSE vs return period for GEV-NN and hybrid model under PUB/PUR;
(d) NSE improvement of hybrid model over GEV-NN (delta NSE) by return
period.

GEV-NN (+flow) vs GEV-NN (base) performance composite figure.
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

TAG      = "Figure5_HybridModel_Performance"
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

# Base vs Flow explicitly mapped
MODEL_META = {
    "GEV-base": ("11_GEV_NN", "GEV_NN_ST", "base"),
    "GEV-Flow": ("11_GEV_NN", "GEV_NN_ST", "+flow"),
}

CLIMATE_ORDER = ["Tropical", "Arid", "Temperate", "Cold"]

C_PUB = "#2166AC"
C_PUR = "#B2182B"
C_GEV_BASE = "#D97706"  # Dark/Burnt Orange
C_GEV_FLOW = "#22C55E"  # Bright Green

MAP_CMAP = "RdYlGn"
MAP_VMIN, MAP_VMAX = 0.0, 1.0

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
# Metrics & IO
# ─────────────────────────────────────────────────────────────────────────────
def _nse(qt: np.ndarray, qp: np.ndarray) -> float:
    ok = np.isfinite(qt) & np.isfinite(qp) & (qt > 0) & (qp > 0)
    if ok.sum() < 3: return np.nan
    t, p = qt[ok], qp[ok]
    ss = float(np.sum((t - t.mean()) ** 2))
    return float(1 - np.sum((t - p) ** 2) / ss) if ss > 0 else np.nan

def _seed_from(name: str) -> str:
    m = re.search(r"_s(\d+)(?:_split\d+)?_", str(name))
    return m.group(1) if m else "unk"

def _pur_region_from(name: str, fn_prefix: str, ftag: str) -> str:
    text = str(name)
    i = text.find("PUR_")
    j = re.search(r"_s\d+", text)
    if i >= 0 and j and j.start() > i + 4:
        return text[i+4:j.start()]
    return ""

def _load_raw(model: str, exp: str) -> pd.DataFrame:
    tag_dir, fn_prefix, ftag = MODEL_META[model]
    src = PROC / tag_dir
    pat = (str(src / f"predictions*{fn_prefix}_PUB_PUB_s*_{ftag}.csv") if exp == "PUB"
           else str(src / f"predictions*{fn_prefix}_PUR_*_s*_{ftag}.csv"))
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
    return raw

def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty: return pd.DataFrame()
    qt = [f"Q{t}_true" for t in RETURN_PERIODS]
    qp = [f"Q{t}_pred" for t in RETURN_PERIODS]
    missing = [c for c in qt + qp if c not in raw.columns]
    if missing: return pd.DataFrame()
    first_cols = [c for c in ["lat", "lon"] + qt if c in raw.columns]
    g = raw.groupby("station_id")
    return pd.concat([g[first_cols].first(), g[qp].median()], axis=1).reset_index()

def _by_seed(raw: pd.DataFrame) -> dict:
    if raw.empty or "seed" not in raw.columns: return {}
    return {str(s): _aggregate(d) for s, d in raw.groupby("seed", sort=True)}

def _load_nc_coords() -> pd.DataFrame:
    if not NC_PATH.exists(): return pd.DataFrame(columns=["station_id", "lat", "lon"])
    lat_kw = ["static_gauge_lat", "gauge_lat", "lat"]
    lon_kw = ["static_gauge_lon", "gauge_lon", "lon"]
    with nc.Dataset(NC_PATH) as ds:
        stns = np.array(ds.variables["station"][:]).astype(str)
        lv = next((k for k in lat_kw if k in ds.variables), None)
        lov = next((k for k in lon_kw if k in ds.variables), None)
        lat = np.ma.filled(ds.variables[lv][:], np.nan).astype(float) if lv else np.full(len(stns), np.nan)
        lon = np.ma.filled(ds.variables[lov][:], np.nan).astype(float) if lov else np.full(len(stns), np.nan)
    df = pd.DataFrame({"station_id": stns, "lat": lat, "lon": lon}).dropna(subset=["lat", "lon"])
    return df.drop_duplicates("station_id")

def _attach_coords(df: pd.DataFrame, nc_df: pd.DataFrame, basin_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["lat", "lon"]:
        if col not in out.columns: out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if not nc_df.empty:
        out = out.merge(nc_df.rename(columns={"lat": "_lat", "lon": "_lon"}), on="station_id", how="left")
        for c, _c in [("lat", "_lat"), ("lon", "_lon")]:
            out[c] = out[c].where(np.isfinite(out[c]), out[_c])
        out = out.drop(columns=["_lat", "_lon"], errors="ignore")
    return out

def _assign_climate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["climate_zone"] = "Unknown"
    if not CLIMATE_SHP.exists(): return out
    cz = gpd.read_file(CLIMATE_SHP)
    if "Name" not in cz.columns: return out
    if cz.crs is None: cz = cz.set_crs("EPSG:4326")
    elif str(cz.crs) != "EPSG:4326": cz = cz.to_crs("EPSG:4326")
    pts = out.dropna(subset=["lat", "lon"]).copy()
    gpts = gpd.GeoDataFrame(pts, geometry=[Point(xy) for xy in zip(pts["lon"], pts["lat"])], crs="EPSG:4326")
    joined = gpd.sjoin(gpts, cz[["Name", "geometry"]], how="left", predicate="within")
    mapped = joined[["station_id", "Name"]].drop_duplicates("station_id").rename(columns={"Name": "climate_zone"})
    out = out.drop(columns=["climate_zone"], errors="ignore").merge(mapped, on="station_id", how="left")
    out["climate_zone"] = out["climate_zone"].fillna("Unknown")
    return out

def _nse_by_climate(model: str, nc_coords: pd.DataFrame) -> dict:
    raw = _load_raw(model, "PUR")
    if raw.empty: return {}
    wide = _aggregate(raw)
    if wide.empty: return {}
    wide = _attach_coords(wide, nc_coords, pd.DataFrame(columns=["station_id"]))
    wide = _assign_climate(wide)
    out = {}
    for zone in CLIMATE_ORDER:
        grp = wide[wide["climate_zone"] == zone]
        if len(grp) < 3: out[zone] = np.nan; continue
        qt = grp[f"Q{Q_TAIL}_true"].to_numpy(float)
        qp = grp[f"Q{Q_TAIL}_pred"].to_numpy(float)
        out[zone] = _nse(qt, qp)
    return out

def _pur_region_nse(model: str, basin_assign: pd.DataFrame, nc_coords: pd.DataFrame):
    _, fn_prefix, ftag = MODEL_META[model]
    raw = _load_raw(model, "PUR")
    if raw.empty: return pd.DataFrame(), pd.DataFrame(columns=["station_id", "lat", "lon"])
    raw = raw.copy()
    raw["pur_region"] = raw["source_file"].map(lambda n: _pur_region_from(n, fn_prefix, ftag))
    raw = raw[raw["pur_region"].astype(str).str.strip() != ""]
    if raw.empty: return pd.DataFrame(), pd.DataFrame()
    wide_all = _aggregate(raw)
    wide_all = _attach_coords(wide_all, nc_coords, basin_assign)
    pts = wide_all[["station_id", "lat", "lon"]].drop_duplicates("station_id")
    rows = []
    for region, g in raw.groupby("pur_region"):
        w = _aggregate(g)
        if w.empty: continue
        qt = w[f"Q{Q_TAIL}_true"].to_numpy(float)
        qp = w[f"Q{Q_TAIL}_pred"].to_numpy(float)
        rows.append({"basin_label": str(region), "n_stations": int(w["station_id"].nunique()), "NSE_Q100": _nse(qt, qp)})
    region_df = pd.DataFrame(rows)
    if not region_df.empty and not basin_assign.empty:
        basin_meta = basin_assign[["basin_label", "continent", "HYBAS_ID"]].drop_duplicates("basin_label")
        region_df = region_df.merge(basin_meta, on="basin_label", how="left")
    return region_df, pts

def _seed_nse_curve(model: str) -> pd.DataFrame:
    rows = []
    for exp in ["PUB", "PUR"]:
        raw = _load_raw(model, exp)
        for seed, w in _by_seed(raw).items():
            for t in RETURN_PERIODS:
                qt = w[f"Q{t}_true"].to_numpy(float)
                qp = w[f"Q{t}_pred"].to_numpy(float)
                rows.append({"model": model, "experiment": exp, "seed": seed, "return_period": t, "NSE": _nse(qt, qp)})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.groupby(["model", "experiment", "return_period"], as_index=False)["NSE"]
            .agg(["mean", "std"]).reset_index()
            .rename(columns={"mean": "NSE_mean", "std": "NSE_std"}))

def _climate_table(model: str, climate_nse: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "model": [model] * len(CLIMATE_ORDER),
        "climate_zone": CLIMATE_ORDER,
        "NSE_Q100": [climate_nse.get(z, np.nan) for z in CLIMATE_ORDER],
    })

def _delta_nse_table(df_base: pd.DataFrame, df_flow: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "experiment",
        "return_period",
        "GEV-base_NSE_mean",
        "Hybrid_Model_NSE_mean",
        "Delta_NSE_Flow_minus_Base",
    ]
    if df_base.empty or df_flow.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp in ["PUB", "PUR"]:
        for t in RETURN_PERIODS:
            vb = df_base[(df_base["experiment"] == exp) & (df_base["return_period"] == t)]
            vf = df_flow[(df_flow["experiment"] == exp) & (df_flow["return_period"] == t)]
            if vb.empty or vf.empty:
                continue
            vb_val = float(vb["NSE_mean"].iloc[0])
            vf_val = float(vf["NSE_mean"].iloc[0])
            rows.append({
                "experiment": exp,
                "return_period": t,
                "GEV-base_NSE_mean": vb_val,
                "Hybrid_Model_NSE_mean": vf_val,
                "Delta_NSE_Flow_minus_Base": vf_val - vb_val,
            })

    return pd.DataFrame(rows, columns=cols)

# ─────────────────────────────────────────────────────────────────────────────
# Geometry loaders
# ─────────────────────────────────────────────────────────────────────────────
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
    for cont, shp in sorted(shp_map.items()):
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
    bl["continent"] = bl["continent"].str.upper()
    bl["HYBAS_ID"] = bl["HYBAS_ID"].astype("int64")
    out = out.merge(bl, on=["continent", "HYBAS_ID"], how="left")
    out["basin_label"] = out["basin_label"].fillna(out["continent"] + "_" + out["HYBAS_ID"].astype(str))
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
def _draw_map(ax, world_gdf, perf_gdf, pts, title: str, letter: str, climate_nse: dict, bar_color: str):
    if not world_gdf.empty:
        world_gdf.plot(ax=ax, facecolor="#e9e9e9", edgecolor="#151515", linewidth=0.30, alpha=1.0, zorder=1)
    if not perf_gdf.empty and "NSE_Q100" in perf_gdf.columns:
        norm = mcolors.Normalize(vmin=MAP_VMIN, vmax=MAP_VMAX)
        perf_gdf.plot(ax=ax, column="NSE_Q100", cmap=MAP_CMAP, norm=norm,
                      edgecolor="#101010", linewidth=1.05, alpha=0.98, zorder=3,
                      missing_kwds={"color": "#d0d0d0", "edgecolor": "#000000"})
        '''
        # Add black border to region boundaries, aligned with Res1
        b = perf_gdf.geometry.bounds
        for minx, miny, maxx, maxy in b.itertuples(index=False, name=None):
            if not np.isfinite((minx, miny, maxx, maxy)).all():
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
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    ax.set_xlim(-145, 180)
    ax.set_ylim(-60, 85)
    ax.grid(False)
    ax.set_axis_off()
    
    # Shift the X-axis coordinate from -119.5 to -135.0
    ax.text(-135.0, -45, title, fontsize=32, fontweight="bold", ha="left")
    # Use data coordinates for x (lon = -125 is approx US west coast) and axes for y to fix it visually 
    ax.text(-125.0, 1.03, letter, transform=ax.get_xaxis_transform(),
            fontsize=36, fontweight="bold", va="top", ha="left", zorder=10)

    if climate_nse and bar_color:
        ins = ax.inset_axes([0.82, 0.60, 0.18, 0.25])
        zones = CLIMATE_ORDER[::-1]
        vals = [climate_nse.get(z, np.nan) for z in zones]
        y = np.arange(len(zones))
        bars = ins.barh(y, vals, height=0.6, color=bar_color, alpha=0.85, edgecolor="white")
        ins.set_yticks(y)
        ins.set_yticklabels(zones, fontsize=16)
        ins.set_xticks([])
        ins.set_xlim(0, 0.65)
        ins.spines['top'].set_visible(False)
        ins.spines['right'].set_visible(False)
        ins.spines['bottom'].set_visible(False)
        ins.set_facecolor((1, 1, 1, 0.85))
        for i, v in enumerate(vals):
            if np.isfinite(v): ins.text(v + 0.02, i, f"{v:.2f}", va='center', ha='left', fontsize=14)

def _draw_lines(ax, df_base, df_flow, letter):
    ax.axvspan(3.5, 5.5, color="#DDE8F5", alpha=0.55, zorder=0)
    ax.text(4.5, 0.98, "Extreme\ntail", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=18, color="#3A5B8C", fontstyle="italic", linespacing=1.3)

    x = np.arange(len(RETURN_PERIODS))
    
    def _smooth(xv, yv, ev):
        xs = np.linspace(float(xv.min()), float(xv.max()), 240)
        if len(xv) >= 3:
            ys = PchipInterpolator(xv, yv)(xs)
            es = PchipInterpolator(xv, ev)(xs)
        else:
            ys = np.interp(xs, xv, yv)
            es = np.interp(xs, xv, ev)
        return xs, ys, np.maximum(es, 0.010)

    # Replaced the label here with GEV-NN and Hybrid Model
    specs = [
        (df_base, "PUB", C_PUB, 3.2, "o", None, "GEV-NN PUB"),
        (df_base, "PUR", C_PUR, 3.2, "o", None, "GEV-NN PUR"),
        (df_flow, "PUB", C_PUB, 2.8, "s", (6, 3), "Hybrid Model PUB"),
        (df_flow, "PUR", C_PUR, 2.8, "s", (6, 3), "Hybrid Model PUR"),
    ]
    all_y = []
    
    for _df, exp, color, lw, marker, dash, label in specs:
        sub = _df[_df["experiment"] == exp]
        if sub.empty: continue
        y = np.array([sub[sub["return_period"] == t]["NSE_mean"].iloc[0] if not sub[sub["return_period"] == t].empty else np.nan for t in RETURN_PERIODS])
        e = np.array([sub[sub["return_period"] == t]["NSE_std"].iloc[0] if not sub[sub["return_period"] == t].empty else 0 for t in RETURN_PERIODS])
        ok = np.isfinite(y)
        if ok.sum() < 2: continue
        xv, yv, ev = x[ok].astype(float), y[ok].astype(float), e[ok].astype(float)
        xs, ys, es = _smooth(xv, yv, ev)

        ln, = ax.plot(xs, ys, color=color, lw=lw, marker=marker, markevery=40, ms=8, mec="white", mew=1.2, label=label, zorder=4)
        if dash: ln.set_dashes(dash)
        ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.16 if dash else 0.12, linewidth=0, zorder=2)
        all_y.extend(ys.tolist())

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{t}" for t in RETURN_PERIODS], fontsize=20)
    ax.set_ylabel("NSE", fontsize=24)
    ax.axhline(0, color="#BBBBBB", lw=0.9, ls=":")
    if all_y:
        ax.set_ylim(max(0.0, float(np.nanmin(all_y)) - 0.05), min(1.0, float(np.nanmax(all_y)) + 0.07))
    ax.legend(ncol=2, loc="lower left", handlelength=1.8, columnspacing=0.8, fontsize=18)
    _letter(ax, letter)

def _draw_delta_bars(ax, df_base, df_flow, letter):
    rows = []
    for exp in ["PUB", "PUR"]:
        for t in RETURN_PERIODS:
            v_b = df_base[(df_base["experiment"] == exp) & (df_base["return_period"] == t)]
            v_f = df_flow[(df_flow["experiment"] == exp) & (df_flow["return_period"] == t)]
            if not v_b.empty and not v_f.empty:
                delta = float(v_f["NSE_mean"].iloc[0]) - float(v_b["NSE_mean"].iloc[0])
                rows.append({"experiment": exp, "return_period": t, "delta_NSE": delta})
    delta_df = pd.DataFrame(rows)
    if delta_df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        _letter(ax, letter)
        return

    x = np.arange(len(RETURN_PERIODS))
    w = 0.38 # Same as Res1
    all_vals = []
    for i, (exp, color) in enumerate([("PUB", C_PUB), ("PUR", C_PUR)]):
        pos = x - w/2 if i == 0 else x + w/2
        sub = delta_df[delta_df["experiment"] == exp]
        vals = [float(sub[sub["return_period"] == t]["delta_NSE"].iloc[0]) if not sub[sub["return_period"] == t].empty else np.nan for t in RETURN_PERIODS]
        bars = ax.bar(pos, vals, width=w, color=color, alpha=0.60, label=exp, edgecolor=color, linewidth=1.5)
        all_vals.extend(vals)
        for b, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.003 if val >= 0 else val - 0.003, f"{val:+.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=16)

    # Slightly increase the y-axis limit to leave space at the top, visually breaking the red bars
    finite_vals = [v for v in all_vals if np.isfinite(v)]
    if finite_vals:
        max_v = max(finite_vals)
        min_v = min(finite_vals)
        ax.set_ylim(min(0.0, min_v - 0.01), max_v * 1.35)

    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="-")
    ax.axvspan(3.5, 5.5, color="#DDE8F5", alpha=0.45, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{t}" for t in RETURN_PERIODS], fontsize=20)
    ax.set_ylabel("ΔNSE (Hybrid Model − GEV-NN)", fontsize=24)
    ax.legend(loc="upper right", fontsize=20, frameon=False)
    _letter(ax, letter)

    # Hide the maximum y-axis label to prevent overlap with the panel letter
    yticks = ax.get_yticks()
    if len(yticks) > 0:
        ax.set_yticks(yticks)
        labs = ax.get_yticklabels()
        if len(labs) > 0:
            labs[-1].set_visible(False)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _setup_style()
    LOG.info("Start %s", TAG)
    
    basin_df = pd.read_csv(BASIN_CSV, keep_default_na=False) if BASIN_CSV.exists() else pd.DataFrame(columns=["station_id","continent","HYBAS_ID","basin_label"])
    if not basin_df.empty:
        basin_df["station_id"] = basin_df["station_id"].astype(str).str.strip()
        basin_df["continent"] = basin_df["continent"].where(basin_df["continent"].notna(), np.nan)
        basin_df["continent"] = basin_df["continent"].astype(str).str.strip().str.upper()
        basin_df["continent"] = basin_df["continent"].replace({"": np.nan, "NAN": np.nan, "NONE": np.nan}) 
        basin_df["HYBAS_ID"] = pd.to_numeric(basin_df["HYBAS_ID"], errors="coerce").astype("int64")
        basin_df = basin_df.dropna(subset=["station_id", "continent", "HYBAS_ID"]).drop_duplicates("station_id")

    nc_df = _load_nc_coords()
    world_gdf = _load_world_basemap()
    try: pur_polys = _load_pur_basin_polys(basin_df)
    except: pur_polys = gpd.GeoDataFrame(columns=["basin_label", "geometry"], geometry="geometry", crs="EPSG:4326")

    # Map data
    r_base, p_base = _pur_region_nse("GEV-base", basin_df, nc_df)
    clim_base = _nse_by_climate("GEV-base", nc_df)
    map_base = pur_polys.merge(r_base[["basin_label", "NSE_Q100"]], on="basin_label", how="right") if not r_base.empty else pur_polys.copy()

    r_flow, p_flow = _pur_region_nse("GEV-Flow", basin_df, nc_df)
    clim_flow = _nse_by_climate("GEV-Flow", nc_df)
    map_flow = pur_polys.merge(r_flow[["basin_label", "NSE_Q100"]], on="basin_label", how="right") if not r_flow.empty else pur_polys.copy()

    # Line data
    line_base = _seed_nse_curve("GEV-base")
    line_flow = _seed_nse_curve("GEV-Flow")

    # Subplot statistics (for CSV export) without touching plotting logic.
    df_a_region = r_base.copy()
    df_b_region = r_flow.copy()
    df_a_climate = _climate_table("GEV-base", clim_base)
    df_b_climate = _climate_table("GEV-Flow", clim_flow)
    if line_base.empty and line_flow.empty:
        df_c_curve = pd.DataFrame(columns=["model", "experiment", "return_period", "NSE_mean", "NSE_std"])
    else:
        df_c_curve = pd.concat([line_base, line_flow], ignore_index=True)
    df_d_delta = _delta_nse_table(line_base, line_flow)

    csv_exports = {
        "subplot_a_region_q100_nse.csv": df_a_region,
        "subplot_a_climate_q100_nse.csv": df_a_climate,
        "subplot_b_region_q100_nse.csv": df_b_region,
        "subplot_b_climate_q100_nse.csv": df_b_climate,
        "subplot_c_return_period_nse_curve.csv": df_c_curve,
        "subplot_d_delta_nse.csv": df_d_delta,
    }
    for name, df_out in csv_exports.items():
        out_csv = OUT_DATA / name
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        LOG.info("Saved CSV: %s", out_csv)

    # Res1 2x2 Layout mapping replacing empty slots
    fig = plt.figure(figsize=(24, 13))
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[1.0, 1.0],
        width_ratios=[4.0, 1.55],
        hspace=0.15, wspace=0.05,
        left=0.01, right=0.97, top=0.97, bottom=0.05,
    )
    
    ax_map_base = fig.add_subplot(gs[0, 0])
    ax_curve    = fig.add_subplot(gs[0, 1])

    ax_map_flow = fig.add_subplot(gs[1, 0])
    ax_delta    = fig.add_subplot(gs[1, 1])

    _draw_map(ax_map_base, world_gdf, map_base, p_base, "GEV-NN", "a", clim_base, C_GEV_BASE)
    _draw_map(ax_map_flow, world_gdf, map_flow, p_flow, "Hybrid\nModel", "b", clim_flow, C_GEV_FLOW)
    _draw_lines(ax_curve, line_base, line_flow, "c")
    _draw_delta_bars(ax_delta, line_base, line_flow, "d")

    # Horizontal colorbars for both maps (a and b) in the same position
    sm = plt.cm.ScalarMappable(cmap=MAP_CMAP,
                               norm=mcolors.Normalize(MAP_VMIN, MAP_VMAX))
    sm.set_array([])
    
    for ax_m in [ax_map_base, ax_map_flow]:
        cbar_ax = ax_m.inset_axes([0.47, 0.05, 0.28, 0.035])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cb.set_ticks(np.linspace(MAP_VMIN, MAP_VMAX, 6))
        cb.set_label("Q100 NSE", fontsize=22, labelpad=8)
        cb.ax.tick_params(labelsize=18, length=5, width=1.2, color="#444444")
        cb.outline.set_linewidth(1.2)
        cb.outline.set_edgecolor("#444444")
        cbar_ax.set_facecolor((1, 1, 1, 0.84))

    # --- Export Statistical Data to Excel ---
    try:
        excel_path = OUT_DATA / f"{TAG}_Data_Summary.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            # 1. Basin NSE Data
            df_basin = r_base.copy()
            if not df_basin.empty:
                df_basin.rename(columns={"NSE_Q100": "GEV-base_NSE_Q100"}, inplace=True)
            if not r_flow.empty and not df_basin.empty:
                df_basin = df_basin.merge(r_flow[["basin_label", "NSE_Q100"]].rename(columns={"NSE_Q100": "Hybrid_Model_NSE_Q100"}), on="basin_label", how="outer")
            if not df_basin.empty:
                df_basin.to_excel(writer, sheet_name="Basin_Q100_NSE", index=False)

            # 2. Climate Zone NSE Data
            df_clim = pd.DataFrame([
                {"Climate_Zone": z, "GEV-base_Q100_NSE": clim_base.get(z, np.nan), "Hybrid_Model_Q100_NSE": clim_flow.get(z, np.nan)}
                for z in CLIMATE_ORDER
            ])
            df_clim.to_excel(writer, sheet_name="Climate_Q100_NSE", index=False)

            # 3. Return Period (Line) Data
            if not df_c_curve.empty:
                df_c_curve.to_excel(writer, sheet_name="Return_Period_NSE", index=False)

            # 4. Delta NSE (Flow - base)
            if not df_d_delta.empty:
                df_d_delta.to_excel(writer, sheet_name="Delta_NSE", index=False)

        LOG.info("Saved data summary to Excel: %s", excel_path)
    except Exception as e:
        LOG.error("Failed to export Excel (openpyxl missing?): %s", e)

    for fmt in ("png", "pdf"):
        fig.savefig(OUT_FIG / f"{TAG}.{fmt}")
        LOG.info("Saved: %s", OUT_FIG / f"{TAG}.{fmt}")
    plt.close(fig)

if __name__ == "__main__":
    main()