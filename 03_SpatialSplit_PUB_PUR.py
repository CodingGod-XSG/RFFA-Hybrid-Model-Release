# -*- coding: utf-8 -*-
"""
03_SpatialSplit_PUB_PUR.py
================================================================================
Merged spatial-split definitions for the two ungauged-prediction protocols used
throughout the pipeline: PUR (Prediction in Ungauged Regions) and PUB
(Prediction in Ungauged Basins). This script has no direct manuscript figure;
it produces the PUB/PUR split definitions consumed by all downstream
model-training and figure scripts.

Both protocols share the same station-eligibility filter (the intersection of
stations with a valid observed GEV fit and a valid simulated GEV fit), so that
filter is loaded exactly once and then branches into two independent stages.

Inputs
------
    data/proceed/Caravan-GRDC/02_Data-Clean/4_Cara-GRDC-35_cleaned.nc
    data/proceed/Caravan-GRDC/02_Data-Clean/gev_cleaned.csv
    data/proceed/Caravan-GRDC/04_Sim_GEV-Fit/sim_gev_station_params.csv
    data/raw/Hydrosheds/hybas_{na,sa,eu,af,au}_lev01-12_v1c/hybas_*_lev02_v1c.shp

Outputs
-------
Stage A - PUR basin assignment (was 05_PUR_Basin_Select.py):
    data/proceed/Caravan-GRDC/05_PUR_Basin_Select/
      station_basin_assignment.csv   (station_id, lat, lon, continent, HYBAS_ID, basin_label)
      pur_retained_basins.csv        (retained Top-N PUR basin fold list)
      pur_retained_basins.shp        (retained PUR basin polygons; column 'basin_label')
      report.txt, log.txt
    figures/Caravan-GRDC/05_PUR_Basin_Select/*.png

Stage B - PUB train/val/test split (was 06_PUB_Station_Select.py):
    data/proceed/Caravan-GRDC/06_PUB_Station_Select/
      06_PUB_Station_Select.csv
      report.txt, log.txt
    figures/Caravan-GRDC/06_PUB_Station_Select/*.png, *.pdf
"""
from __future__ import annotations

import importlib
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import DATA_RAW, DATA_PROCEED, FIGURE_ROOT, stage_dir
from src.splits import pub_split

warnings.filterwarnings("ignore")


# ============================================================
# 0. PATHS & CONSTANTS
# ============================================================
NC_PATH = DATA_PROCEED / "02_Data-Clean" / "4_Cara-GRDC-35_cleaned.nc"
OBS_GEV_CLEAN_CSV = DATA_PROCEED / "02_Data-Clean" / "gev_cleaned.csv"
SIM_GEV_CSV = DATA_PROCEED / "04_Sim_GEV-Fit" / "sim_gev_station_params.csv"
HYDRO_ROOT = DATA_RAW / "Hydrosheds"

HYBAS_LEVEL = 2
STATION_COORD = "station"
LAT_VAR = "static_gauge_lat"
LON_VAR = "static_gauge_lon"
MIN_PUR_STATIONS = 50
PUR_KEEP_N = 15

# Stage-tagged output directories (created immediately, tags preserved exactly).
OUT_DATA_05 = stage_dir(DATA_PROCEED, "05_PUR_Basin_Select")
OUT_FIG_05 = stage_dir(FIGURE_ROOT, "05_PUR_Basin_Select")
OUT_DATA_06 = stage_dir(DATA_PROCEED, "06_PUB_Station_Select")
OUT_FIG_06 = stage_dir(FIGURE_ROOT, "06_PUB_Station_Select")
TAG_06 = "06_PUB_Station_Select"


def _hydrosheds_shp_map(hydro_root: Path, level: int) -> dict:
    lvl = f"{level:02d}"
    return {
        "NA": hydro_root / "hybas_na_lev01-12_v1c" / f"hybas_na_lev{lvl}_v1c.shp",
        "SA": hydro_root / "hybas_sa_lev01-12_v1c" / f"hybas_sa_lev{lvl}_v1c.shp",
        "EU": hydro_root / "hybas_eu_lev01-12_v1c" / f"hybas_eu_lev{lvl}_v1c.shp",
        "AF": hydro_root / "hybas_af_lev01-12_v1c" / f"hybas_af_lev{lvl}_v1c.shp",
        "AU": hydro_root / "hybas_au_lev01-12_v1c" / f"hybas_au_lev{lvl}_v1c.shp",
    }


def _discover_all_hydrosheds_shps(hydro_root: Path, level: int) -> dict[str, Path]:
    """Discover all available HydroSHEDS level shapefiles under hydro_root."""
    lvl = f"{level:02d}"
    out = {}
    for folder in sorted(hydro_root.glob("hybas_*_lev01-12_v1c")):
        if not folder.is_dir():
            continue
        parts = folder.name.split("_")
        if len(parts) < 2:
            continue
        code = parts[1].upper()
        shp = folder / f"hybas_{code.lower()}_lev{lvl}_v1c.shp"
        if shp.exists():
            out[code] = shp
    return out


CONTINENT_SHPS = _hydrosheds_shp_map(HYDRO_ROOT, HYBAS_LEVEL)
ALL_CONTINENT_SHPS = _discover_all_hydrosheds_shps(HYDRO_ROOT, HYBAS_LEVEL)


# ============================================================
# 1. LOGGING (shared setup, written to both stage folders)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DATA_05 / "log.txt", mode="w", encoding="utf-8"),
        logging.FileHandler(OUT_DATA_06 / "log.txt", mode="w", encoding="utf-8"),
    ],
)
LOG = logging.getLogger(__name__)


# ============================================================
# 2. LOAD STATIONS & SHARED GEV INTERSECTION FILTER
#    (identical logic previously copy-pasted in 05 and 06; loaded once here)
# ============================================================
def load_station_points() -> pd.DataFrame:
    if not NC_PATH.exists():
        raise FileNotFoundError(f"NC file not found: {NC_PATH}")

    with xr.open_dataset(NC_PATH) as ds:
        if STATION_COORD not in ds.coords:
            raise KeyError(
                f"Coordinate '{STATION_COORD}' not found in {NC_PATH}. "
                f"Available coords: {list(ds.coords)}"
            )
        if LAT_VAR not in ds.variables or LON_VAR not in ds.variables:
            raise KeyError(
                f"Lat/Lon variables not found: lat='{LAT_VAR}', lon='{LON_VAR}'. "
                f"Available vars: {list(ds.variables)}"
            )

        station_ids = ds.coords[STATION_COORD].values.astype(str)
        lat = np.asarray(ds[LAT_VAR].values, dtype=float)
        lon = np.asarray(ds[LON_VAR].values, dtype=float)

    df = pd.DataFrame({"station_id": station_ids, "lat": lat, "lon": lon})
    df["station_id"] = df["station_id"].astype(str).str.strip()
    df = df[(df["station_id"] != "") & df["lat"].notna() & df["lon"].notna()].copy()
    df = df.drop_duplicates(subset="station_id", keep="first")

    LOG.info(f"Loaded {len(df)} station points from NC.")
    return df


def _read_station_id_set(csv_path: Path, label: str) -> set[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"{label} file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "station_id" not in df.columns:
        raise KeyError(f"{label} missing required column 'station_id': {csv_path}")
    ids = (
        df["station_id"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
    )
    return set(ids.tolist())


def filter_points_by_gev_intersection(points_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    obs_ids = _read_station_id_set(OBS_GEV_CLEAN_CSV, "obs cleaned GEV")
    sim_ids = _read_station_id_set(SIM_GEV_CSV, "sim GEV")
    inter_ids = obs_ids & sim_ids

    filtered = points_df[points_df["station_id"].isin(inter_ids)].copy()
    stats = {
        "n_nc": int(len(points_df)),
        "n_obs_clean": int(len(obs_ids)),
        "n_sim": int(len(sim_ids)),
        "n_intersection": int(len(inter_ids)),
        "n_after_filter": int(len(filtered)),
        "obs_clean_csv": str(OBS_GEV_CLEAN_CSV),
        "sim_gev_csv": str(SIM_GEV_CSV),
    }

    LOG.info("Station intersection for PUR/PUB eligibility:")
    LOG.info(f"  NC stations               : {stats['n_nc']}")
    LOG.info(f"  Obs cleaned GEV stations  : {stats['n_obs_clean']}")
    LOG.info(f"  Sim GEV stations          : {stats['n_sim']}")
    LOG.info(f"  Obs∩Sim stations          : {stats['n_intersection']}")
    LOG.info(f"  Used for downstream splits: {stats['n_after_filter']}")

    if filtered.empty:
        raise RuntimeError(
            "No stations remain after intersecting obs cleaned GEV and sim GEV station sets."
        )

    return filtered, stats


# ============================================================
# 3A. STAGE A - PUR BASIN ASSIGNMENT
# ============================================================
def assign_basins(points_df: pd.DataFrame) -> pd.DataFrame:
    try:
        gpd = importlib.import_module("geopandas")
        from shapely.geometry import Point
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "geopandas and shapely are required for basin assignment. "
            "Install with: pip install geopandas shapely"
        ) from exc

    pts = gpd.GeoDataFrame(
        points_df,
        geometry=[Point(lo, la) for lo, la in zip(points_df["lon"], points_df["lat"])],
        crs="EPSG:4326",
    )

    rows = []
    for cont, shp_path in CONTINENT_SHPS.items():
        if not shp_path.exists():
            LOG.warning(f"Shapefile not found, skip {cont}: {shp_path}")
            continue

        basins = gpd.read_file(shp_path)
        if "HYBAS_ID" not in basins.columns:
            LOG.warning(f"HYBAS_ID not found in {shp_path}, skip {cont}")
            continue
        basins = basins[["HYBAS_ID", "geometry"]].copy()
        if basins.crs is None:
            basins = basins.set_crs("EPSG:4326")
        elif str(basins.crs) != "EPSG:4326":
            basins = basins.to_crs("EPSG:4326")

        joined = gpd.sjoin(pts, basins, how="inner", predicate="within")
        if len(joined) == 0:
            LOG.warning(f"{cont}: no stations matched")
            continue

        joined = joined.copy()
        joined["continent"] = cont
        joined["HYBAS_ID"] = joined["HYBAS_ID"].astype("int64")
        joined["basin_label"] = cont + "_" + joined["HYBAS_ID"].astype(str)
        rows.append(joined[["station_id", "lat", "lon", "continent", "HYBAS_ID", "basin_label"]])
        LOG.info(f"{cont}: matched {len(joined)} stations")

    if not rows:
        raise RuntimeError("No station was assigned to HydroSHEDS basins.")

    assign_df = pd.concat(rows, ignore_index=True)
    assign_df = assign_df.drop_duplicates(subset="station_id", keep="first")

    n_total = len(points_df)
    n_assigned = len(assign_df)
    LOG.info(f"Assigned {n_assigned}/{n_total} stations. Unassigned: {n_total - n_assigned}")

    return assign_df


def plot_pur_selection_maps(points_all: pd.DataFrame, points_filtered: pd.DataFrame, assign_df: pd.DataFrame) -> None:
    """Plot publication-style HydroSHEDS maps and export the retained PUR basins
    as both CSV (fold list) and shapefile (polygon geometry, 'basin_label' column).
    """
    try:
        gpd = importlib.import_module("geopandas")
    except ModuleNotFoundError:
        LOG.warning(
            "geopandas not found; skip PUR selection map plotting and "
            "pur_retained_basins.shp export"
        )
        return

    # Basin counts from assigned stations.
    basin_counts = (
        assign_df.groupby(["continent", "HYBAS_ID", "basin_label"], observed=True)["station_id"]
        .size()
        .reset_index(name="n_stations")
        .sort_values("n_stations", ascending=False)
    )
    if basin_counts.empty:
        LOG.warning("No basin assignment rows for PUR map plotting")
        return

    candidate = basin_counts[basin_counts["n_stations"] >= MIN_PUR_STATIONS].copy()
    if candidate.empty:
        # Fallback to all assigned basins if threshold is too strict.
        candidate = basin_counts.copy()

    retained = candidate.head(max(PUR_KEEP_N, 1)).copy()

    # Save retained PUR basin list for reproducibility.
    out_retained = OUT_DATA_05 / "pur_retained_basins.csv"
    retained.to_csv(out_retained, index=False)

    # Build polygon layers for retained basin IDs by continent.
    keep_ids_by_cont = {
        cont: set(retained.loc[retained["continent"] == cont, "HYBAS_ID"].astype(int).tolist())
        for cont in retained["continent"].unique()
    }

    world_frames = []
    retained_frames = []

    for cont, shp_path in CONTINENT_SHPS.items():
        if not shp_path.exists():
            continue
        try:
            basins = gpd.read_file(shp_path)
        except Exception as exc:
            LOG.warning(f"Failed reading basin shapefile {shp_path}: {exc}")
            continue
        if basins.empty:
            continue
        if basins.crs is None:
            basins = basins.set_crs("EPSG:4326")
        elif str(basins.crs) != "EPSG:4326":
            basins = basins.to_crs("EPSG:4326")

        if "HYBAS_ID" not in basins.columns:
            continue
        basins = basins[["HYBAS_ID", "geometry"]].copy()
        basins["HYBAS_ID"] = pd.to_numeric(basins["HYBAS_ID"], errors="coerce").astype("Int64")
        basins = basins[basins["HYBAS_ID"].notna()].copy()
        basins["HYBAS_ID"] = basins["HYBAS_ID"].astype(int)

        world_sub = basins[["HYBAS_ID", "geometry"]].copy()
        world_sub["continent"] = cont
        world_frames.append(world_sub)

        if cont in keep_ids_by_cont:
            sub_keep = basins[basins["HYBAS_ID"].isin(keep_ids_by_cont[cont])].copy()
            if not sub_keep.empty:
                sub_keep["continent"] = cont
                sub_keep["basin_label"] = cont + "_" + sub_keep["HYBAS_ID"].astype(str)
                retained_frames.append(sub_keep)

    world_gdf = gpd.GeoDataFrame(
        pd.concat(world_frames, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    ) if world_frames else None
    retained_gdf = gpd.GeoDataFrame(
        pd.concat(retained_frames, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    ) if retained_frames else None

    # Export retained PUR basins as a shapefile (basin_label column + polygon geometry).
    # Consumed downstream by the FigureS4 script. Guarded so a write failure never
    # crashes the whole pipeline run.
    if retained_gdf is not None and len(retained_gdf) > 0:
        retained_gdf_export = retained_gdf.merge(
            retained[["basin_label", "n_stations"]], on="basin_label", how="left"
        )
        out_shp = OUT_DATA_05 / "pur_retained_basins.shp"
        try:
            retained_gdf_export.to_file(out_shp, driver="ESRI Shapefile", encoding="utf-8")
            LOG.info(f"Saved retained PUR basins shapefile: {out_shp}")
        except Exception as exc:
            LOG.warning(f"Failed to write retained PUR basins shapefile {out_shp}: {exc}")
    else:
        LOG.warning("No retained PUR basin geometries available; pur_retained_basins.shp was not written.")

    # Build optimized full-world HydroSHEDS basemap from all discoverable continent packages.
    full_world_frames = []
    shp_map_for_full = ALL_CONTINENT_SHPS if ALL_CONTINENT_SHPS else CONTINENT_SHPS
    for _, shp_path in shp_map_for_full.items():
        if not shp_path.exists():
            continue
        try:
            basins = gpd.read_file(shp_path)
        except Exception as exc:
            LOG.warning(f"Failed reading full-world basin shapefile {shp_path}: {exc}")
            continue
        if basins.empty:
            continue
        if basins.crs is None:
            basins = basins.set_crs("EPSG:4326")
        elif str(basins.crs) != "EPSG:4326":
            basins = basins.to_crs("EPSG:4326")
        if "geometry" in basins.columns:
            full_world_frames.append(basins[["geometry"]].copy())

    full_world_gdf = gpd.GeoDataFrame(
        pd.concat(full_world_frames, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    ) if full_world_frames else world_gdf

    def _style_map_axis(ax):
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(False)

    # Single publication figure: retained Top-N PUR regions with unique colors.
    fig2, ax2 = plt.subplots(1, 1, figsize=(14.5, 6.4), constrained_layout=True)
    if world_gdf is not None and len(world_gdf) > 0:
        world_gdf.plot(
            ax=ax2,
            facecolor="#eeeeee",
            edgecolor="#111111",
            linewidth=0.20,
            alpha=1.0,
            zorder=1,
        )

    if retained_gdf is not None and len(retained_gdf) > 0:
        retained_order = retained["basin_label"].astype(str).tolist()
        palette = [
            "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
            "#E6AB02", "#A6761D", "#1F78B4", "#B2DF8A", "#FB9A99",
            "#F781BF", "#A6CEE3", "#B15928", "#6A3D9A", "#33A02C",
        ]
        color_map = {label: palette[i % len(palette)] for i, label in enumerate(retained_order)}

        for label in retained_order:
            sub = retained_gdf[retained_gdf["basin_label"] == label]
            if sub.empty:
                continue
            sub.plot(
                ax=ax2,
                facecolor=color_map[label],
                edgecolor="#000000",
                linewidth=0.62,
                alpha=0.96,
                zorder=3,
            )

    # Keep station support as a light overlay, restricted to retained PUR basins.
    retained_labels = set(retained["basin_label"].astype(str).tolist())
    retained_points = assign_df[assign_df["basin_label"].isin(retained_labels)].copy()
    if not retained_points.empty:
        ax2.scatter(
            retained_points["lon"],
            retained_points["lat"],
            s=5,
            c="#4D0019",
            alpha=0.24,
            linewidths=0,
            zorder=4,
        )

    _style_map_axis(ax2)
    ax2.text(
        0.012, 0.02,
        f"actual_keep={len(retained)}  |  retained_stations={len(retained_points)}  |  min_pur_stations={MIN_PUR_STATIONS}",
        transform=ax2.transAxes,
        ha="left", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#C3C3C3", alpha=0.92),
    )
    out_retained_png = OUT_FIG_05 / "pur_retained_regions_top15_map.png"
    fig2.savefig(out_retained_png, dpi=320)
    plt.close(fig2)
    LOG.info(f"Saved retained PUR map: {out_retained_png}")

    # Optimized version: full HydroSHEDS global basemap as gray filled polygons.
    fig3, ax3 = plt.subplots(1, 1, figsize=(14.5, 6.4), constrained_layout=True)
    if full_world_gdf is not None and len(full_world_gdf) > 0:
        full_world_gdf.plot(
            ax=ax3,
            facecolor="#e9e9e9",
            edgecolor="#111111",
            linewidth=0.16,
            alpha=1.0,
            zorder=1,
        )

    if retained_gdf is not None and len(retained_gdf) > 0:
        retained_order = retained["basin_label"].astype(str).tolist()
        palette = [
            "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
            "#E6AB02", "#A6761D", "#1F78B4", "#B2DF8A", "#FB9A99",
            "#F781BF", "#A6CEE3", "#B15928", "#6A3D9A", "#33A02C",
        ]
        color_map = {label: palette[i % len(palette)] for i, label in enumerate(retained_order)}

        for label in retained_order:
            sub = retained_gdf[retained_gdf["basin_label"] == label]
            if sub.empty:
                continue
            sub.plot(
                ax=ax3,
                facecolor=color_map[label],
                edgecolor="#000000",
                linewidth=0.62,
                alpha=0.96,
                zorder=3,
            )

    retained_labels = set(retained["basin_label"].astype(str).tolist())
    retained_points = assign_df[assign_df["basin_label"].isin(retained_labels)].copy()
    if not retained_points.empty:
        ax3.scatter(
            retained_points["lon"],
            retained_points["lat"],
            s=5,
            c="#4D0019",
            alpha=0.24,
            linewidths=0,
            zorder=4,
        )

    _style_map_axis(ax3)
    ax3.text(
        0.012, 0.02,
        f"actual_keep={len(retained)}  |  retained_stations={len(retained_points)}  |  min_pur_stations={MIN_PUR_STATIONS}",
        transform=ax3.transAxes,
        ha="left", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#C3C3C3", alpha=0.92),
    )
    out_retained_opt_png = OUT_FIG_05 / "pur_retained_regions_top15_map_full_hydrosheds.png"
    fig3.savefig(out_retained_opt_png, dpi=320)
    plt.close(fig3)
    LOG.info(f"Saved retained PUR map (full HydroSHEDS basemap): {out_retained_opt_png}")
    LOG.info(f"Saved retained PUR basin list: {out_retained}")

    # Station-level scatter map: each station colored by its PUR basin.
    palette = [
        "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
        "#E6AB02", "#A6761D", "#1F78B4", "#B2DF8A", "#FB9A99",
        "#F781BF", "#A6CEE3", "#B15928", "#6A3D9A", "#33A02C",
    ]
    retained_order = retained["basin_label"].astype(str).tolist()
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(retained_order)}

    fig4, ax4 = plt.subplots(1, 1, figsize=(14.5, 6.4), constrained_layout=True)
    if full_world_gdf is not None and len(full_world_gdf) > 0:
        full_world_gdf.plot(
            ax=ax4,
            facecolor="#f0f0f0",
            edgecolor="#bbbbbb",
            linewidth=0.14,
            alpha=1.0,
            zorder=1,
        )

    # Non-retained assigned stations in gray.
    other_points = assign_df[~assign_df["basin_label"].isin(retained_labels)].copy()
    if not other_points.empty:
        ax4.scatter(
            other_points["lon"], other_points["lat"],
            s=6, c="#cccccc", alpha=0.45, linewidths=0, zorder=2,
        )

    # Retained stations colored by PUR basin, with legend.
    legend_handles = []
    for label in retained_order:
        sub = assign_df[assign_df["basin_label"] == label]
        if sub.empty:
            continue
        n = int(retained.loc[retained["basin_label"] == label, "n_stations"].values[0])
        sc = ax4.scatter(
            sub["lon"], sub["lat"],
            s=14, c=color_map[label], alpha=0.85, linewidths=0.2,
            edgecolors="#333333", zorder=3, label=f"{label} (n={n})",
        )
        legend_handles.append(sc)

    ax4.legend(
        handles=legend_handles,
        title="PUR Basin",
        fontsize=7,
        title_fontsize=8,
        loc="lower left",
        framealpha=0.88,
        markerscale=1.4,
        ncol=2,
    )
    _style_map_axis(ax4)
    ax4.text(
        0.012, 0.97,
        f"Retained PUR stations={len(retained_points)}  |  Other assigned={len(other_points)}",
        transform=ax4.transAxes,
        ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#C3C3C3", alpha=0.92),
    )
    out_scatter_png = OUT_FIG_05 / "pur_retained_stations_scatter.png"
    fig4.savefig(out_scatter_png, dpi=320)
    plt.close(fig4)
    LOG.info(f"Saved retained PUR station scatter map: {out_scatter_png}")

    # Clean map: only the retained PUR basin polygons, no station overlay.
    if retained_gdf is not None and len(retained_gdf) > 0:
        fig5, ax5 = plt.subplots(1, 1, figsize=(14.5, 6.4), constrained_layout=True)
        if full_world_gdf is not None and len(full_world_gdf) > 0:
            full_world_gdf.plot(
                ax=ax5, facecolor="#f2f2f2", edgecolor="#bbbbbb",
                linewidth=0.14, alpha=1.0, zorder=1,
            )
        legend_handles5 = []
        for i, label in enumerate(retained_order):
            sub = retained_gdf[retained_gdf["basin_label"] == label]
            if sub.empty:
                continue
            n = int(retained.loc[retained["basin_label"] == label, "n_stations"].values[0])
            sub.plot(
                ax=ax5, facecolor=palette[i % len(palette)],
                edgecolor="#000000", linewidth=0.7, alpha=0.92, zorder=2,
            )
            legend_handles5.append(
                plt.matplotlib.patches.Patch(
                    facecolor=palette[i % len(palette)],
                    edgecolor="#000000", linewidth=0.6,
                    label=f"{label} (n={n})",
                )
            )
        ax5.legend(
            handles=legend_handles5, title="PUR Basin",
            fontsize=7.5, title_fontsize=8.5,
            loc="lower left", framealpha=0.9, ncol=2,
        )
        _style_map_axis(ax5)
        out_pur_only_png = OUT_FIG_05 / "pur_15regions_only.png"
        fig5.savefig(out_pur_only_png, dpi=320)
        plt.close(fig5)
        LOG.info(f"Saved retained PUR regions clean map: {out_pur_only_png}")


def save_outputs_pur(assign_df: pd.DataFrame, n_input: int, inter_stats: dict) -> None:
    out_csv = OUT_DATA_05 / "station_basin_assignment.csv"
    assign_df.to_csv(out_csv, index=False)

    fold_sizes = assign_df.groupby("basin_label")["station_id"].size().sort_values(ascending=False)
    cont_sizes = assign_df.groupby("continent")["station_id"].size().sort_values(ascending=False)

    lines = [
        "HydroSHEDS Basin Assignment Report",
        "=" * 72,
        f"Input station points (NC) : {inter_stats['n_nc']}",
        f"Obs cleaned GEV stations  : {inter_stats['n_obs_clean']}",
        f"Sim GEV stations          : {inter_stats['n_sim']}",
        f"Obs∩Sim intersection      : {inter_stats['n_intersection']}",
        f"Input station points used : {n_input}",
        f"Assigned stations         : {len(assign_df)}",
        f"Unassigned stations       : {n_input - len(assign_df)}",
        f"Unique basin labels       : {assign_df['basin_label'].nunique()}",
        f"HydroSHEDS level          : {HYBAS_LEVEL}",
        f"Station source NC         : {NC_PATH}",
        f"Obs cleaned GEV source    : {inter_stats['obs_clean_csv']}",
        f"Sim GEV source            : {inter_stats['sim_gev_csv']}",
        f"HydroSHEDS root           : {HYDRO_ROOT}",
        f"Output CSV                : {out_csv}",
        "",
        "Stations per continent:",
    ]
    for cont, cnt in cont_sizes.items():
        lines.append(f"  - {cont}: {int(cnt)}")

    lines.append("")
    lines.append("Top 20 basins by station count:")
    for label, cnt in fold_sizes.head(20).items():
        lines.append(f"  - {label}: {int(cnt)}")

    report_path = OUT_DATA_05 / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info(f"Saved basin assignment: {out_csv}")
    LOG.info(f"Saved report: {report_path}")


# ============================================================
# 3B. STAGE B - PUB TRAIN/VAL/TEST SPLIT
# ============================================================
def plot_pub_selection_map(points_all: pd.DataFrame, points_filtered: pd.DataFrame) -> None:
    try:
        gpd = importlib.import_module("geopandas")
        from shapely.geometry import Point
    except ModuleNotFoundError:
        LOG.warning("geopandas not found; skip PUB selection map plotting")
        return

    full_world_frames = []
    # Force identical geometry constraints with Stage A (only the specified 5 continents, no Asia).
    shp_map_for_full = CONTINENT_SHPS
    for _, shp_path in shp_map_for_full.items():
        if not shp_path.exists():
            continue
        try:
            basins = gpd.read_file(shp_path)
        except Exception as exc:
            LOG.warning(f"Failed reading full-world basin shapefile {shp_path}: {exc}")
            continue
        if basins.empty:
            continue
        if basins.crs is None:
            basins = basins.set_crs("EPSG:4326")
        elif str(basins.crs) != "EPSG:4326":
            basins = basins.to_crs("EPSG:4326")

        if "HYBAS_ID" in basins.columns:
            basins["HYBAS_ID"] = pd.to_numeric(basins["HYBAS_ID"], errors="coerce").astype("Int64")
            basins = basins[basins["HYBAS_ID"].notna()].copy()

        if "geometry" in basins.columns:
            full_world_frames.append(basins[["geometry"]].copy())

    full_world_gdf = gpd.GeoDataFrame(
        pd.concat(full_world_frames, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    ) if full_world_frames else None

    def filter_by_basin(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or full_world_gdf is None or full_world_gdf.empty:
            return df
        pts = gpd.GeoDataFrame(
            df,
            geometry=[Point(lo, la) for lo, la in zip(df["lon"], df["lat"])],
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(pts, full_world_gdf, how="inner", predicate="within")
        return joined.drop_duplicates(subset="station_id", keep="first").drop(columns=["geometry", "index_right"], errors="ignore")

    fig, ax = plt.subplots(1, 1, figsize=(14.5, 6.4), constrained_layout=True)
    if full_world_gdf is not None and not full_world_gdf.empty:
        full_world_gdf.plot(ax=ax, facecolor="#e9e9e9", edgecolor="#111111", linewidth=0.16, alpha=1.0, zorder=1)

    filtered_out_ids = set(points_all["station_id"]) - set(points_filtered["station_id"])
    df_out = points_all[points_all["station_id"].isin(filtered_out_ids)]

    # Filter points outside basins.
    df_out = filter_by_basin(df_out)

    if not df_out.empty:
        ax.scatter(df_out["lon"], df_out["lat"], s=5, c="#A0A0A0", alpha=0.6, linewidths=0, zorder=2, label=f"Filtered Out ({len(df_out)})")

    if not points_filtered.empty:
        # Generate the PUB data splits using random seed 42 to denote train/val vs test.
        tr_idx, val_idx, te_idx = pub_split(len(points_filtered), seed=42)

        # Reset index to safely use iloc.
        points_filtered_reset = points_filtered.reset_index(drop=True)
        train_val_idx = np.concatenate([tr_idx, val_idx])

        df_train_val = filter_by_basin(points_filtered_reset.iloc[train_val_idx].copy())
        df_test = filter_by_basin(points_filtered_reset.iloc[te_idx].copy())

        if not df_train_val.empty:
            ax.scatter(df_train_val["lon"], df_train_val["lat"], s=8, c="#2166AC", alpha=0.85, linewidths=0, zorder=3, label=f"PUB Train/Val ({len(df_train_val)})")
        if not df_test.empty:
            ax.scatter(df_test["lon"], df_test["lat"], s=70, marker="*", c="#d7191c", alpha=0.95, linewidths=0.4, edgecolors="white", zorder=4, label=f"PUB Test ({len(df_test)})")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.legend(loc="lower left", fontsize=12, frameon=True, facecolor="white", edgecolor="#C3C3C3")

    ax.text(0.012, 0.02, f"Total Candidate Stations: {len(points_all)}", transform=ax.transAxes, ha="left", va="bottom", fontsize=11, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#C3C3C3", alpha=0.92))

    for fmt in ["png", "pdf"]:
        out_file = OUT_FIG_06 / f"{TAG_06}.{fmt}"
        fig.savefig(out_file, dpi=320)
        LOG.info(f"Saved PUB map: {out_file}")
    plt.close(fig)


def save_outputs_pub(points_filtered: pd.DataFrame, inter_stats: dict) -> None:
    out_csv = OUT_DATA_06 / f"{TAG_06}.csv"
    points_filtered.to_csv(out_csv, index=False)

    lines = [
        "PUB Station Selection Report",
        "=" * 72,
        f"Input station points (NC) : {inter_stats['n_nc']}",
        f"Obs cleaned GEV stations  : {inter_stats['n_obs_clean']}",
        f"Sim GEV stations          : {inter_stats['n_sim']}",
        f"Obs & Sim intersection  : {inter_stats['n_intersection']}",
        f"Assigned PUB stations     : {len(points_filtered)}",
        f"Filtered out stations     : {inter_stats['n_nc'] - len(points_filtered)}",
        f"Station source NC         : {NC_PATH}",
        f"Output CSV                : {out_csv}",
    ]

    report_path = OUT_DATA_06 / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info(f"Saved PUB assignment: {out_csv}")
    LOG.info(f"Saved report: {report_path}")


# ============================================================
# 4. MAIN
# ============================================================
def main() -> None:
    LOG.info("=" * 72)
    LOG.info("03_SpatialSplit_PUB_PUR start")
    LOG.info("=" * 72)

    points_all = load_station_points()
    points_filtered, inter_stats = filter_points_by_gev_intersection(points_all)

    # Stage A: PUR basin assignment.
    LOG.info("-" * 72)
    LOG.info("Stage A: PUR basin assignment (05_PUR_Basin_Select)")
    LOG.info("-" * 72)
    assign_df = assign_basins(points_filtered)
    save_outputs_pur(assign_df, n_input=len(points_filtered), inter_stats=inter_stats)
    plot_pur_selection_maps(points_all, points_filtered, assign_df)

    # Stage B: PUB train/val/test split.
    LOG.info("-" * 72)
    LOG.info("Stage B: PUB train/val/test split (06_PUB_Station_Select)")
    LOG.info("-" * 72)
    save_outputs_pub(points_filtered, inter_stats)
    plot_pub_selection_map(points_all, points_filtered)

    LOG.info("Done.")


if __name__ == "__main__":
    main()
