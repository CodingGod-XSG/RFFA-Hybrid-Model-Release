# -*- coding: utf-8 -*-
"""
05_GHM_LeakageMatch.py
================================================================================
Match our PUR (Prediction in Ungauged Regions) test stations against the GHM
(Global Hydrological Model, Ji et al., 2025) calibration/test station lists,
to quantify potential leakage between the GHM's own training data and the
stations used in this study's leakage-sensitivity analysis (manuscript
Figure S4, section 5 discussion: "1,251 of 4,447; 28.1%" overlap).

Matching is performed by extracting the raw numeric/alphanumeric station code
from each dataset's naming convention and intersecting the resulting code
sets (exact match). A secondary <1 km spatial-proximity check (via a KD-tree
over lat/lon) is computed purely as a diagnostic on unmatched stations, to
gauge how many additional stations *could* be considered leaked under a
looser criterion; it is reported/logged only and does NOT modify the
``matched_ghm_train`` / ``matched_ghm_test`` columns written to the output
CSV, which reflect exact-code matches only.

Inputs
------
    data/raw/Sim-Dis/ModelTrainingStation.csv
    data/raw/Sim-Dis/ModelTestStation.csv
        GHM calibration/test station lists (Ji et al., 2025).
        Required columns: STAID, LAT_GAGE, LNG_GAGE.
        STAID naming convention, e.g.:
            "HYSETS__08074000", "GRDC__4115400", "LamaH__203851", "USGS__01013500"

    data/proceed/Caravan-GRDC/05_PUR_Basin_Select/pur_retained_basins.shp
    data/proceed/Caravan-GRDC/05_PUR_Basin_Select/station_basin_assignment.csv
        This study's PUR basin selection and station-to-basin assignment.
        station_basin_assignment.csv naming convention, e.g.:
            "hysets_08074000", "GRDC_4115400", "lamah_203851"

Output
------
    data/proceed/Caravan-GRDC/05_GHM_LeakageMatch/
        pur_stations_exact_match.csv
            Columns: station_id, basin_label, lat, lon, code,
                     matched_ghm_train, matched_ghm_test
            Consumed by the Figure S4 script (GHM leakage analysis).
        report.txt
            Plain-text summary of the matching statistics (mirrors the
            console log).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.paths import DATA_RAW, DATA_PROCEED, stage_dir

# ============================================================
# 0. PATHS & CONFIG
# ============================================================
GHM_TRAIN_CSV = DATA_RAW / "Sim-Dis" / "ModelTrainingStation.csv"
GHM_TEST_CSV = DATA_RAW / "Sim-Dis" / "ModelTestStation.csv"

PUR_SHP = DATA_PROCEED / "05_PUR_Basin_Select" / "pur_retained_basins.shp"
STATION_ASSIGNMENT_CSV = (
    DATA_PROCEED / "05_PUR_Basin_Select" / "station_basin_assignment.csv"
)

OUT_DATA = stage_dir(DATA_PROCEED, "05_GHM_LeakageMatch")
OUT_CSV = OUT_DATA / "pur_stations_exact_match.csv"
OUT_REPORT = OUT_DATA / "report.txt"

SPATIAL_MATCH_KM = 1.0
DEG_TO_KM = 111.0  # crude constant-latitude approximation, matches source script

# ============================================================
# 1. LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger(__name__)


# ============================================================
# 2. CODE EXTRACTION HELPERS
# ============================================================
def extract_ghm_code(station_id: str) -> str:
    """Extract the raw station code from a GHM STAID value.

    GHM format uses a double-underscore prefix separator, e.g.:
        "HYSETS__08074000" -> "08074000"
        "GRDC__4115400"     -> "4115400"
        "LamaH__203851"     -> "203851"
        "USGS__01013500"    -> "01013500"
    """
    s = str(station_id).replace("__", "_")
    parts = s.split("_", 1)
    return parts[1].strip().lower() if len(parts) > 1 else s.strip().lower()


def extract_our_code(station_id: str) -> str:
    """Extract the raw station code from our station_id naming convention.

    Our format uses a single-underscore prefix separator, e.g.:
        "hysets_08074000" -> "08074000"
        "GRDC_4115400"     -> "4115400"
        "lamah_203851"     -> "203851"
    """
    s = str(station_id)
    parts = s.split("_", 1)
    return parts[1].strip().lower() if len(parts) > 1 else s.strip().lower()


# ============================================================
# 3. DATA LOADING
# ============================================================
def load_ghm_station_lists() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the GHM calibration/test station lists and attach extracted codes."""
    if not GHM_TRAIN_CSV.exists():
        raise FileNotFoundError(f"GHM training station list not found: {GHM_TRAIN_CSV}")
    if not GHM_TEST_CSV.exists():
        raise FileNotFoundError(f"GHM test station list not found: {GHM_TEST_CSV}")

    ghm_train = pd.read_csv(GHM_TRAIN_CSV)
    ghm_test = pd.read_csv(GHM_TEST_CSV)

    for name, df, path in [("GHM training", ghm_train, GHM_TRAIN_CSV),
                            ("GHM test", ghm_test, GHM_TEST_CSV)]:
        missing = {"STAID", "LAT_GAGE", "LNG_GAGE"} - set(df.columns)
        if missing:
            raise KeyError(f"{name} station list missing required column(s) {missing}: {path}")

    ghm_train = ghm_train.copy()
    ghm_test = ghm_test.copy()
    ghm_train["code"] = ghm_train["STAID"].apply(extract_ghm_code)
    ghm_test["code"] = ghm_test["STAID"].apply(extract_ghm_code)

    LOG.info("Loaded GHM training stations: %d", len(ghm_train))
    LOG.info("Loaded GHM test stations    : %d", len(ghm_test))
    return ghm_train, ghm_test


def load_pur_stations() -> pd.DataFrame:
    """Load PUR-retained basins and the station-to-basin assignment table,
    restricted to stations whose basin_label falls within the retained PUR set.
    """
    if not PUR_SHP.exists():
        raise FileNotFoundError(f"PUR retained basins shapefile not found: {PUR_SHP}")
    if not STATION_ASSIGNMENT_CSV.exists():
        raise FileNotFoundError(f"Station-basin assignment CSV not found: {STATION_ASSIGNMENT_CSV}")

    import geopandas as gpd

    pur_gdf = gpd.read_file(PUR_SHP)
    if "basin_labe" in pur_gdf.columns:
        pur_gdf = pur_gdf.rename(columns={"basin_labe": "basin_label"})
    if "basin_label" not in pur_gdf.columns:
        raise KeyError(f"'basin_label' column not found in {PUR_SHP}")
    pur_labels = set(pur_gdf["basin_label"].tolist())

    ba = pd.read_csv(STATION_ASSIGNMENT_CSV, keep_default_na=False)
    required_cols = {"station_id", "basin_label", "lat", "lon"}
    missing = required_cols - set(ba.columns)
    if missing:
        raise KeyError(f"Station-basin assignment CSV missing required column(s) {missing}: {STATION_ASSIGNMENT_CSV}")

    ba["basin_label"] = ba["basin_label"].str.strip()
    ba_pur = ba[ba["basin_label"].isin(pur_labels)].copy()
    if ba_pur.empty:
        raise RuntimeError(
            "No stations remain after restricting to PUR-retained basins. "
            f"Check consistency between {PUR_SHP} and {STATION_ASSIGNMENT_CSV}."
        )

    ba_pur["code"] = ba_pur["station_id"].apply(extract_our_code)

    LOG.info("Loaded PUR-retained basins  : %d", len(pur_labels))
    LOG.info("Loaded PUR stations         : %d", len(ba_pur))
    return ba_pur


# ============================================================
# 4. MATCHING
# ============================================================
def match_stations(
    ba_pur: pd.DataFrame, ghm_train: pd.DataFrame, ghm_test: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Match PUR stations against GHM train/test station codes.

    Returns the annotated ``ba_pur`` DataFrame (with ``matched_ghm_train`` /
    ``matched_ghm_test`` boolean columns, exact-code matches only) plus a
    list of human-readable summary lines for logging/reporting.
    """
    lines: list[str] = []

    ghm_train_codes = set(ghm_train["code"])
    ghm_test_codes = set(ghm_test["code"])
    our_codes = set(ba_pur["code"])

    exact_train = our_codes & ghm_train_codes
    exact_test = our_codes & ghm_test_codes

    n_pur = len(ba_pur)
    lines.append("=== Exact ID match ===")
    lines.append(
        f"Our PUR test vs GHM train : {len(exact_train):>4d} / {n_pur} "
        f"({100 * len(exact_train) / n_pur:.1f}%)"
    )
    lines.append(
        f"Our PUR test vs GHM test  : {len(exact_test):>4d} / {n_pur} "
        f"({100 * len(exact_test) / n_pur:.1f}%)"
    )

    ba_pur = ba_pur.copy()
    ba_pur["matched_ghm_train"] = ba_pur["code"].isin(ghm_train_codes)
    ba_pur["matched_ghm_test"] = ba_pur["code"].isin(ghm_test_codes)

    # Matched-GHM-train breakdown by data source (our prefix).
    lines.append("")
    lines.append("Matched GHM-train by data source (our prefix):")
    matched = ba_pur[ba_pur["matched_ghm_train"]]
    src = matched["station_id"].str.extract(r"^([a-zA-Z]+)")[0]
    lines.append(src.value_counts().to_string())

    # Matched-GHM-train breakdown per PUR basin.
    lines.append("")
    lines.append("Matched GHM-train per PUR basin:")
    for basin, grp in ba_pur.groupby("basin_label"):
        n = grp["matched_ghm_train"].sum()
        t = len(grp)
        lines.append(f"  {basin:<25s}: {n:>3d}/{t:>3d} ({100 * n / t:.0f}%)")

    # Diagnostic-only spatial proximity fallback among unmatched stations.
    # NOTE: this does NOT alter matched_ghm_train / matched_ghm_test.
    unmatched = ba_pur[~ba_pur["matched_ghm_train"]].copy()
    if len(unmatched) > 0 and len(ghm_train) > 0:
        tree = cKDTree(ghm_train[["LAT_GAGE", "LNG_GAGE"]].values)
        dist_deg, _ = tree.query(unmatched[["lat", "lon"]].values, k=1)
        dist_km = dist_deg * DEG_TO_KM
        n_spatial = int((dist_km <= SPATIAL_MATCH_KM).sum())
        lines.append("")
        lines.append(
            f"Additional spatial matches (<{SPATIAL_MATCH_KM:.0f} km) among unmatched: {n_spatial}"
        )
        total_matched = len(exact_train) + n_spatial
        lines.append(
            f"Total matched (ID + spatial): {total_matched} / {n_pur} "
            f"({100 * total_matched / n_pur:.1f}%)"
        )

    return ba_pur, lines


# ============================================================
# 5. OUTPUT
# ============================================================
def write_report(ba_pur: pd.DataFrame, summary_lines: list[str]) -> None:
    """Write the matched-station CSV and the plain-text summary report."""
    out_cols = [
        "station_id",
        "basin_label",
        "lat",
        "lon",
        "code",
        "matched_ghm_train",
        "matched_ghm_test",
    ]
    ba_pur[out_cols].to_csv(OUT_CSV, index=False)
    LOG.info("Saved matched stations -> %s", OUT_CSV)

    OUT_REPORT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    LOG.info("Saved report -> %s", OUT_REPORT)


# ============================================================
# 6. MAIN
# ============================================================
def main() -> None:
    LOG.info("=" * 72)
    LOG.info("05_GHM_LeakageMatch start")
    LOG.info("=" * 72)

    ghm_train, ghm_test = load_ghm_station_lists()
    ba_pur = load_pur_stations()
    ba_pur, summary_lines = match_stations(ba_pur, ghm_train, ghm_test)

    for line in summary_lines:
        LOG.info(line)

    write_report(ba_pur, summary_lines)

    LOG.info("Done.")


if __name__ == "__main__":
    main()
