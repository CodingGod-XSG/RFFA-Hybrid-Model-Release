# -*- coding: utf-8 -*-
"""Reusable helpers for PUR region precheck (pure data transforms)."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def normalize_station_ids(values: Iterable) -> pd.Series:
    """Normalize station IDs to trimmed strings for stable joins."""
    return pd.Series(values, dtype="object").astype(str).str.strip()


def load_assignment_table(basin_csv: str | bytes | "os.PathLike[str]") -> pd.DataFrame:
    """Load station-to-basin assignment with normalized station IDs."""
    df = pd.read_csv(basin_csv)
    required = {"station_id", "basin_label"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns in basin CSV: {sorted(miss)}")

    out = df.copy()
    out["station_id"] = normalize_station_ids(out["station_id"])
    out = out[out["station_id"] != ""].copy()

    if "continent" not in out.columns:
        out["continent"] = out["basin_label"].astype(str).str.split("_", n=1).str[0]
    else:
        out["continent"] = out["continent"].astype(str).str.strip()

    return out


def build_fold_table(assign_df: pd.DataFrame,
                     stations: Iterable,
                     min_fold_n: int) -> pd.DataFrame:
    """Create per-basin fold table and available flag under threshold."""
    station_set = set(normalize_station_ids(stations).tolist())
    df = assign_df[assign_df["station_id"].isin(station_set)].copy()

    grp = (
        df.groupby(["basin_label", "continent"], dropna=False)["station_id"]
        .nunique()
        .reset_index(name="n_stations")
        .sort_values("n_stations", ascending=False)
        .reset_index(drop=True)
    )
    grp["available"] = grp["n_stations"] >= int(min_fold_n)
    return grp


def build_threshold_table(fold_df: pd.DataFrame,
                          thresholds: Iterable[int]) -> pd.DataFrame:
    """Compute available fold/station counts for a sequence of thresholds."""
    rows = []
    n_total_folds = int(len(fold_df))
    n_total_stn = int(fold_df["n_stations"].sum()) if n_total_folds else 0

    for th in thresholds:
        m = fold_df["n_stations"] >= int(th)
        n_f = int(m.sum())
        n_s = int(fold_df.loc[m, "n_stations"].sum())
        rows.append(
            {
                "threshold": int(th),
                "n_available_folds": n_f,
                "n_available_stations": n_s,
                "pct_folds": (100.0 * n_f / n_total_folds) if n_total_folds else 0.0,
                "pct_stations": (100.0 * n_s / n_total_stn) if n_total_stn else 0.0,
            }
        )

    return pd.DataFrame(rows)


def summarize_fold_table(fold_df: pd.DataFrame, min_fold_n: int) -> dict:
    """Return compact summary stats for report writing."""
    total_folds = int(len(fold_df))
    total_stn = int(fold_df["n_stations"].sum()) if total_folds else 0
    kept = fold_df[fold_df["available"]].copy()
    kept_folds = int(len(kept))
    kept_stn = int(kept["n_stations"].sum()) if kept_folds else 0
    return {
        "min_fold_n": int(min_fold_n),
        "total_folds": total_folds,
        "total_stations": total_stn,
        "kept_folds": kept_folds,
        "kept_stations": kept_stn,
        "dropped_folds": total_folds - kept_folds,
        "dropped_stations": total_stn - kept_stn,
    }
