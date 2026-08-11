# -*- coding: utf-8 -*-
"""
4_12_base/splits.py
Index-based split helpers (no data copying).

Functions
---------
pub_split(n, seed)
    → (tr_idx, val_idx, te_idx)  numpy int arrays

pur_splits(basin_csv, stations)
    → dict {fold_label: te_idx_array}

pur_train_val_split(all_idx, te_idx, seed, val_ratio=0.15)
    → (tr_idx, val_idx)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TEST_RATIO = 0.20
VAL_RATIO  = 0.15
MIN_FOLD_N = 50


def pub_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng   = np.random.RandomState(seed)
    idx   = rng.permutation(n)
    n_te  = max(1, int(n * TEST_RATIO))
    n_val = max(1, int((n - n_te) * VAL_RATIO))
    te_idx  = idx[:n_te]
    val_idx = idx[n_te: n_te + n_val]
    tr_idx  = idx[n_te + n_val:]
    return tr_idx, val_idx, te_idx


def pur_splits(basin_csv: Path,
               stations: np.ndarray,
               min_fold_n: int = MIN_FOLD_N) -> dict[str, np.ndarray]:
    """
    Returns {fold_label: test_indices_into_stations}.
    Uses 'basin_label' column from station_basin_assignment.csv.
    Only folds with >= min_fold_n stations (present in `stations`) are included.
    """
    df = pd.read_csv(basin_csv)
    df["station_id"] = df["station_id"].astype(str)
    df = df[df["station_id"].isin(stations)].copy()
    stn_to_i = {s: i for i, s in enumerate(stations)}
    df["idx"] = df["station_id"].map(stn_to_i)
    folds = {}
    for label, grp in df.groupby("basin_label"):
        idx = grp["idx"].dropna().astype(int).values
        if len(idx) >= min_fold_n:
            folds[str(label)] = idx
    return folds


def pur_train_val_split(all_idx: np.ndarray,
                        te_idx: np.ndarray,
                        seed: int,
                        val_ratio: float = VAL_RATIO
                        ) -> tuple[np.ndarray, np.ndarray]:
    te_set   = set(te_idx.tolist())
    pool     = np.array([i for i in all_idx if i not in te_set])
    if len(pool) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    rng      = np.random.RandomState(seed)
    rng.shuffle(pool)
    if len(pool) == 1:
        return pool.copy(), np.array([], dtype=int)

    n_val    = max(1, int(len(pool) * val_ratio))
    n_val    = min(n_val, len(pool) - 1)
    val_idx  = pool[:n_val]
    tr_idx   = pool[n_val:]
    return tr_idx, val_idx
