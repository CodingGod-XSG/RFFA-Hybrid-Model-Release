# -*- coding: utf-8 -*-
"""
FigureS2S3_FeatureSpace_PUB_PUR.py
================================================================================
Reproduces Figures S2-S3 -- feature-space diagnostics contrasting PUB (random
split) vs PUR (regional holdout): PCA scatter of catchment attribute space,
Mahalanobis distance from test stations to the training distribution vs
per-station NSE for RF and ANN-Joint, and Q100 CDF comparison, confirming the
PUB-PUR gap reflects genuine covariate shift.

PUB strategy: random 20 % test split — test stations scatter throughout the same
              attribute distribution as training stations.
PUR strategy: one HydroSHEDS Level-2 basin held out entirely — test stations are
              geographically clustered in a region unseen during training, producing
              a systematic covariate shift.

Panels produced
---------------
  (a) PCA scatter – PUB: PC1/PC2, train (gray) vs test (red)
  (b) PCA scatter – PUR overview: PC1/PC2, train pool (gray) + each fold's
      test stations coloured by region (all folds on one axis)
  (c) Mahalanobis distance boxplot/CDF: PUB test vs PUR test (per fold)
  (d) Distance vs per-station log-NSE: RF vs ANN-Joint LOWESS trend lines
  (e) Q100 CDF: all-train / PUB test / PUR test (pooled)

Supplementary
  pca_pur_grid.png – small-multiples: one subplot per PUR fold,
                     each fold's own training set (gray) vs test (coloured)

Outputs
-------
  data/proceed/Caravan-GRDC/FigureS2S3_FeatureSpace_PUB_PUR/
      feature_distances.csv   – per-station: mahal_dist, nse_rf, nse_ann, Q100_true
      report.txt
  figures/Caravan-GRDC/FigureS2S3_FeatureSpace_PUB_PUR/
      FigureS2S3_main.png
      FigureS2S3_pca_pur_grid.png
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIG & PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from src.paths import DATA_PROCEED, FIGURE_ROOT, stage_dir

TAG = "FigureS2S3_FeatureSpace_PUB_PUR"

NC_PATH   = DATA_PROCEED / "02_Data-Clean" / "4_Cara-GRDC-35_cleaned.nc"
GEV_CSV   = DATA_PROCEED / "01_GEV-Fit"   / "gev_station_params.csv"
FLOW_CSV  = DATA_PROCEED / "03_Streamflow-Process" / "sim_flow_features_per_station.csv"
BASIN_CSV = DATA_PROCEED / "05_PUR_Basin_Select"   / "station_basin_assignment.csv"

RF_DATA_DIR  = DATA_PROCEED / "06_RF"
ANN_DATA_DIR = DATA_PROCEED / "09_ANN"
XGB_DATA_DIR = DATA_PROCEED / "08_XGBoost"

SPLIT_SEEDS   = [42, 123, 456]
MIN_PUR_N     = 50
REP_SEED      = SPLIT_SEEDS[0]        # representative seed for PUB PCA panel
FLOW_TAG      = "+flow"               # feature variant for Mahalanobis distance
NSE_FTAG      = "base"                # feature variant for NSE (matches Figure3_ML_Performance.py)
USE_FLOW_FEAT = False                 # False → X_base (36 static) for PCA/Mahal
MAX_PUR_FOLDS_IN_OVERVIEW = 15        # cap colours in PUR overview panel
PCA_REPS      = 2                     # number of PCA components to show

OUT_DATA = stage_dir(DATA_PROCEED, TAG)
OUT_FIG  = stage_dir(FIGURE_ROOT, TAG)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
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
# 2.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> dict:
    sys.path.insert(0, str(SCRIPT_DIR))
    from src.dataset import DatasetBuilder
    return DatasetBuilder(NC_PATH, GEV_CSV, FLOW_CSV).build()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SPLIT RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_splits(data: dict) -> tuple[dict, dict]:
    """
    Returns
    -------
    pub_splits : {seed: (tr_idx, val_idx, te_idx)}
    pur_folds  : {fold_label: te_idx}   (test indices into data arrays)
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from src.splits import pub_split, pur_splits

    N = len(data["stations"])
    pub = {seed: pub_split(N, seed) for seed in SPLIT_SEEDS}
    pur = pur_splits(BASIN_CSV, data["stations"], min_fold_n=MIN_PUR_N)
    LOG.info(f"PUB splits: {len(pub)}  PUR folds: {len(pur)}")
    return pub, pur


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PCA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _feature_matrix(data: dict) -> np.ndarray:
    return data["X_full"] if USE_FLOW_FEAT else data["X_base"]


def compute_pca(X_tr: np.ndarray, X_all: np.ndarray, n_components: int = 2):
    """Fit PCA on training set; transform all stations."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_all_s = scaler.transform(X_all)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_tr_s)
    return pca.transform(X_all_s), pca.explained_variance_ratio_


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAHALANOBIS DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mahal(X_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf regularised Mahalanobis distance from each test point
    to the training distribution.
    """
    from sklearn.covariance import LedoitWolf
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    lw = LedoitWolf().fit(X_tr_s)
    mu   = lw.location_
    prec = lw.precision_
    diffs = X_te_s - mu
    return np.sqrt(np.einsum("ni,ij,nj->n", diffs, prec, diffs))


def build_mahal_table(data: dict, pub_splits: dict, pur_folds: dict) -> pd.DataFrame:
    """Per-station Mahalanobis distance for PUB (seed=REP_SEED) and all PUR folds."""
    X = _feature_matrix(data)
    rows = []

    # PUB
    tr_idx, _, te_idx = pub_splits[REP_SEED]
    dists = _compute_mahal(X[tr_idx], X[te_idx])
    for i, d in zip(te_idx, dists):
        rows.append(dict(
            station_id = data["stations"][i],
            split_type = "PUB",
            fold       = "PUB",
            mahal_dist = float(d),
            q100_true  = float(data["q_true"][i, 5]),
        ))

    # PUR
    all_idx = np.arange(len(data["stations"]))
    for fold_label, te_idx in sorted(pur_folds.items()):
        te_set  = set(te_idx.tolist())
        tr_pool = np.array([i for i in all_idx if i not in te_set])
        dists   = _compute_mahal(X[tr_pool], X[te_idx])
        for i, d in zip(te_idx, dists):
            rows.append(dict(
                station_id = data["stations"][i],
                split_type = "PUR",
                fold       = fold_label,
                mahal_dist = float(d),
                q100_true  = float(data["q_true"][i, 5]),
            ))

    df = pd.DataFrame(rows)
    LOG.info(f"Mahalanobis table: {len(df)} rows  "
             f"({(df.split_type=='PUB').sum()} PUB, {(df.split_type=='PUR').sum()} PUR)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LOAD MODEL PREDICTIONS → PER-STATION LOG-NSE
# ─────────────────────────────────────────────────────────────────────────────

RETURN_PERIODS = [2, 5, 10, 20, 50, 100]
_SAFE = lambda s: s.replace("-", "_").replace(" ", "_")


def _glob_pred_files(pred_dir: Path, model_tag: str, exp: str,
                     fold: str, flow: str) -> list[Path]:
    # RF filenames contain both train_seed and split_seed; ANN only split_seed
    safe_fold = _SAFE(fold)
    pattern   = f"predictions_{_SAFE(model_tag)}_{exp}_{safe_fold}*{flow}.csv"
    return list(pred_dir.glob(pattern))


def _avg_predictions(files: list[Path]) -> Optional[pd.DataFrame]:
    if not files:
        return None
    dfs_idx = []
    for f in files:
        df = pd.read_csv(f)
        if "station_id" not in df.columns:
            continue
        df = df.set_index("station_id")
        dfs_idx.append(df)
    if not dfs_idx:
        return None

    q_pred_cols = [c for c in dfs_idx[0].columns if c.endswith("_pred")]
    meta_cols   = [c for c in dfs_idx[0].columns if not c.endswith("_pred")]

    avg = (pd.concat([d[q_pred_cols] for d in dfs_idx], axis=0)
             .groupby(level=0).mean())
    result = dfs_idx[0][meta_cols].copy()
    result = result.join(avg)
    return result.reset_index()


def _station_nse(df: pd.DataFrame) -> pd.Series:
    """
    Per-station log-NSE across all 6 return periods.
    NSE = 1 - MSE_log / Var_log(true)
    """
    lp = np.stack([np.log(df[f"Q{T}_pred"].clip(lower=1e-6).values)
                   for T in RETURN_PERIODS], axis=1)
    lt = np.stack([np.log(df[f"Q{T}_true"].clip(lower=1e-6).values)
                   for T in RETURN_PERIODS], axis=1)
    ss_res = ((lp - lt) ** 2).sum(axis=1)
    mean_lt = lt.mean(axis=1, keepdims=True)
    ss_tot  = ((lt - mean_lt) ** 2).sum(axis=1)
    nse = 1.0 - ss_res / np.where(ss_tot < 1e-12, 1e-12, ss_tot)
    return pd.Series(nse, index=df["station_id"].values, name="nse")


def load_model_nse(model_label: str, pred_dir: Path,
                   pub_splits_dict: dict, pur_folds: dict) -> dict[str, pd.Series]:
    """
    Returns {station_id → nse} for PUB and each PUR fold.
    Uses REP_SEED for PUB; aggregates across all available seed CSVs.
    """
    result = {}

    # PUB
    files = _glob_pred_files(pred_dir, model_label, "PUB", "PUB", FLOW_TAG)
    if files:
        df = _avg_predictions(files)
        if df is not None and not df.empty:
            result["PUB"] = _station_nse(df)
            LOG.info(f"  {model_label} PUB predictions: {len(df)} stations from {len(files)} files")
    else:
        LOG.warning(f"  {model_label}: no PUB prediction files found in {pred_dir}")

    # PUR
    for fold_label in sorted(pur_folds.keys()):
        safe_fold = _SAFE(fold_label)
        files = _glob_pred_files(pred_dir, model_label, "PUR", safe_fold, FLOW_TAG)
        if not files:
            continue
        df = _avg_predictions(files)
        if df is not None and not df.empty:
            result[fold_label] = _station_nse(df)

    LOG.info(f"  {model_label}: NSE loaded for PUB + {len(result)-1} PUR folds")
    return result


def attach_nse_to_mahal(mahal_df: pd.DataFrame,
                        rf_nse:  dict[str, pd.Series],
                        ann_nse: dict[str, pd.Series]) -> pd.DataFrame:
    """Add nse_rf and nse_ann columns to the Mahalanobis distance table."""
    df = mahal_df.copy()
    df["nse_rf"]  = np.nan
    df["nse_ann"] = np.nan

    for split_type, fold in df[["split_type", "fold"]].drop_duplicates().values:
        key = fold if fold in rf_nse else None
        mask = (df["split_type"] == split_type) & (df["fold"] == fold)
        sids = df.loc[mask, "station_id"].values

        if key and key in rf_nse:
            nse = rf_nse[key].reindex(sids)
            df.loc[mask, "nse_rf"] = nse.values

        if key and key in ann_nse:
            nse = ann_nse[key].reindex(sids)
            df.loc[mask, "nse_ann"] = nse.values

    n_rf  = df["nse_rf"].notna().sum()
    n_ann = df["nse_ann"].notna().sum()
    LOG.info(f"NSE attached: RF={n_rf}, ANN={n_ann} out of {len(df)} rows")
    return df


def load_fold_pool_nse(model_label: str, pred_dir: Path,
                       pur_folds: dict) -> dict[str, float]:
    """
    Per-fold cross-station Q100 pool-NSE (base predictions, linear space).
    Matches the metric used in Figure3_ML_Performance.py.
    """
    result: dict[str, float] = {}
    for fold_label in sorted(pur_folds.keys()):
        files = _glob_pred_files(pred_dir, model_label, "PUR",
                                 _SAFE(fold_label), NSE_FTAG)
        if not files:
            continue
        df = _avg_predictions(files)
        if df is None or df.empty:
            continue
        if "Q100_true" not in df.columns or "Q100_pred" not in df.columns:
            continue
        qt = df["Q100_true"].clip(lower=1e-6).values
        qp = df["Q100_pred"].clip(lower=1e-6).values
        ok = np.isfinite(qt) & np.isfinite(qp) & (qt > 0)
        if ok.sum() < 5:
            continue
        t, p = qt[ok], qp[ok]
        ss = float(np.sum((t - t.mean()) ** 2))
        nse = float(1 - np.sum((t - p) ** 2) / ss) if ss > 0 else np.nan
        result[fold_label] = nse

    LOG.info(f"  {model_label} fold pool-NSE: {len(result)} folds")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PUBLICATION STYLE & PALETTE
# ─────────────────────────────────────────────────────────────────────────────

# ── Nature PALETTE (from nature-figure skill api.md) ────────────────────────
# Semantic: train = neutral, pub = blue (familiar), pur = red (challenge),
#           rf = teal (baseline), ann = violet (comparison method)
C = dict(
    train    = "#CFCECE",   # neutral_light
    pub_test = "#0F4D92",   # blue_main
    pur_test = "#B64342",   # red_strong
    rf       = "#42949E",   # teal
    ann      = "#9A4D8E",   # violet
    ref      = "#767676",   # neutral_mid
)

FOLD_COLORS = [
    "#1B9E77","#D95F02","#7570B3","#E7298A","#66A61E",
    "#E6AB02","#A6761D","#4393C3","#B2DF8A","#FB9A99",
    "#F781BF","#A6CEE3","#B15928","#6A3D9A","#33A02C",
    "#FF7F00","#CAB2D6","#FDBF6F","#B2182B","#4DAF4A",
]


def _fold_palette(folds: list[str]) -> dict[str, str]:
    return {f: FOLD_COLORS[i % len(FOLD_COLORS)] for i, f in enumerate(folds)}


def _build_fold_style(fold_labels: list[str]) -> tuple[dict, dict]:
    """
    Single source of truth for fold short names and colors —
    must be called once and passed to every panel that shows PUR regions
    so that the SAME fold always gets the SAME color and the SAME label.

    Returns
    -------
    short_names : {"EU_2020018240": "EU2", "SA_6020021870": "SA4", ...}
    palette     : {"EU_2020018240": "#hex", ...}
    """
    from collections import defaultdict
    sorted_folds = sorted(fold_labels)

    cont_groups: dict[str, list] = defaultdict(list)
    for f in sorted_folds:
        cont = f.split("_")[0] if "_" in f else "XX"
        cont_groups[cont].append(f)

    short_names: dict[str, str] = {}
    for cont in sorted(cont_groups.keys()):
        for i, fold in enumerate(sorted(cont_groups[cont]), 1):
            short_names[fold] = f"{cont}{i}"

    palette: dict[str, str] = {
        f: FOLD_COLORS[i % len(FOLD_COLORS)]
        for i, f in enumerate(sorted_folds)
    }
    return short_names, palette


def _setup_pub_style() -> None:
    plt.rcParams.update({
        # ── MANDATORY: editable text in SVG/PDF (Nature requirement) ──────────
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype":          "none",   # keeps text as <text> nodes, not paths
        "pdf.fonttype":          42,       # TrueType embedding, editable in Illustrator
        # ── Font sizes: WRR supplement figure (~7 in wide) ───────────────────
        "font.size":             8.5,
        "axes.labelsize":        8.5,
        "axes.titlesize":        9.0,
        "axes.titleweight":      "bold",
        # ── Spines & axes ─────────────────────────────────────────────────────
        "axes.linewidth":        0.8,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        # ── Ticks: inward, consistent weight ─────────────────────────────────
        "xtick.labelsize":       8.0,
        "ytick.labelsize":       8.0,
        "xtick.major.size":      3.0,
        "ytick.major.size":      3.0,
        "xtick.minor.size":      1.8,
        "xtick.major.width":     0.7,
        "ytick.major.width":     0.7,
        "xtick.direction":       "in",
        "ytick.direction":       "in",
        # ── Legend: frameless ────────────────────────────────────────────────
        "legend.fontsize":       8.0,
        "legend.frameon":        False,
        "legend.handlelength":   1.4,
        "legend.handleheight":   0.8,
        "legend.labelspacing":   0.3,
        # ── Lines & patches ──────────────────────────────────────────────────
        "lines.linewidth":       1.3,
        "patch.linewidth":       0.7,
        # ── Save defaults ────────────────────────────────────────────────────
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.05,
    })


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _light_grid(ax, axis: str = "y") -> None:
    ax.set_axisbelow(True)
    kw = dict(linestyle="--", linewidth=0.45, color="#E8E8E8", zorder=0)
    if axis in ("y", "both"):
        ax.yaxis.grid(True, **kw)
    if axis in ("x", "both"):
        ax.xaxis.grid(True, **kw)


def _panel_label(ax, label: str, x: float = -0.08, y: float = 1.05) -> None:
    """Nature-style panel label: small bold lowercase, top-left edge."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left",
            fontfamily="sans-serif")


def _confidence_ellipse(ax, x: np.ndarray, y: np.ndarray,
                        n_std: float = 2.0, **kwargs) -> None:
    """Draw a covariance-based confidence ellipse (eigenvalue decomposition)."""
    from matplotlib.patches import Ellipse
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2.0 * n_std * np.sqrt(np.abs(vals))
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ell)


def _binned_stats(x: np.ndarray, y: np.ndarray,
                  n_bins: int = 12) -> tuple[np.ndarray, ...]:
    """
    Bin x into n_bins equal-count bins; return bin centres, medians,
    25th-percentile, 75th-percentile of y.  Bins with < 5 valid points skipped.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    if len(xm) < n_bins * 5:
        n_bins = max(4, len(xm) // 10)
    edges  = np.percentile(xm, np.linspace(0, 100, n_bins + 1))
    edges  = np.unique(edges)
    idx    = np.digitize(xm, edges, right=False) - 1
    idx    = np.clip(idx, 0, len(edges) - 2)

    ctrs, meds, q25s, q75s = [], [], [], []
    for i in range(len(edges) - 1):
        sel = ym[idx == i]
        if len(sel) < 5:
            continue
        ctrs.append(np.median(xm[idx == i]))
        meds.append(np.median(sel))
        q25s.append(np.percentile(sel, 25))
        q75s.append(np.percentile(sel, 75))
    return (np.array(ctrs), np.array(meds),
            np.array(q25s), np.array(q75s))


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PUBLICATION-QUALITY PANEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _pc_xlabel(evr, i=0): return f"PC{i+1} ({evr[i]*100:.1f}% variance)"
def _pc_ylabel(evr, i=1): return f"PC{i+1} ({evr[i]*100:.1f}% variance)"


# ── Panel A: PCA – PUB ────────────────────────────────────────────────────────
def panel_pca_pub(ax, data: dict, pub_splits_dict: dict) -> None:
    X = _feature_matrix(data)
    tr_idx, _, te_idx = pub_splits_dict[REP_SEED]
    coords, evr = compute_pca(X[tr_idx], X, n_components=2)
    xtr, ytr = coords[tr_idx, 0], coords[tr_idx, 1]
    xte, yte = coords[te_idx, 0], coords[te_idx, 1]

    # Training: small gray dots + 1σ / 2σ ellipses
    ax.scatter(xtr, ytr, s=3, c=C["train"], alpha=0.30, linewidths=0,
               zorder=1, label=f"Training  (n = {len(tr_idx):,})")
    _confidence_ellipse(ax, xtr, ytr, n_std=1.0,
                        facecolor=C["train"], alpha=0.18,
                        edgecolor="#888888", linewidth=0.9, linestyle="-",
                        zorder=2)
    _confidence_ellipse(ax, xtr, ytr, n_std=2.0,
                        facecolor="none",
                        edgecolor="#888888", linewidth=0.9, linestyle="--",
                        zorder=2)

    # PUB test: coloured dots + ellipse
    ax.scatter(xte, yte, s=12, c=C["pub_test"], alpha=0.65, linewidths=0,
               zorder=3, label=f"PUB test  (n = {len(te_idx):,})")
    _confidence_ellipse(ax, xte, yte, n_std=2.0,
                        facecolor="none",
                        edgecolor=C["pub_test"], linewidth=1.1,
                        linestyle="--", zorder=4)

    ax.set_xlabel(_pc_xlabel(evr)); ax.set_ylabel(_pc_ylabel(evr))
    ax.set_title("PCA – PUB (random split)")
    leg = ax.legend(loc="upper right", markerscale=1.6,
                    handletextpad=0.4, borderaxespad=0.5)
    _despine(ax)
    _panel_label(ax, "a")


# ── Panel B: PCA – PUR overview ───────────────────────────────────────────────
def panel_pca_pur(ax, data: dict, pub_splits_dict: dict, pur_folds: dict,
                  fold_styles: tuple) -> None:
    short_names, palette = fold_styles
    X = _feature_matrix(data)
    tr_idx, _, _ = pub_splits_dict[REP_SEED]
    coords, evr  = compute_pca(X[tr_idx], X, n_components=2)
    xtr, ytr = coords[tr_idx, 0], coords[tr_idx, 1]

    ax.scatter(xtr, ytr, s=2, c=C["train"], alpha=0.18, linewidths=0,
               zorder=1, label="Training", rasterized=True)
    _confidence_ellipse(ax, xtr, ytr, n_std=2.0, facecolor="none",
                        edgecolor="#AAAAAA", linewidth=1.0,
                        linestyle="--", zorder=2)

    for fold_label in sorted(pur_folds.keys())[:MAX_PUR_FOLDS_IN_OVERVIEW]:
        te_idx = pur_folds[fold_label]
        ax.scatter(coords[te_idx, 0], coords[te_idx, 1],
                   s=10, c=palette[fold_label], alpha=0.75, linewidths=0,
                   zorder=3, label=short_names[fold_label], rasterized=True)

    ax.set_xlabel(_pc_xlabel(evr)); ax.set_ylabel(_pc_ylabel(evr))
    ax.set_title("PCA – PUR (regional holdout, all folds)")
    ax.legend(loc="upper left", fontsize=6.5, ncol=2,
              markerscale=1.5, handletextpad=0.3,
              borderaxespad=0.3, labelspacing=0.25)
    _despine(ax)
    _panel_label(ax, "b")


# ── Panel A: Mahalanobis per-fold horizontal boxplot ─────────────────────────
def panel_mahal_folds(ax, dist_df: pd.DataFrame, pur_folds: dict,
                      fold_styles: tuple) -> None:
    from scipy.stats import mannwhitneyu
    short_names, palette = fold_styles

    pub_d = dist_df.loc[dist_df["split_type"] == "PUB", "mahal_dist"].dropna().values
    folds = sorted(pur_folds.keys())
    fold_meds = {f: np.median(dist_df.loc[dist_df["fold"] == f,
                                           "mahal_dist"].dropna().values)
                 for f in folds}
    folds_sorted = sorted(folds, key=lambda f: fold_meds[f])

    all_data   = [pub_d] + [
        dist_df.loc[dist_df["fold"] == f, "mahal_dist"].dropna().values
        for f in folds_sorted
    ]
    all_labels = ["PUB test"] + [short_names[f] for f in folds_sorted]
    all_colors = [C["pub_test"]] + [palette[f] for f in folds_sorted]
    n_counts   = [len(d) for d in all_data]

    # ── Clip x-axis at 97th percentile – removes distracting outlier scatter ──
    all_vals = np.concatenate(all_data)
    x_clip   = float(np.percentile(all_vals, 97))
    n_out    = int(np.sum(all_vals > x_clip))

    bp = ax.boxplot(
        all_data, vert=False, patch_artist=True,
        showfliers=False,          # hide only extreme outlier dots
        widths=0.60,
        medianprops=dict(color="white", linewidth=1.8),
        whiskerprops=dict(linewidth=0.8, color="#555555"),
        capprops=dict(linewidth=0.8, color="#555555"),
        boxprops=dict(linewidth=0.7),
    )
    for patch, col in zip(bp["boxes"], all_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.82)

    # PUB median reference line
    ax.axvline(np.median(pub_d), color=C["pub_test"],
               lw=0.7, ls=":", alpha=0.55, zorder=0)

    # Separator between PUB and PUR groups
    ax.axhline(1.5, color="#CCCCCC", lw=0.5, ls="-", zorder=0)

    # Clip x at IQR-based range (only show box bodies, no far-right space)
    q75_all = np.percentile(all_vals, 75)
    iqr_all = q75_all - np.percentile(all_vals, 25)
    x_show  = q75_all + 2.5 * iqr_all
    ax.set_xlim(-0.3, min(x_show, x_clip))

    ax.set_yticks(range(1, len(all_labels) + 1))
    ax.set_yticklabels(all_labels, fontsize=6.5)
    ax.set_xlabel("Mahalanobis distance to training set")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))

    # MWU p-value as small annotation (bottom-right, no title)
    pur_d = np.concatenate([d for d in all_data[1:]])
    _, pval = mannwhitneyu(pub_d, pur_d, alternative="less")
    pstr = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
    ax.text(0.97, 0.02, f"MWU {pstr}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=C["ref"])

    _despine(ax)
    _light_grid(ax, "x")


# ── Panel B: Distance vs model performance ───────────────────────────────────
def panel_dist_nse(ax, dist_df: pd.DataFrame,
                   xgb_fold_nse: dict[str, float],
                   ann_fold_nse: dict[str, float],
                   fold_styles: tuple) -> None:
    """
    Paired dot plot — each PUR fold at its median Mahalanobis distance.
    XGBoost (circle) and ANN (square) connected by a thin line:
      green  = ANN > XGBoost  (ANN extrapolates better)
      orange = XGBoost > ANN
    Fold short names annotated above each pair.
    x = fold median Mahalanobis distance (base features, Ledoit-Wolf)
    y = cross-station Q100 pool-NSE  (base predictions, linear space)
    """
    short_names, _palette = fold_styles

    fold_med_dist = (dist_df[dist_df["split_type"] == "PUR"]
                     .groupby("fold")["mahal_dist"].median().to_dict())

    C_ANN_WIN = "#27AE60"   # green  – ANN wins
    C_XGB_WIN = "#E67E22"   # orange – XGBoost wins
    Y_FLOOR   = -1.05       # clip extreme outliers for readability

    common = sorted(
        [f for f in xgb_fold_nse
         if f in ann_fold_nse and f in fold_med_dist
         and np.isfinite(xgb_fold_nse[f]) and np.isfinite(ann_fold_nse[f])],
        key=lambda f: fold_med_dist[f],
    )

    # ── Light background: "high-extrapolation" zone (dist > 6.5) ─────────────
    x_vals = [fold_med_dist[f] for f in common]
    x_hi = max(x_vals) + 0.4
    ax.axvspan(6.5, x_hi, color="#F5F0EC", alpha=1.0, zorder=0)
    ax.text(6.55, 1.00, "high extrapolation", ha="left", va="top",
            fontsize=7.0, color="#333333", style="italic")

    # Track clipped folds for annotation
    clipped_folds = []

    for i, fold in enumerate(common):
        x      = fold_med_dist[fold]
        y_xgb  = xgb_fold_nse[fold]
        y_ann  = ann_fold_nse[fold]
        y_xgb_c = max(y_xgb, Y_FLOOR)
        y_ann_c = max(y_ann, Y_FLOOR)
        win_col = C_ANN_WIN if y_ann >= y_xgb else C_XGB_WIN

        if y_xgb < Y_FLOOR or y_ann < Y_FLOOR:
            clipped_folds.append((fold, x, y_xgb, y_ann))

        # Connector line (thicker, more visible)
        ax.plot([x, x], [y_xgb_c, y_ann_c],
                color=win_col, lw=2.2, alpha=0.72, zorder=2,
                solid_capstyle="round")

        # Model dots with white outline
        ax.scatter(x, y_xgb_c, s=42, color=C["rf"],  marker="o", zorder=4,
                   linewidths=0.8, edgecolors="white")
        ax.scatter(x, y_ann_c, s=42, color=C["ann"], marker="s", zorder=4,
                   linewidths=0.8, edgecolors="white")

        # Staggered fold labels: odd→above, even→below, only for high-dist or extreme folds
        sname = short_names.get(fold, fold)
        y_top = max(y_xgb_c, y_ann_c)
        y_bot = min(y_xgb_c, y_ann_c)
        above = (i % 2 == 0)
        if above:
            ax.text(x, y_top + 0.07, sname, ha="center", va="bottom",
                    fontsize=6.5, color="#444444")
        else:
            ax.text(x, y_bot - 0.07, sname, ha="center", va="top",
                    fontsize=6.5, color="#444444")

    # Annotate clipped folds at floor
    for fold, x, y_rf, y_ann in clipped_folds:
        sname = short_names.get(fold, fold)
        worst = min(y_rf, y_ann)
        ax.annotate(f"{sname}: {worst:.2f}",
                    xy=(x, Y_FLOOR), xytext=(x + 0.2, Y_FLOOR + 0.08),
                    fontsize=6.5, color=C["ref"],
                    arrowprops=dict(arrowstyle="-", color=C["ref"], lw=0.5))

    # Reference lines (no text labels)
    ax.axhline(0.0, color="#888888", lw=0.9, ls="--", zorder=1, alpha=0.8)
    ax.axhline(0.5, color="#AAAAAA", lw=0.7, ls=":",  zorder=1, alpha=0.8)

    # Legend: model shapes only (win/lose shown by connector color)
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["rf"],
               markersize=5.5, markeredgecolor="white", label="XGBoost"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C["ann"],
               markersize=5.5, markeredgecolor="white", label="ANN"),
    ]
    ax.legend(handles=legend_els, loc="lower left", ncol=1,
              handlelength=0.9, handletextpad=0.35, borderaxespad=0.4)

    # Remove NSE reference text (lines remain, no text clutter)
    ax.axhline(0.0, color="#888888", lw=0.9, ls="--", zorder=1, alpha=0.8)
    ax.axhline(0.5, color="#AAAAAA", lw=0.7, ls=":",  zorder=1, alpha=0.8)

    ax.set_xlim(min(x_vals) - 0.5, x_hi)
    ax.set_ylim(Y_FLOOR - 0.05, 1.10)
    ax.set_xlabel("Fold median Mahalanobis distance to training set")
    ax.set_ylabel(r"Q$_{100}$ pool-NSE (per fold)")
    _despine(ax)


# ── Panel D: Test set performance – PUB vs PUR × RF vs ANN ──────────────────
def panel_test_nse_comparison(ax, dist_df: pd.DataFrame) -> None:
    """
    Violin plots comparing log-NSE on PUB test vs PUR test for RF and ANN.

    Layout (4 violins):
        pos 1  RF / PUB  (blue)
        pos 2  RF / PUR  (red)
        ── gap ──
        pos 4  ANN / PUB (blue)
        pos 5  ANN / PUR (red)
    """
    import matplotlib.transforms as transforms
    from matplotlib.patches import Patch

    NSE_CLIP = -2.0
    pub = dist_df[dist_df["split_type"] == "PUB"]
    pur = dist_df[dist_df["split_type"] == "PUR"]

    groups = [
        (pub["nse_rf"].dropna().clip(lower=NSE_CLIP).values,  1, C["pub_test"]),
        (pur["nse_rf"].dropna().clip(lower=NSE_CLIP).values,  2, C["pur_test"]),
        (pub["nse_ann"].dropna().clip(lower=NSE_CLIP).values, 4, C["pub_test"]),
        (pur["nse_ann"].dropna().clip(lower=NSE_CLIP).values, 5, C["pur_test"]),
    ]

    for vals, pos, colour in groups:
        if len(vals) == 0:
            continue
        vp = ax.violinplot([vals], positions=[pos], widths=0.68,
                           showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(colour)
            body.set_alpha(0.48)
            body.set_edgecolor(colour)
            body.set_linewidth(0.8)
        vp["cmedians"].set_color("white")
        vp["cmedians"].set_linewidth(1.5)

        # Annotate median
        med = float(np.median(vals))
        ax.text(pos, med + 0.06, f"{med:.2f}", ha="center", va="bottom",
                fontsize=5.5, color=colour, fontweight="bold")

    # Reference lines
    ax.axhline(0.0, color=C["ref"], lw=0.8, ls="--", zorder=1)
    ax.axhline(0.5, color=C["ref"], lw=0.8, ls=":",  zorder=1)
    ax.text(5.6, 0.51, "0.5", ha="left", va="bottom",
            color=C["ref"], fontsize=5.5)

    # Vertical separator between RF and ANN groups
    ax.axvline(3.0, color=C["ref"], lw=0.5, ls=":", alpha=0.5)

    # Model group header labels (data-x, axes-y transform)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.5, 1.03, "RF",  transform=trans, ha="center",
            fontsize=7, fontweight="bold")
    ax.text(4.5, 1.03, "ANN", transform=trans, ha="center",
            fontsize=7, fontweight="bold")

    # Legend
    ax.legend(
        handles=[
            Patch(facecolor=C["pub_test"], alpha=0.65, label="PUB test"),
            Patch(facecolor=C["pur_test"], alpha=0.65, label="PUR test"),
        ],
        loc="lower right", handlelength=1.0,
    )

    ax.set_xticks([1, 2, 4, 5])
    ax.set_xticklabels(["PUB", "PUR", "PUB", "PUR"])
    ax.set_xlim(0.3, 6.0)
    ax.set_ylim(NSE_CLIP - 0.1, 1.18)
    ax.set_ylabel("Log-space NSE")
    ax.set_title("Test set performance: PUB vs PUR")
    _despine(ax)
    _panel_label(ax, "d")


# ── Panel C: Q100 CDF ────────────────────────────────────────────────────────
def panel_q100_cdf(ax, data: dict, pub_splits_dict: dict,
                   pur_folds: dict) -> None:
    q100   = data["q_true"][:, 5]
    tr_idx, _, te_idx = pub_splits_dict[REP_SEED]
    pur_te = np.unique(np.concatenate(list(pur_folds.values())))

    # Training: subdued gray; PUB: bold blue solid; PUR: bold red dashed
    layers = [
        (q100[tr_idx], f"Training  (n = {len(tr_idx):,})",
         "#AAAAAA",    "-",        1.4),
        (q100[te_idx], f"PUB test  (n = {len(te_idx):,})",
         C["pub_test"], "-",       2.2),
        (q100[pur_te], f"PUR test  (n = {len(pur_te):,})",
         C["pur_test"], (0, (5, 2)), 2.2),
    ]
    cdf_store = {}
    for vals, label, colour, ls, lw in layers:
        v   = np.sort(vals[vals > 0])
        cdf = np.arange(1, len(v) + 1) / len(v)
        ax.semilogx(v, cdf, color=colour, lw=lw, ls=ls,
                    label=label, zorder=3, solid_capstyle="round")
        cdf_store[label.split()[0]] = (v, cdf)
        # Median tick mark on each curve
        med = float(np.median(v))
        med_cdf = float(np.interp(med, v, cdf))
        ax.plot(med, med_cdf, marker="|", color=colour,
                ms=7, mew=1.5, zorder=5)

    # Shaded gap between PUB and PUR – use PUR color tinted very lightly
    if "PUB" in cdf_store and "PUR" in cdf_store:
        v_pub, c_pub = cdf_store["PUB"]
        v_pur, c_pur = cdf_store["PUR"]
        v_min = max(v_pub.min(), v_pur.min())
        v_max = min(v_pub.max(), v_pur.max())
        v_com = np.logspace(np.log10(v_min), np.log10(v_max), 500)
        c_pub_i = np.interp(v_com, v_pub, c_pub)
        c_pur_i = np.interp(v_com, v_pur, c_pur)
        ax.fill_between(v_com, c_pub_i, c_pur_i,
                        color=C["pur_test"], alpha=0.12, zorder=2)

    ax.set_xlabel(r"$Q_{100}$ (mm d$^{-1}$)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.grid(False)
    ax.legend(loc="lower right", handlelength=2.0, labelspacing=0.30)
    _despine(ax)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SUPPLEMENTARY: PUR PCA SMALL MULTIPLES  (publication-quality)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_pur_grid(data: dict, pub_splits_dict: dict,
                     pur_folds: dict, out_path: Path) -> None:
    """
    4×4 grid (16 panels): first panel = PUB, then 15 PUR folds in sorted order.
    No overall suptitle; each subplot has only the region short name as title.
    """
    _setup_pub_style()

    X         = _feature_matrix(data)
    fold_list = sorted(pur_folds.keys())
    short_names, palette = _build_fold_style(fold_list)
    n_folds   = len(fold_list)          # 15
    n_total   = 1 + n_folds            # PUB + 15 PUR = 16
    n_cols    = 4
    n_rows    = int(np.ceil(n_total / n_cols))   # 4
    all_idx   = np.arange(len(data["stations"]))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.4 * n_rows))
    fig.subplots_adjust(hspace=0.40, wspace=0.32)

    axes_flat = [axes[r, c] for r in range(n_rows) for c in range(n_cols)]

    # ── Panel 0: PUB ─────────────────────────────────────────────────────────
    ax = axes_flat[0]
    tr_idx, _, te_idx = pub_splits_dict[REP_SEED]
    coords_pub, evr_pub = compute_pca(X[tr_idx], X, n_components=2)
    ax.scatter(coords_pub[tr_idx, 0], coords_pub[tr_idx, 1],
               s=2.5, c=C["train"], alpha=0.22, linewidths=0, zorder=1, rasterized=True)
    _confidence_ellipse(ax, coords_pub[tr_idx, 0], coords_pub[tr_idx, 1],
                        n_std=2.0, facecolor="none",
                        edgecolor="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)
    ax.scatter(coords_pub[te_idx, 0], coords_pub[te_idx, 1],
               s=14, c=C["pub_test"], alpha=0.82, linewidths=0, zorder=3,
               label=f"Test  n = {len(te_idx)}")
    _confidence_ellipse(ax, coords_pub[te_idx, 0], coords_pub[te_idx, 1],
                        n_std=2.0, facecolor="none",
                        edgecolor=C["pub_test"], linewidth=1.0, linestyle="-", zorder=4)
    ax.set_title("PUB", fontsize=8, pad=3)
    ax.set_xlabel(f"PC1 ({evr_pub[0]*100:.0f}%)", fontsize=7.5)
    ax.set_ylabel(f"PC2 ({evr_pub[1]*100:.0f}%)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    leg = ax.legend(fontsize=6.5, loc="lower right",
                    handletextpad=0.3, borderaxespad=0.4, markerscale=1.3)
    leg.get_frame().set_linewidth(0.5)
    _despine(ax); _light_grid(ax, "both")

    # ── Panels 1–15: PUR folds ────────────────────────────────────────────────
    for i, fold_label in enumerate(fold_list):
        ax = axes_flat[i + 1]

        te_idx_f = pur_folds[fold_label]
        te_set   = set(te_idx_f.tolist())
        tr_pool  = np.array([j for j in all_idx if j not in te_set])

        coords, evr = compute_pca(X[tr_pool], X, n_components=2)
        xtr, ytr = coords[tr_pool,    0], coords[tr_pool,    1]
        xte, yte = coords[te_idx_f,   0], coords[te_idx_f,   1]

        ax.scatter(xtr, ytr, s=2.5, c=C["train"], alpha=0.22,
                   linewidths=0, zorder=1, rasterized=True)
        _confidence_ellipse(ax, xtr, ytr, n_std=2.0, facecolor="none",
                            edgecolor="#AAAAAA", linewidth=0.8, linestyle="--", zorder=2)
        ax.scatter(xte, yte, s=14, c=palette[fold_label], alpha=0.82,
                   linewidths=0, zorder=3,
                   label=f"Test  n = {len(te_idx_f)}")
        _confidence_ellipse(ax, xte, yte, n_std=2.0, facecolor="none",
                            edgecolor=palette[fold_label], linewidth=1.0,
                            linestyle="-", zorder=4)

        ax.set_title(short_names.get(fold_label, fold_label), fontsize=8, pad=3)
        ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)", fontsize=7.5)
        ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        leg = ax.legend(fontsize=6.5, loc="lower right",
                        handletextpad=0.3, borderaxespad=0.4, markerscale=1.3)
        leg.get_frame().set_linewidth(0.5)
        _despine(ax); _light_grid(ax, "both")

    # Hide any surplus axes (none expected for 4×4=16, but guard just in case)
    for j in range(n_total, n_rows * n_cols):
        axes_flat[j].set_visible(False)

    stem = out_path.with_suffix("")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOG.info(f"Saved PUR PCA grid: {stem}.[svg/pdf/png]")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN COMPOSITE FIGURE  (5 panels, publication layout)
# ─────────────────────────────────────────────────────────────────────────────

def build_main_figure(data: dict, pub_splits_dict: dict, pur_folds: dict,
                      dist_df: pd.DataFrame,
                      xgb_fold_nse: dict[str, float],
                      ann_fold_nse: dict[str, float],
                      out_path: Path) -> None:
    _setup_pub_style()

    fold_styles = _build_fold_style(sorted(pur_folds.keys()))

    # ── 3-panel layout ────────────────────────────────────────────────────────
    # (a) Mahalanobis per-fold horizontal boxplot   [tall – 16 rows]
    # (b) Distance vs model performance – fold scatter [medium]
    # (c) Q100 CDF training / PUB / PUR             [medium]
    fig = plt.figure(figsize=(7.2, 10.8))
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[4.0, 2.9, 2.1],
        hspace=0.46,
        left=0.12, right=0.97, top=0.97, bottom=0.06,
    )
    ax_mahal = fig.add_subplot(gs[0])
    ax_dist  = fig.add_subplot(gs[1])
    ax_q100  = fig.add_subplot(gs[2])

    panel_mahal_folds(ax_mahal, dist_df, pur_folds, fold_styles)
    panel_dist_nse   (ax_dist,  dist_df, xgb_fold_nse, ann_fold_nse, fold_styles)
    panel_q100_cdf   (ax_q100,  data, pub_splits_dict, pur_folds)

    # ── Aligned panel labels in figure coordinates ───────────────────────────
    # Force layout so ax.get_position() returns final values
    fig.canvas.draw()
    label_x = ax_q100.get_position().x0 - 0.09   # anchored to tightest left edge
    for ax, lbl in [(ax_mahal, "a"), (ax_dist, "b"), (ax_q100, "c")]:
        pos = ax.get_position()
        fig.text(label_x, pos.y1 + 0.002, lbl,
                 fontsize=9, fontweight="bold", va="bottom", ha="left",
                 fontfamily="sans-serif")

    stem = out_path.with_suffix("")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOG.info(f"Saved main figure: {stem}.[svg/pdf/png]")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(dist_df: pd.DataFrame, data: dict,
                 pub_splits_dict: dict, pur_folds: dict):
    out_csv = OUT_DATA / "feature_distances.csv"
    dist_df.to_csv(out_csv, index=False)
    LOG.info(f"Saved feature distances: {out_csv}")

    pub_d  = dist_df.loc[dist_df["split_type"] == "PUB", "mahal_dist"].dropna()
    pur_d  = dist_df.loc[dist_df["split_type"] == "PUR", "mahal_dist"].dropna()

    tr_idx, _, te_idx = pub_splits_dict[REP_SEED]
    q100 = data["q_true"][:, 5]

    lines = [
        "Feature-Space Distribution Report",
        "=" * 72,
        "",
        "Strategy definitions:",
        f"  PUB: random 20% test split (split_seed = {REP_SEED})",
        "  PUR: HydroSHEDS Lv-2 regional holdout – one basin excluded per fold",
        "",
        f"Feature set used  : {'X_base (static, 36 cols)' if not USE_FLOW_FEAT else 'X_full (static+flow, 48 cols)'}",
        f"Total stations    : {len(data['stations'])}",
        f"PUB train / test  : {len(tr_idx)} / {len(te_idx)}",
        f"PUR folds         : {len(pur_folds)}",
        "",
        "Mahalanobis distance (regularised, Ledoit-Wolf):",
        f"  PUB test  –  median={pub_d.median():.2f}  mean={pub_d.mean():.2f}  "
        f"std={pub_d.std():.2f}  n={len(pub_d)}",
        f"  PUR test  –  median={pur_d.median():.2f}  mean={pur_d.mean():.2f}  "
        f"std={pur_d.std():.2f}  n={len(pur_d)}",
        "",
        "Q100 distribution (m³/s):",
        f"  Train    –  median={np.median(q100[tr_idx]):.1f}",
        f"  PUB test –  median={np.median(q100[te_idx]):.1f}",
        "",
        "PUR fold breakdown (Mahalanobis median):",
    ]
    for fold_label in sorted(pur_folds.keys()):
        fd = dist_df.loc[dist_df["fold"] == fold_label, "mahal_dist"]
        lines.append(f"  {fold_label:30s}  n={len(fd):4d}  median={fd.median():.2f}")

    report_path = OUT_DATA / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info(f"Saved report: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    LOG.info("=" * 72)
    LOG.info(f"{TAG}  start")
    LOG.info("=" * 72)

    # ── data ──────────────────────────────────────────────────────────────────
    data = load_data()
    pub_splits_dict, pur_folds = reconstruct_splits(data)

    # ── distances ─────────────────────────────────────────────────────────────
    dist_df = build_mahal_table(data, pub_splits_dict, pur_folds)

    # ── model NSE (per-station, +flow, for CSV only) ─────────────────────────
    LOG.info("Loading RF predictions (per-station NSE) …")
    rf_nse  = load_model_nse("RF",         RF_DATA_DIR,  pub_splits_dict, pur_folds)

    LOG.info("Loading ANN-Single predictions (per-station NSE) …")
    ann_nse = load_model_nse("ANN_Single", ANN_DATA_DIR, pub_splits_dict, pur_folds)

    dist_df = attach_nse_to_mahal(dist_df, rf_nse, ann_nse)

    # ── fold-level pool NSE (base, linear Q100, for panel b) ─────────────────
    LOG.info("Loading fold-level pool-NSE (base predictions) …")
    xgb_fold_nse = load_fold_pool_nse("XGBoost",    XGB_DATA_DIR, pur_folds)
    ann_fold_nse = load_fold_pool_nse("ANN_Single", ANN_DATA_DIR, pur_folds)

    # ── save CSV ──────────────────────────────────────────────────────────────
    save_outputs(dist_df, data, pub_splits_dict, pur_folds)

    # ── figures ───────────────────────────────────────────────────────────────
    LOG.info("Generating main composite figure …")
    build_main_figure(
        data, pub_splits_dict, pur_folds, dist_df,
        xgb_fold_nse, ann_fold_nse,
        OUT_FIG / f"{TAG}_main.png",
    )

    LOG.info("Generating PCA grid figure (PUB + PUR) …")
    plot_pca_pur_grid(
        data, pub_splits_dict, pur_folds,
        OUT_FIG / f"{TAG}_pca_pur_grid.png",
    )

    LOG.info("Done.")


if __name__ == "__main__":
    main()
