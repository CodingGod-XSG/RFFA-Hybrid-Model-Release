# -*- coding: utf-8 -*-
"""04_GHM_Benchmark.py
================================================================================
Pipeline stage 07 -- GHM (Global Hydrological Model) benchmark: fits GEV to
simulated streamflow and compares against observed GEV, globally and across
PUR holdout basins. Required input for Figure 6 (`merged_obs_sim_pur.csv`,
`pur_selected_obs_sim_pur.csv`).

Key outputs
-----------
Data:
  data/proceed/Caravan-GRDC/04_GHM_Benchmark/
    merged_obs_sim_pur.csv
    overall_metrics.csv
    overall_metrics_by_variable.csv
    pur_selected_basin_counts.csv
    pur_metrics_by_basin_variable.csv
    pur_metrics_summary.csv
    report.txt

Figures:
  figures/Caravan-GRDC/04_GHM_Benchmark/
    fig_overall_scatter_panels.png
    fig_global_error_maps.png
    fig_pur_heatmap_r2_top_basins.png
    fig_pur_q100_rankings.png
"""
from __future__ import annotations

import importlib
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import DATA_PROCEED, FIGURE_ROOT, stage_dir


GEV_VARS = ["mu", "sigma", "xi", "Q2", "Q5", "Q10", "Q20", "Q50", "Q100"]
MAP_VARS = ["Q10", "Q100", "xi"]
SCATTER_VARS = ["mu", "sigma", "xi", "Q10", "Q50", "Q100"]
FLOW_SCALE_VARS = ["mu", "sigma", "Q2", "Q5", "Q10", "Q20", "Q50", "Q100"]

OBS_GEV_CSV = DATA_PROCEED / "01_GEV-Fit" / "gev_station_params.csv"
SIM_GEV_CSV = DATA_PROCEED / "04_Sim_GEV-Fit" / "sim_gev_station_params.csv"
BASIN_CSV = DATA_PROCEED / "05_PUR_Basin_Select" / "station_basin_assignment.csv"

OUT_DATA = stage_dir(DATA_PROCEED, "04_GHM_Benchmark")
OUT_FIG = stage_dir(FIGURE_ROOT, "04_GHM_Benchmark")

MIN_PUR_STATIONS = 50
TOP_BASIN_N = 24
APPLY_SCALE_ADJUST = True
SCALE_REF_VAR = "Q100"
LAYER2_TOP_BASINS = 12
LAYER4_TOP_BASINS = 12

T_COLORS = {
    2: "#3b82f6",
    5: "#10b981",
    10: "#22c55e",
    20: "#f59e0b",
    50: "#f97316",
    100: "#ef4444",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DATA / "log.txt", mode="w", encoding="utf-8"),
    ],
)
LOG = logging.getLogger(__name__)


def _setup_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 260,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#d0d7de",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _assert_inputs_exist(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        txt = "\n".join(["Missing required input file(s):", *[f"  - {m}" for m in missing]])
        raise FileNotFoundError(txt)


def _safe_station_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _scale_ratio_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in FLOW_SCALE_VARS:
        obs = merged[f"{v}_obs"].to_numpy(float)
        sim = merged[f"{v}_sim"].to_numpy(float)
        mask = np.isfinite(obs) & np.isfinite(sim) & (obs > 0) & (sim > 0)
        if int(mask.sum()) == 0:
            rows.append({
                "variable": v,
                "n": 0,
                "median_sim_over_obs": np.nan,
                "p10_sim_over_obs": np.nan,
                "p90_sim_over_obs": np.nan,
            })
            continue
        r = sim[mask] / obs[mask]
        rows.append(
            {
                "variable": v,
                "n": int(mask.sum()),
                "median_sim_over_obs": float(np.nanmedian(r)),
                "p10_sim_over_obs": float(np.nanpercentile(r, 10)),
                "p90_sim_over_obs": float(np.nanpercentile(r, 90)),
            }
        )
    return pd.DataFrame(rows)


def _estimate_global_scale_factor(merged: pd.DataFrame, ref_var: str) -> float:
    if ref_var not in FLOW_SCALE_VARS:
        raise ValueError(f"scale_ref_var must be in {FLOW_SCALE_VARS}, got {ref_var}")
    obs = merged[f"{ref_var}_obs"].to_numpy(float)
    sim = merged[f"{ref_var}_sim"].to_numpy(float)
    mask = np.isfinite(obs) & np.isfinite(sim) & (obs > 0) & (sim > 0)
    if int(mask.sum()) < 10:
        return np.nan
    ratio = sim[mask] / obs[mask]
    return float(np.nanmedian(ratio))


def _load_and_merge() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _assert_inputs_exist([OBS_GEV_CSV, SIM_GEV_CSV, BASIN_CSV])

    obs = pd.read_csv(OBS_GEV_CSV)
    sim = pd.read_csv(SIM_GEV_CSV)
    basin = pd.read_csv(BASIN_CSV)

    for name, df in [("obs", obs), ("sim", sim), ("basin", basin)]:
        if "station_id" not in df.columns:
            raise KeyError(f"station_id is missing in {name} dataframe")

    obs["station_id"] = _safe_station_id(obs["station_id"])
    sim["station_id"] = _safe_station_id(sim["station_id"])
    basin["station_id"] = _safe_station_id(basin["station_id"])

    keep_obs = obs.copy()
    keep_sim = sim.copy()

    if "fit_ok" in keep_obs.columns:
        keep_obs = keep_obs[keep_obs["fit_ok"] == True].copy()
    if "fit_ok" in keep_sim.columns:
        keep_sim = keep_sim[keep_sim["fit_ok"] == True].copy()

    need_cols = ["station_id"] + GEV_VARS
    for col in need_cols:
        if col not in keep_obs.columns:
            raise KeyError(f"Observed GEV CSV missing column: {col}")
        if col not in keep_sim.columns:
            raise KeyError(f"Simulated GEV CSV missing column: {col}")

    obs_sub = keep_obs[need_cols].copy()
    sim_sub = keep_sim[need_cols].copy()

    merged = obs_sub.merge(sim_sub, on="station_id", how="inner", suffixes=("_obs", "_sim"))
    merged = merged.merge(
        basin[["station_id", "lat", "lon", "continent", "HYBAS_ID", "basin_label"]],
        on="station_id",
        how="left",
    )

    scale_table = _scale_ratio_table(merged)
    scale_table.to_csv(OUT_DATA / "scale_ratio_diagnostics.csv", index=False)

    scale_factor = np.nan
    scale_applied = False
    if APPLY_SCALE_ADJUST:
        scale_factor = _estimate_global_scale_factor(merged, SCALE_REF_VAR)
        if np.isfinite(scale_factor) and scale_factor > 0:
            for v in FLOW_SCALE_VARS:
                merged[f"{v}_sim_raw"] = merged[f"{v}_sim"]
                merged[f"{v}_sim"] = merged[f"{v}_sim"] / scale_factor
            scale_applied = True
            LOG.info(
                "Applied global scale adjustment to simulated flow-like GEV vars: factor=%.4f (ref=%s)",
                scale_factor,
                SCALE_REF_VAR,
            )
        else:
            LOG.warning("Scale adjustment requested but factor could not be estimated; fallback to raw comparison")

    for v in GEV_VARS:
        merged[f"err_{v}"] = merged[f"{v}_sim"] - merged[f"{v}_obs"]
        obs_abs = merged[f"{v}_obs"].abs().replace(0.0, np.nan)
        merged[f"relerr_{v}"] = 100.0 * merged[f"err_{v}"] / obs_abs

    merged = merged.dropna(subset=[f"{v}_obs" for v in GEV_VARS] + [f"{v}_sim" for v in GEV_VARS]).copy()

    basin_counts = (
        merged.dropna(subset=["basin_label"])
        .groupby("basin_label")["station_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("n_stations")
        .reset_index()
    )
    selected = basin_counts[basin_counts["n_stations"] >= MIN_PUR_STATIONS].copy()
    selected_labels = set(selected["basin_label"].tolist())

    pur_selected = merged[merged["basin_label"].isin(selected_labels)].copy()

    LOG.info("Observed fit_ok stations: %d", len(keep_obs))
    LOG.info("Simulated fit_ok stations: %d", len(keep_sim))
    LOG.info("Merged stations (obs ∩ sim): %d", len(merged))
    LOG.info("PUR-selected basins (n >= %d): %d", MIN_PUR_STATIONS, len(selected))
    LOG.info("Stations in PUR-selected basins: %d", len(pur_selected))

    selected.to_csv(OUT_DATA / "pur_selected_basin_counts.csv", index=False)
    merged.to_csv(OUT_DATA / "merged_obs_sim_pur.csv", index=False)
    pur_selected.to_csv(OUT_DATA / "pur_selected_obs_sim_pur.csv", index=False)

    scale_info = {
        "apply_scale_adjust": APPLY_SCALE_ADJUST,
        "scale_applied": scale_applied,
        "scale_ref_var": SCALE_REF_VAR,
        "scale_factor": float(scale_factor) if np.isfinite(scale_factor) else np.nan,
    }

    return merged, pur_selected, scale_info


def _calc_metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    n = int(len(obs))
    if n < 3:
        return {
            "n": n,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "RelBiasPct": np.nan,
            "Slope": np.nan,
            "Intercept": np.nan,
            "PearsonR": np.nan,
            "SpearmanR": np.nan,
        }

    err = sim - obs
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    obs_mean = float(np.mean(obs))
    rel_bias = float(100.0 * bias / obs_mean) if obs_mean != 0 else np.nan

    ss_res = float(np.sum((obs - sim) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    slope, intercept = np.polyfit(obs, sim, deg=1)
    pearson = float(np.corrcoef(obs, sim)[0, 1])

    rank_obs = pd.Series(obs).rank(method="average").to_numpy(dtype=float)
    rank_sim = pd.Series(sim).rank(method="average").to_numpy(dtype=float)
    spearman = float(np.corrcoef(rank_obs, rank_sim)[0, 1])

    return {
        "n": n,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "RelBiasPct": rel_bias,
        "Slope": float(slope),
        "Intercept": float(intercept),
        "PearsonR": pearson,
        "SpearmanR": spearman,
    }


def _overall_metrics_table(df: pd.DataFrame, scope_name: str) -> pd.DataFrame:
    rows = []
    for v in GEV_VARS:
        m = _calc_metrics(df[f"{v}_obs"].to_numpy(float), df[f"{v}_sim"].to_numpy(float))
        rows.append({"scope": scope_name, "variable": v, **m})
    return pd.DataFrame(rows)


def _metrics_by_basin(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(columns=["basin_label", "continent", "variable", "n", "R2", "RMSE", "MAE", "Bias", "RelBiasPct", "Slope", "Intercept", "PearsonR", "SpearmanR"])

    grouped = df.groupby(["basin_label", "continent"], dropna=False)
    for (label, cont), g in grouped:
        for v in GEV_VARS:
            m = _calc_metrics(g[f"{v}_obs"].to_numpy(float), g[f"{v}_sim"].to_numpy(float))
            rows.append(
                {
                    "basin_label": label,
                    "continent": cont,
                    "variable": v,
                    **m,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["variable", "R2"], ascending=[True, False]).reset_index(drop=True)


def _collect_vectors_by_return_period(df: pd.DataFrame, return_period: int) -> tuple[np.ndarray, np.ndarray]:
    obs_col = f"Q{return_period}_obs"
    sim_col = f"Q{return_period}_sim"
    if obs_col not in df.columns or sim_col not in df.columns:
        return np.array([], dtype=float), np.array([], dtype=float)
    obs = df[obs_col].to_numpy(float)
    sim = df[sim_col].to_numpy(float)
    mask = np.isfinite(obs) & np.isfinite(sim) & (obs > 0) & (sim > 0)
    return obs[mask], sim[mask]


def _collect_vectors_all_return_periods(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_all = []
    sim_all = []
    t_all = []
    for t in [2, 5, 10, 20, 50, 100]:
        x, y = _collect_vectors_by_return_period(df, t)
        if len(x) == 0:
            continue
        obs_all.append(x)
        sim_all.append(y)
        t_all.append(np.full(len(x), t, dtype=int))
    if not obs_all:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=int)
    return np.concatenate(obs_all), np.concatenate(sim_all), np.concatenate(t_all)


def _build_layer_metrics_tables(pur_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Layer 1: all PUR regions, separated by return period.
    l1_rows = []
    for t in [2, 5, 10, 20, 50, 100]:
        x, y = _collect_vectors_by_return_period(pur_df, t)
        m = _calc_metrics(x, y)
        l1_rows.append({"return_period": t, **m})
    layer1 = pd.DataFrame(l1_rows)

    # Layer 2: each region, pooled over all return periods.
    l2_rows = []
    for label, g in pur_df.groupby("basin_label"):
        x, y, _ = _collect_vectors_all_return_periods(g)
        m = _calc_metrics(x, y)
        continent = str(g["continent"].iloc[0]) if "continent" in g.columns else "NA"
        l2_rows.append({"basin_label": label, "continent": continent, **m})
    layer2 = pd.DataFrame(l2_rows).sort_values(["R2", "n"], ascending=[False, False]).reset_index(drop=True)

    # Layer 3: all regions pooled with all return periods.
    x_all, y_all, t_all = _collect_vectors_all_return_periods(pur_df)
    m_all = _calc_metrics(x_all, y_all)
    layer3 = pd.DataFrame([
        {
            "scope": "all_regions_all_return_periods",
            "n_points": int(len(x_all)),
            "n_return_periods": int(len(np.unique(t_all))) if len(t_all) > 0 else 0,
            **m_all,
        }
    ])

    # Layer 4: each region at Q100.
    l4_rows = []
    for label, g in pur_df.groupby("basin_label"):
        x, y = _collect_vectors_by_return_period(g, 100)
        m = _calc_metrics(x, y)
        continent = str(g["continent"].iloc[0]) if "continent" in g.columns else "NA"
        l4_rows.append({"basin_label": label, "continent": continent, "return_period": 100, **m})
    layer4 = pd.DataFrame(l4_rows).sort_values(["R2", "n"], ascending=[False, False]).reset_index(drop=True)

    return layer1, layer2, layer3, layer4


def _plot_layer1_return_period_all_pur(pur_df: pd.DataFrame, layer1_df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.8), constrained_layout=True)
    axes = axes.ravel()
    for ax, t in zip(axes, [2, 5, 10, 20, 50, 100]):
        x, y = _collect_vectors_by_return_period(pur_df, t)
        if len(x) == 0:
            ax.set_title(f"Q{t}: no data")
            ax.axis("off")
            continue
        ax.scatter(x, y, s=8, alpha=0.35, color=T_COLORS[t], rasterized=True)
        lo, hi = _nice_limits(x, y, positive_only=True)
        lo = max(lo, 1e-6)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Observed Q{t}")
        ax.set_ylabel(f"Simulated Q{t}")
        row = layer1_df[layer1_df["return_period"] == t]
        if not row.empty:
            r = row.iloc[0]
            ax.set_title(f"Q{t} | R2={r['R2']:.3f}, RMSE={r['RMSE']:.2f}, PBIAS={r['RelBiasPct']:.1f}%")
    fig.suptitle("Layer 1: Different Return Periods Across All PUR Regions", fontsize=15)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_layer2_regions_all_return_periods(pur_df: pd.DataFrame, layer2_df: pd.DataFrame, out_png: Path) -> None:
    top_labels = layer2_df.head(max(1, LAYER2_TOP_BASINS))["basin_label"].tolist()
    if not top_labels:
        return
    n_pan = len(top_labels)
    ncols = 4
    nrows = int(np.ceil(n_pan / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.8), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    legend_handles = []
    for t in [2, 5, 10, 20, 50, 100]:
        legend_handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=T_COLORS[t], label=f"Q{t}", markersize=5))

    for i, label in enumerate(top_labels):
        ax = axes[i]
        g = pur_df[pur_df["basin_label"] == label]
        xx, yy, tt = _collect_vectors_all_return_periods(g)
        if len(xx) == 0:
            ax.axis("off")
            continue
        for t in [2, 5, 10, 20, 50, 100]:
            m = tt == t
            if m.sum() == 0:
                continue
            ax.scatter(xx[m], yy[m], s=8, alpha=0.35, color=T_COLORS[t], rasterized=True)
        lo, hi = _nice_limits(xx, yy, positive_only=True)
        lo = max(lo, 1e-6)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        r = layer2_df[layer2_df["basin_label"] == label]
        if not r.empty:
            rr = r.iloc[0]
            ax.set_title(f"{label}\nR2={rr['R2']:.2f}, n={int(rr['n'])}", fontsize=8.5)
        else:
            ax.set_title(label, fontsize=8.5)
        ax.set_xlabel("Observed (all RP)", fontsize=8)
        ax.set_ylabel("Simulated (all RP)", fontsize=8)

    for j in range(n_pan, len(axes)):
        axes[j].axis("off")

    fig.legend(handles=legend_handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Layer 2: Different PUR Regions, Each Region Across All Return Periods", fontsize=14)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_layer3_all_regions_all_return_periods(pur_df: pd.DataFrame, layer3_df: pd.DataFrame, out_png: Path) -> None:
    x, y, t = _collect_vectors_all_return_periods(pur_df)
    if len(x) == 0:
        return
    fig, ax = plt.subplots(figsize=(10.8, 8.6), constrained_layout=True)

    for rp in [2, 5, 10, 20, 50, 100]:
        m = t == rp
        if m.sum() == 0:
            continue
        ax.scatter(x[m], y[m], s=7, alpha=0.25, color=T_COLORS[rp], rasterized=True, label=f"Q{rp}")

    lo, hi = _nice_limits(x, y, positive_only=True)
    lo = max(lo, 1e-6)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed (all regions, all RP)")
    ax.set_ylabel("Simulated (all regions, all RP)")

    if not layer3_df.empty:
        r = layer3_df.iloc[0]
        txt = f"n={int(r['n'])}\nR2={r['R2']:.3f}\nRMSE={r['RMSE']:.3f}\nBias={r['Bias']:.3f}\nr={r['PearsonR']:.3f}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#7f8c8d", alpha=0.9), fontsize=9)
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("Layer 3: All PUR Regions and All Return Periods (Pooled)")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_layer4_q100_region_scatter(pur_df: pd.DataFrame, layer4_df: pd.DataFrame, out_png: Path) -> None:
    top_labels = layer4_df.head(max(1, LAYER4_TOP_BASINS))["basin_label"].tolist()
    if not top_labels:
        return
    n_pan = len(top_labels)
    ncols = 4
    nrows = int(np.ceil(n_pan / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.1, nrows * 3.8), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i, label in enumerate(top_labels):
        ax = axes[i]
        g = pur_df[pur_df["basin_label"] == label]
        x, y = _collect_vectors_by_return_period(g, 100)
        if len(x) == 0:
            ax.axis("off")
            continue
        ax.scatter(x, y, s=10, alpha=0.45, color="#ef4444", rasterized=True)
        lo, hi = _nice_limits(x, y, positive_only=True)
        lo = max(lo, 1e-6)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        r = layer4_df[layer4_df["basin_label"] == label]
        if not r.empty:
            rr = r.iloc[0]
            ax.set_title(f"{label}\nQ100 R2={rr['R2']:.2f}, n={int(rr['n'])}", fontsize=8.5)
        else:
            ax.set_title(label, fontsize=8.5)
        ax.set_xlabel("Observed Q100", fontsize=8)
        ax.set_ylabel("Simulated Q100", fontsize=8)

    for j in range(n_pan, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Layer 4: Q100 Comparison for Each PUR Region", fontsize=14)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _nice_limits(x: np.ndarray, y: np.ndarray, positive_only: bool) -> tuple[float, float]:
    v = np.concatenate([x, y])
    v = v[np.isfinite(v)]
    if positive_only:
        v = v[v > 0]
    if len(v) == 0:
        return (0.0, 1.0)
    lo = float(np.nanpercentile(v, 1))
    hi = float(np.nanpercentile(v, 99))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(v))
        hi = float(np.nanmax(v))
        if lo == hi:
            lo, hi = lo * 0.8, hi * 1.2 if hi != 0 else (0.0, 1.0)
    return lo, hi


def _plot_scatter_panels(df: pd.DataFrame, metrics_df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6), constrained_layout=True)
    axes = axes.ravel()

    panel_colors = {
        "obs": "#1f3b5c",
        "sim": "#b83b5e",
        "fit": "#2a9d8f",
    }

    for ax, var in zip(axes, SCATTER_VARS):
        x = df[f"{var}_obs"].to_numpy(float)
        y = df[f"{var}_sim"].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        is_pos = var != "xi"
        if is_pos:
            valid2 = (x > 0) & (y > 0)
            x = x[valid2]
            y = y[valid2]

        hb = ax.hexbin(
            x,
            y,
            gridsize=44,
            mincnt=1,
            cmap="YlGnBu",
            linewidths=0.0,
            alpha=0.95,
        )

        lo, hi = _nice_limits(x, y, positive_only=is_pos)
        ax.plot([lo, hi], [lo, hi], color="#2d3142", lw=1.4, linestyle="--", label="1:1")

        if len(x) >= 3:
            slope, intercept = np.polyfit(x, y, deg=1)
            xx = np.array([lo, hi])
            yy = slope * xx + intercept
            ax.plot(xx, yy, color=panel_colors["fit"], lw=1.6, label="fit")

        if is_pos:
            ax.set_xscale("log")
            ax.set_yscale("log")
            lo = max(lo, 1e-6)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Observed {var}")
        ax.set_ylabel(f"Simulated {var}")
        ax.set_title(f"{var}: Simulated vs Observed")

        m = metrics_df.loc[metrics_df["variable"] == var]
        if not m.empty:
            mm = m.iloc[0]
            txt = (
                f"n={int(mm['n'])}\n"
                f"R2={mm['R2']:.3f}\n"
                f"RMSE={mm['RMSE']:.3f}\n"
                f"Bias={mm['Bias']:.3f}\n"
                f"r={mm['PearsonR']:.3f}"
            )
            ax.text(
                0.03,
                0.97,
                txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.8,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="#8d99ae", alpha=0.9),
            )

        cbar = fig.colorbar(hb, ax=ax, shrink=0.88, pad=0.01)
        cbar.ax.set_ylabel("Count", rotation=90)

    fig.suptitle("Global Simulated-vs-Observed GEV Comparison", fontsize=15, y=1.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_world_background(ax: plt.Axes) -> None:
    # Optional land background using geopandas if available.
    try:
        gpd = importlib.import_module("geopandas")
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        world.plot(ax=ax, color="#f2f4f7", edgecolor="#d8dee9", linewidth=0.4, zorder=0)
    except Exception:
        pass


def _plot_global_maps(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    cmap = "RdYlBu_r"

    for ax, v in zip(axes, MAP_VARS):
        col = f"relerr_{v}" if v != "xi" else "err_xi"
        vv = df[col].to_numpy(float)
        lo = float(np.nanpercentile(vv, 5))
        hi = float(np.nanpercentile(vv, 95))
        lim = max(abs(lo), abs(hi))
        if not math.isfinite(lim) or lim == 0:
            lim = 1.0

        _draw_world_background(ax)
        sc = ax.scatter(
            df["lon"],
            df["lat"],
            c=vv,
            s=14,
            cmap=cmap,
            vmin=-lim,
            vmax=lim,
            edgecolors="none",
            alpha=0.88,
            zorder=2,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        title = f"{v} relative error (%)" if v != "xi" else "xi absolute error"
        ax.set_title(title)

        cb = fig.colorbar(sc, ax=ax, shrink=0.84, pad=0.02)
        cb.set_label("Error")

    fig.suptitle("Global Spatial Error Patterns (Simulated minus Observed)", fontsize=14)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_pur_heatmap(metrics_basin: pd.DataFrame, out_png: Path) -> None:
    q100 = metrics_basin[metrics_basin["variable"] == "Q100"].copy()
    q100 = q100.sort_values(["R2", "n"], ascending=[False, False]).head(TOP_BASIN_N)
    top_labels = q100["basin_label"].tolist()

    sub = metrics_basin[metrics_basin["basin_label"].isin(top_labels)].copy()
    pivot = (
        sub.pivot_table(index="basin_label", columns="variable", values="R2", aggfunc="mean")
        .reindex(top_labels)
        .reindex(columns=GEV_VARS)
    )

    fig, ax = plt.subplots(figsize=(12.8, max(7.4, 0.32 * len(pivot))), constrained_layout=True)
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap="viridis", vmin=-0.2, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"PUR Basin-level R2 Heatmap (Top {len(pivot)} basins by Q100 R2)")
    ax.set_xlabel("GEV variable")
    ax.set_ylabel("Basin label")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white")

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015)
    cbar.set_label("R2")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_pur_q100_rankings(metrics_basin: pd.DataFrame, out_png: Path) -> None:
    q100 = metrics_basin[metrics_basin["variable"] == "Q100"].copy()
    if q100.empty:
        return

    q100 = q100[np.isfinite(q100["R2"])].copy()
    if q100.empty:
        return

    # Keep rankings numerically stable for plotting while preserving original metrics in tables.
    q100["R2_plot"] = q100["R2"].clip(lower=-1.0, upper=1.0)
    q100 = q100.sort_values(["R2_plot", "n"], ascending=[False, False]).head(TOP_BASIN_N)
    q100 = q100.iloc[::-1]

    cont_colors = {
        "AF": "#2a9d8f",
        "AU": "#e9c46a",
        "EU": "#457b9d",
        "NA": "#e76f51",
        "SA": "#8d5a97",
    }
    colors = [cont_colors.get(str(c), "#808b96") for c in q100["continent"]]

    fig, ax = plt.subplots(figsize=(11.8, max(7.2, 0.31 * len(q100))), constrained_layout=True)
    bars = ax.barh(q100["basin_label"], q100["R2_plot"], color=colors, alpha=0.95)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("R2 for Q100")
    ax.set_ylabel("Basin label")
    ax.set_title(f"PUR Basin Ranking by Q100 Agreement (Top {len(q100)})")

    for b, (_, row) in zip(bars, q100.iterrows()):
        txt_x = float(np.clip(b.get_width() + 0.03, -0.95, 0.95))
        ax.text(
            txt_x,
            b.get_y() + b.get_height() / 2,
            f"n={int(row['n'])}  RMSE={row['RMSE']:.2f}  R2={row['R2']:.2f}",
            va="center",
            ha="left" if b.get_width() < 0.9 else "right",
            fontsize=8,
        )

    handles = []
    seen = set()
    for cont in q100["continent"].astype(str).tolist():
        if cont in seen:
            continue
        seen.add(cont)
        handles.append(plt.Line2D([0], [0], color=cont_colors.get(cont, "#808b96"), lw=6, label=cont))
    if handles:
        ax.legend(handles=handles, title="Continent", loc="lower right")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_report(merged: pd.DataFrame, pur_df: pd.DataFrame, overall: pd.DataFrame, pur_summary: pd.DataFrame, scale_info: dict) -> None:
    lines = [
        "SimGEV vs ObGEV Comparison Report",
        "=" * 88,
        f"Observed GEV CSV          : {OBS_GEV_CSV}",
        f"Simulated GEV CSV         : {SIM_GEV_CSV}",
        f"Basin assignment CSV      : {BASIN_CSV}",
        f"Minimum PUR stations      : {MIN_PUR_STATIONS}",
        f"Scale adjustment requested: {scale_info.get('apply_scale_adjust')}",
        f"Scale adjustment applied  : {scale_info.get('scale_applied')}",
        f"Scale reference variable  : {scale_info.get('scale_ref_var')}",
        f"Global scale factor(sim/obs): {scale_info.get('scale_factor')}",
        "",
        f"Merged stations (obs ∩ sim)          : {len(merged)}",
        f"Stations in selected PUR basins      : {len(pur_df)}",
        f"Unique selected PUR basins           : {pur_df['basin_label'].nunique() if len(pur_df) > 0 else 0}",
        "",
        "Overall metrics highlights (PUR-selected subset):",
    ]

    for var in ["mu", "sigma", "xi", "Q10", "Q50", "Q100"]:
        row = overall[(overall["scope"] == "PUR-selected") & (overall["variable"] == var)]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"  - {var:<5} | n={int(r['n']):>4d} | R2={r['R2']:.3f} | RMSE={r['RMSE']:.3f} | Bias={r['Bias']:.3f} | r={r['PearsonR']:.3f}"
        )

    lines.extend(["", "Top basins by Q100 R2 (PUR-selected):"])
    q100 = pur_summary[pur_summary["variable"] == "Q100"].sort_values("R2", ascending=False).head(15)
    for _, r in q100.iterrows():
        lines.append(
            f"  - {r['basin_label']} ({r['continent']}): n={int(r['n'])}, R2={r['R2']:.3f}, RMSE={r['RMSE']:.3f}, Bias={r['Bias']:.3f}"
        )

    report_path = OUT_DATA / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Saved report: %s", report_path)


def main() -> None:
    LOG.info("=" * 88)
    LOG.info("04_GHM_Benchmark start")
    LOG.info("=" * 88)

    _setup_plot_style()

    merged, pur_selected, scale_info = _load_and_merge()

    overall_all = _overall_metrics_table(merged, "All-merged")
    overall_pur = _overall_metrics_table(pur_selected, "PUR-selected")
    overall = pd.concat([overall_all, overall_pur], ignore_index=True)

    pur_metrics = _metrics_by_basin(pur_selected)

    overall.to_csv(OUT_DATA / "overall_metrics_by_variable.csv", index=False)

    overall_summary = (
        overall.groupby("scope", as_index=False)
        .agg(
            n_variables=("variable", "count"),
            mean_R2=("R2", "mean"),
            mean_PearsonR=("PearsonR", "mean"),
            mean_abs_RelBiasPct=("RelBiasPct", lambda x: float(np.nanmean(np.abs(x)))),
        )
    )
    overall_summary.to_csv(OUT_DATA / "overall_metrics.csv", index=False)

    pur_metrics.to_csv(OUT_DATA / "pur_metrics_by_basin_variable.csv", index=False)

    pur_summary = (
        pur_metrics.groupby(["variable", "basin_label", "continent"], as_index=False)
        .agg(
            n=("n", "max"),
            R2=("R2", "mean"),
            RMSE=("RMSE", "mean"),
            MAE=("MAE", "mean"),
            Bias=("Bias", "mean"),
            PearsonR=("PearsonR", "mean"),
        )
    )
    pur_summary.to_csv(OUT_DATA / "pur_metrics_summary.csv", index=False)

    # Hierarchical diagnostics: overall -> region -> pooled -> regional Q100.
    layer1_df, layer2_df, layer3_df, layer4_df = _build_layer_metrics_tables(pur_selected)
    layer1_df.to_csv(OUT_DATA / "layer1_metrics_by_return_period_all_pur.csv", index=False)
    layer2_df.to_csv(OUT_DATA / "layer2_metrics_by_region_all_return_periods.csv", index=False)
    layer3_df.to_csv(OUT_DATA / "layer3_overall_pooled_metrics_all_regions_return_periods.csv", index=False)
    layer4_df.to_csv(OUT_DATA / "layer4_q100_metrics_by_region.csv", index=False)

    pur_metrics_for_plot = pur_metrics[pur_metrics["n"] >= 8].copy()

    _plot_scatter_panels(
        pur_selected if len(pur_selected) >= 20 else merged,
        overall_pur if len(pur_selected) >= 20 else overall_all,
        OUT_FIG / "fig_overall_scatter_panels.png",
    )
    _plot_global_maps(
        pur_selected if len(pur_selected) >= 20 else merged,
        OUT_FIG / "fig_global_error_maps.png",
    )

    if len(pur_metrics_for_plot) > 0:
        _plot_pur_heatmap(pur_metrics_for_plot, OUT_FIG / "fig_pur_heatmap_r2_top_basins.png")
        _plot_pur_q100_rankings(pur_metrics_for_plot, OUT_FIG / "fig_pur_q100_rankings.png")

    if len(pur_selected) > 0:
        _plot_layer1_return_period_all_pur(pur_selected, layer1_df, OUT_FIG / "fig_layer1_return_period_scatter_all_pur.png")
        _plot_layer2_regions_all_return_periods(pur_selected, layer2_df, OUT_FIG / "fig_layer2_region_scatter_all_return_periods.png")
        _plot_layer3_all_regions_all_return_periods(pur_selected, layer3_df, OUT_FIG / "fig_layer3_pooled_scatter_all_regions_all_return_periods.png")
        _plot_layer4_q100_region_scatter(pur_selected, layer4_df, OUT_FIG / "fig_layer4_q100_scatter_by_region.png")

    _write_report(merged, pur_selected, overall, pur_summary, scale_info)

    LOG.info("Saved data directory: %s", OUT_DATA)
    LOG.info("Saved figure directory: %s", OUT_FIG)
    LOG.info("Done")


if __name__ == "__main__":
    main()
