# -*- coding: utf-8 -*-
"""FigureS1_Monotonicity_Explanation.py

Reproduces Figure S1 -- explains the mechanism behind flood-quantile
monotonicity violations: log-increment distributions across model
configurations (independent per-return-period networks vs shared/jointly-trained
vs GEV-structurally-constrained), showing the progression from unconstrained to
constrained predictions.

The script reuses station-level predictions produced by
Figure4_GEV_Constrained_Performance.py:

    data/proceed/Caravan-GRDC/Figure4_GEV_Constrained_Performance/
        constrained_station_violations.csv
        model_pub_pur_nse_heatmap_matrix.csv

Outputs:
    data/proceed/Caravan-GRDC/FigureS1_Monotonicity_Explanation/
        monotonicity_model_summary.csv
        monotonicity_pair_summary.csv
        monotonicity_interpretation.md

    figures/Caravan-GRDC/FigureS1_Monotonicity_Explanation/
        FigureS1_Monotonicity_Explanation.png
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.lines
import matplotlib.patches
import matplotlib.ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import DATA_PROCEED, FIGURE_ROOT, stage_dir


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TAG = "FigureS1_Monotonicity_Explanation"

SRC_TAG = "Figure4_GEV_Constrained_Performance"
SRC_DATA = DATA_PROCEED / SRC_TAG

OUT_DATA = stage_dir(DATA_PROCEED, TAG)
OUT_FIG = stage_dir(FIGURE_ROOT, TAG)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DATA / "log.txt", mode="w", encoding="utf-8"),
    ],
)
LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RETURN_PERIODS = [2, 5, 10, 20, 50, 100]
ADJ_PAIRS = list(zip(RETURN_PERIODS[:-1], RETURN_PERIODS[1:]))
Q_PRED = [f"Q{t}_pred" for t in RETURN_PERIODS]
Q_TRUE = [f"Q{t}_true" for t in RETURN_PERIODS]

MODEL_ORDER = ["RF", "SVM", "XGBoost", "ANN", "ANN-Joint", "GEV-NN", "GEV-NN-MSE", "GEV-NN-NLL"]
DIRECT_MODELS = {"RF", "SVM", "XGBoost", "ANN", "ANN-Joint"}
GEV_CONSTRAINED_MODELS = {"GEV-NN", "GEV-NN-MSE", "GEV-NN-NLL"}

MODEL_COLORS = {
    "RF":          "#7A7A7A",
    "SVM":         "#8E44AD",
    "XGBoost":     "#16A085",
    "ANN":         "#D55E00",
    "ANN-Joint":   "#E69F00",
    "GEV-NN":      "#0072B2",
    "GEV-NN-MSE":  "#009E73",
    "GEV-NN-NLL": "#CC79A7",
}

# Three color scheme variants for the log-increment figure
_COLOR_SCHEMES = {
    "A": {  # Wong colorblind-safe (default)
        "RF":          "#7A7A7A",
        "SVM":         "#8E44AD",
        "XGBoost":     "#16A085",
        "ANN":         "#D55E00",
        "ANN-Joint":   "#E69F00",
        "GEV-NN":      "#0072B2",
        "GEV-NN-MSE":  "#009E73",
        "GEV-NN-NLL": "#CC79A7",
    },
    "B": {  # Deep/rich muted tones
        "RF":          "#5C5C5C",
        "SVM":         "#6C3483",
        "XGBoost":     "#0E6655",
        "ANN":         "#B03A2E",
        "ANN-Joint":   "#B7770D",
        "GEV-NN":      "#1A5276",
        "GEV-NN-MSE":  "#1E8449",
        "GEV-NN-NLL": "#76448A",
    },
    "C": {  # Material Design vivid
        "RF":          "#9E9E9E",
        "SVM":         "#9C27B0",
        "XGBoost":     "#009688",
        "ANN":         "#F44336",
        "ANN-Joint":   "#FF9800",
        "GEV-NN":      "#2196F3",
        "GEV-NN-MSE":  "#4CAF50",
        "GEV-NN-NLL": "#E91E63",
    },
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = SRC_DATA / "constrained_station_violations.csv"
    heat_path = SRC_DATA / "model_pub_pur_nse_heatmap_matrix.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing {pred_path}. Run Figure4_GEV_Constrained_Performance.py first."
        )
    st = pd.read_csv(pred_path)
    for c in Q_PRED + Q_TRUE:
        if c in st.columns:
            st[c] = pd.to_numeric(st[c], errors="coerce")
    for c in ["model", "experiment", "station_id", "climate_zone"]:
        if c in st.columns:
            st[c] = st[c].astype(str).str.strip()
    if "violated_any" in st.columns:
        st["violated_any"] = st["violated_any"].astype(str).str.lower().isin(["true", "1", "yes"])

    heat = pd.read_csv(heat_path) if heat_path.exists() else pd.DataFrame()
    return st, heat


def _add_monotonicity_metrics(st: pd.DataFrame) -> pd.DataFrame:
    out = st.copy()
    rel_depths = []
    log_depths = []
    n_viol = np.zeros(len(out), dtype=int)

    for t1, t2 in ADJ_PAIRS:
        q1 = out[f"Q{t1}_pred"].to_numpy(float)
        q2 = out[f"Q{t2}_pred"].to_numpy(float)
        valid = np.isfinite(q1) & np.isfinite(q2) & (q1 > 0) & (q2 > 0)

        diff = q2 - q1
        rel_gap = np.full(len(out), np.nan)
        rel_gap[valid] = diff[valid] / np.maximum(q2[valid], 1e-12)
        log_gap = np.full(len(out), np.nan)
        log_gap[valid] = np.log(q2[valid]) - np.log(q1[valid])

        viol = valid & (q1 > q2)
        out[f"gap_Q{t1}_Q{t2}"] = diff
        out[f"rel_gap_Q{t1}_Q{t2}"] = rel_gap
        out[f"log_gap_Q{t1}_Q{t2}"] = log_gap
        out[f"viol_depth_rel_Q{t1}_Q{t2}"] = np.where(viol, -rel_gap, 0.0)
        out[f"viol_depth_log_Q{t1}_Q{t2}"] = np.where(viol, -log_gap, 0.0)
        n_viol += viol.astype(int)
        rel_depths.append(out[f"viol_depth_rel_Q{t1}_Q{t2}"].to_numpy(float))
        log_depths.append(out[f"viol_depth_log_Q{t1}_Q{t2}"].to_numpy(float))

    out["n_crossed_pairs"] = n_viol
    out["max_violation_depth_rel"] = np.nanmax(np.vstack(rel_depths), axis=0)
    out["sum_violation_depth_rel"] = np.nansum(np.vstack(rel_depths), axis=0)
    out["max_violation_depth_log"] = np.nanmax(np.vstack(log_depths), axis=0)
    out["sum_violation_depth_log"] = np.nansum(np.vstack(log_depths), axis=0)
    out["is_gev_constrained"] = out["model"].isin(GEV_CONSTRAINED_MODELS)
    out["model_family"] = np.where(out["is_gev_constrained"], "GEV-constrained", "Direct/independent quantile")
    return out


def _model_summary(df: pd.DataFrame, heat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, exp), g in df.groupby(["model", "experiment"], observed=True):
        row = {
            "model": model,
            "experiment": exp,
            "model_family": "GEV-constrained" if model in GEV_CONSTRAINED_MODELS else "Direct/independent quantile",
            "n_station_predictions": int(len(g)),
            "any_violation_rate": float(g["violated_any"].mean()),
            "mean_crossed_pairs": float(g["n_crossed_pairs"].mean()),
            "mean_max_relative_violation_depth": float(g["max_violation_depth_rel"].mean()),
            "p95_max_relative_violation_depth": float(g["max_violation_depth_rel"].quantile(0.95)),
            "mean_sum_log_violation_depth": float(g["sum_violation_depth_log"].mean()),
        }
        if not heat.empty and model in heat.columns:
            hh = heat[heat["experiment"].astype(str).str.strip() == exp].copy()
            hh["return_period"] = pd.to_numeric(hh["return_period"], errors="coerce")
            row["mean_nse_all_return_periods"] = pd.to_numeric(hh[model], errors="coerce").mean()
            row["mean_nse_q50_q100"] = pd.to_numeric(
                hh[hh["return_period"].isin([50, 100])][model], errors="coerce"
            ).mean()
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    out["experiment"] = pd.Categorical(out["experiment"], categories=["PUB", "PUR"], ordered=True)
    return out.sort_values(["experiment", "model"]).reset_index(drop=True)


def _pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, exp), g in df.groupby(["model", "experiment"], observed=True):
        for t1, t2 in ADJ_PAIRS:
            col = f"viol_Q{t1}_Q{t2}"
            depth = f"viol_depth_rel_Q{t1}_Q{t2}"
            log_gap = f"log_gap_Q{t1}_Q{t2}"
            if col not in g.columns:
                continue
            viol = g[col].astype(str).str.lower().isin(["true", "1", "yes"])
            rows.append(
                {
                    "model": model,
                    "experiment": exp,
                    "adjacent_pair": f"Q{t1}->Q{t2}",
                    "violation_rate": float(viol.mean()),
                    "mean_relative_depth_if_crossed": float(g.loc[viol, depth].mean()) if viol.any() else 0.0,
                    "median_log_increment": float(g[log_gap].median()),
                    "p05_log_increment": float(g[log_gap].quantile(0.05)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    out["experiment"] = pd.Categorical(out["experiment"], categories=["PUB", "PUR"], ordered=True)
    return out.sort_values(["experiment", "model", "adjacent_pair"]).reset_index(drop=True)


def _climate_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "climate_zone" not in df.columns:
        return pd.DataFrame()
    rows = []
    for (model, exp, zone), g in df.groupby(["model", "experiment", "climate_zone"], observed=True):
        rows.append(
            {
                "model": model,
                "experiment": exp,
                "climate_zone": zone,
                "n": int(len(g)),
                "any_violation_rate": float(g["violated_any"].mean()),
                "mean_max_relative_violation_depth": float(g["max_violation_depth_rel"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _worst_example_rows(df: pd.DataFrame) -> pd.DataFrame:
    ann_pur = df[(df["model"] == "ANN") & (df["experiment"] == "PUR")].copy()
    if ann_pur.empty:
        ann_pur = df[df["model"] == "ANN"].copy()
    if ann_pur.empty:
        return pd.DataFrame()

    ann_pur = ann_pur.sort_values(["n_crossed_pairs", "sum_violation_depth_log"], ascending=False)
    station = ann_pur.iloc[0]["station_id"]
    exp = ann_pur.iloc[0]["experiment"]
    sub = df[(df["station_id"] == station) & (df["experiment"] == exp)].copy()
    sub["model"] = pd.Categorical(sub["model"], categories=MODEL_ORDER, ordered=True)
    return sub.sort_values("model")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_FS = {"title": 12, "label": 11, "tick": 9.5, "legend": 9.5, "annot": 8.5, "panel": 14}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": _FS["label"],
            "axes.titlesize": _FS["title"],
            "axes.titleweight": "bold",
            "axes.labelsize": _FS["label"],
            "xtick.labelsize": _FS["tick"],
            "ytick.labelsize": _FS["tick"],
            "legend.fontsize": _FS["legend"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.30,
            "grid.linestyle": "--",
            "grid.linewidth": 0.55,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _letter(ax, s: str, x: float = -0.12, y: float = 1.07) -> None:
    ax.text(x, y, s, transform=ax.transAxes, fontsize=_FS["panel"], fontweight="bold", va="top")


def _plot_schematic(ax) -> None:
    x = np.arange(len(RETURN_PERIODS), dtype=float)
    direct = np.array([1.00, 1.55, 1.40, 2.20, 2.05, 2.75])
    gev    = np.array([1.00, 1.35, 1.65, 2.00, 2.45, 2.80])

    # Shade crossing intervals before drawing lines
    for i in range(len(direct) - 1):
        if direct[i + 1] < direct[i]:
            ax.axvspan(i, i + 1, alpha=0.13, color=MODEL_COLORS["ANN"], zorder=1)

    ax.plot(x, gev, "o-", color=MODEL_COLORS["GEV-NN"], lw=2.2, ms=5.5, zorder=4,
            label="GEV-constrained (monotone)")
    ax.plot(x, direct, "o-", color=MODEL_COLORS["ANN"], lw=2.2, ms=5.5, zorder=4,
            label="Direct ANN outputs (unconstrained)")

    # Label each crossing region
    for i in range(len(direct) - 1):
        if direct[i + 1] < direct[i]:
            mid_y = (direct[i] + direct[i + 1]) / 2
            ax.text(
                i + 0.5, mid_y + 0.12,
                f"Q{RETURN_PERIODS[i]}→Q{RETURN_PERIODS[i+1]}\ncrossing",
                ha="center", va="bottom",
                fontsize=8.0, color=MODEL_COLORS["ANN"], fontweight="semibold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{t}" for t in RETURN_PERIODS])
    ax.set_ylabel("Predicted flood quantile (relative)")
    ax.set_title("Schematic: unconstrained quantile heads can cross")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.set_ylim(0.55, 3.30)
    ax.set_axisbelow(True)
    _letter(ax, "a")


def _plot_any_violation(ax, model_summary: pd.DataFrame) -> None:
    use = model_summary[model_summary["experiment"].isin(["PUB", "PUR"])].copy()
    use["model"] = use["model"].astype(str)
    xpos = np.arange(len(MODEL_ORDER))
    width = 0.36

    all_pct: list[float] = []
    for offset, exp, hatch, alpha in [
        (-width / 2, "PUB", "",   0.90),
        ( width / 2, "PUR", "//", 0.65),
    ]:
        vals = []
        for m in MODEL_ORDER:
            sub = use[(use["model"] == m) & (use["experiment"].astype(str) == exp)]
            vals.append(float(sub["any_violation_rate"].iloc[0]) if not sub.empty else np.nan)

        pct = [v * 100 if np.isfinite(v) else np.nan for v in vals]
        all_pct.extend([v for v in pct if np.isfinite(v)])

        bars = ax.bar(
            xpos + offset, pct, width=width,
            color=[MODEL_COLORS.get(m, "#999999") for m in MODEL_ORDER],
            alpha=alpha, hatch=hatch, edgecolor="white", linewidth=0.8,
            label=exp, zorder=3,
        )

        for b, v in zip(bars, pct):
            if not np.isfinite(v):
                continue
            x_c = b.get_x() + b.get_width() / 2
            if v >= 0.5:
                ax.text(x_c, v + 0.6, f"{v:.1f}%",
                        ha="center", va="bottom", fontsize=7.5, fontweight="medium")
            else:
                # Explicitly label zero-violation models at baseline
                ax.text(x_c, 0.35, "0%",
                        ha="center", va="bottom", fontsize=7.0,
                        color="#666666", style="italic")

    max_pct = max(all_pct) if all_pct else 5.0

    # Vertical separator and group labels (positions derived from DIRECT_MODELS)
    n_direct = sum(1 for m in MODEL_ORDER if m in DIRECT_MODELS)
    sep_x = n_direct - 0.5
    ax.axvline(sep_x, color="#AAAAAA", linewidth=0.9, linestyle=":", zorder=2)
    y_text = max_pct * 1.14
    direct_center = (n_direct - 1) / 2
    gev_center = n_direct + (len(MODEL_ORDER) - n_direct - 1) / 2
    ax.text(direct_center, y_text, "Direct / independent",
            ha="center", va="bottom", fontsize=8.5, color="#555555", style="italic")
    ax.text(gev_center, y_text, "GEV-constrained",
            ha="center", va="bottom", fontsize=8.5, color="#555555", style="italic")

    ax.set_xticks(xpos)
    ax.set_xticklabels(MODEL_ORDER, rotation=38, ha="right")
    ax.set_ylim(0, max(6.0, max_pct * 1.40))
    ax.set_ylabel("Any-crossing violation rate (%)")
    ax.set_title("Violation rates concentrate in direct ANN predictions")
    ax.legend(frameon=False, ncol=2, title="Experiment", title_fontsize=9)
    ax.set_axisbelow(True)
    _letter(ax, "b")


def _plot_pair_heatmap(ax, pair_summary: pd.DataFrame) -> None:
    use = pair_summary[pair_summary["experiment"].astype(str) == "PUR"].copy()
    pairs = [f"Q{a}->Q{b}" for a, b in ADJ_PAIRS]
    mat = np.full((len(MODEL_ORDER), len(pairs)), np.nan)
    for i, m in enumerate(MODEL_ORDER):
        for j, p in enumerate(pairs):
            sub = use[(use["model"].astype(str) == m) & (use["adjacent_pair"] == p)]
            if not sub.empty:
                mat[i, j] = float(sub["violation_rate"].iloc[0])

    vmax = max(0.01, np.nanmax(mat))
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels(pairs, rotation=35, ha="right")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                text_color = "white" if mat[i, j] > vmax * 0.55 else "#222222"
                label = f"{mat[i, j] * 100:.1f}%" if mat[i, j] > 0 else "0%"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=8.0, color=text_color, fontweight="medium")

    # Dashed separator between Direct and GEV groups
    n_direct = sum(1 for m in MODEL_ORDER if m in DIRECT_MODELS)
    ax.axhline(n_direct - 0.5, color="#444444", linewidth=1.0, linestyle="--", alpha=0.55)

    ax.set_title("Pair-wise crossing violation rate (PUR)")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Violation rate")
    cbar.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    )
    ax.grid(False)
    _letter(ax, "c")


def _plot_worst_station(ax, example: pd.DataFrame) -> None:
    if example.empty:
        ax.text(0.5, 0.5, "No ANN example available",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        _letter(ax, "d")
        return

    station = str(example.iloc[0]["station_id"])
    exp = str(example.iloc[0]["experiment"])
    x = np.array(RETURN_PERIODS, dtype=float)

    true_vals = example.iloc[0][Q_TRUE].to_numpy(float)
    ax.plot(x, true_vals, "k--", lw=2.0, marker="s", ms=5, zorder=5,
            label="Observed GEV target")

    for model in MODEL_ORDER:
        sub = example[example["model"].astype(str) == model]
        if sub.empty:
            continue
        vals = sub.iloc[0][Q_PRED].to_numpy(float)
        ls = "-" if model in GEV_CONSTRAINED_MODELS else "--"
        ax.plot(x, vals, marker="o", ms=4.5, lw=1.8, linestyle=ls,
                color=MODEL_COLORS.get(model, "#666666"),
                label=model, alpha=0.93, zorder=4)

    # Highlight ANN crossing segments with thicker translucent stroke
    ann_sub = example[example["model"].astype(str) == "ANN"]
    if not ann_sub.empty:
        ann_vals = ann_sub.iloc[0][Q_PRED].to_numpy(float)
        for i in range(len(ann_vals) - 1):
            if np.isfinite(ann_vals[i]) and np.isfinite(ann_vals[i + 1]):
                if ann_vals[i] > ann_vals[i + 1]:
                    ax.plot([x[i], x[i + 1]], [ann_vals[i], ann_vals[i + 1]],
                            color=MODEL_COLORS["ANN"], lw=5.0, alpha=0.28, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticklabels([str(t) for t in RETURN_PERIODS])
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Flood quantile (m³/s)")
    ax.set_title(f"Worst-case ANN crossing: station {station} ({exp})")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    ax.set_axisbelow(True)
    _letter(ax, "d")


def _make_figure(
    model_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    example: pd.DataFrame,
) -> Path:
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    _plot_schematic(axes[0, 0])
    _plot_any_violation(axes[0, 1], model_summary)
    _plot_pair_heatmap(axes[1, 0], pair_summary)
    _plot_worst_station(axes[1, 1], example)
    fig.suptitle(
        "Explaining monotonicity failures in direct ANN flood-quantile prediction",
        fontsize=15, fontweight="bold", y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    out = OUT_FIG / f"{TAG}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Log-increment boxplot figure  (3×2 grid, one panel per adjacent pair)
# ---------------------------------------------------------------------------
def _plot_pair_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    t1: int,
    t2: int,
    show_ylabel: bool,
    letter: str,
    colors: dict | None = None,
) -> None:
    if colors is None:
        colors = MODEL_COLORS
    log_col = f"log_gap_Q{t1}_Q{t2}"
    models_present = [m for m in MODEL_ORDER if m in sub["model"].astype(str).unique()]
    n_m = len(models_present)
    xpos = np.arange(n_m)
    box_w = 0.52
    cap_w = box_w * 0.30

    # Subtle shading for GEV-constrained group
    n_direct = sum(1 for m in models_present if m in DIRECT_MODELS)
    if n_direct < n_m:
        ax.axvspan(n_direct - 0.5, n_m - 0.45,
                   alpha=0.045, color="#4477AA", zorder=0)

    for xi, model in enumerate(models_present):
        mdf = sub[sub["model"].astype(str) == model]
        if mdf.empty or log_col not in mdf.columns:
            continue
        vals = mdf[log_col].dropna().to_numpy(float)
        if len(vals) < 5:
            continue

        color = colors.get(model, "#999999")
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        p05, p95 = np.percentile(vals, [5, 95])
        min_val = float(np.nanmin(vals))
        neg_pct = float(np.mean(vals < 0) * 100)

        # IQR box
        ax.bar(xi, q75 - q25, bottom=q25, width=box_w,
               color=color, alpha=0.78, edgecolor="white", linewidth=0.9, zorder=3)
        # Median line — white, inset so it never exceeds box edges
        inset = 0.03
        ax.plot([xi - box_w / 2 + inset, xi + box_w / 2 - inset], [q50, q50],
                color="white", lw=1.8, zorder=4, solid_capstyle="butt")
        # Whisker stems
        for y_lo, y_hi in [(p05, q25), (q75, p95)]:
            ax.plot([xi, xi], [y_lo, y_hi], color=color, lw=0.95, zorder=3, alpha=0.88)
        # Whisker caps
        for y_cap in (p05, p95):
            ax.plot([xi - cap_w, xi + cap_w], [y_cap, y_cap],
                    color=color, lw=0.95, zorder=3, alpha=0.88)

        # Dotted extension to sample minimum for any model with crossings
        if min_val < 0:
            ax.plot([xi, xi], [min_val, p05],
                    color=color, lw=0.85, ls=":", zorder=2, alpha=0.80)
            ax.plot(xi, min_val, marker="v", ms=4.0, color=color,
                    zorder=5, alpha=0.85, markeredgecolor="none")

        # Violation % labels — black bold text + white background for all models
        if neg_pct >= 5.0:
            ax.text(xi, min_val - 0.018, f"{neg_pct:.0f}%",
                    ha="center", va="top", fontsize=7.8, fontweight="bold",
                    color="#000000",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.80))
        elif neg_pct > 0:
            ax.text(xi, min_val - 0.014, f"{neg_pct:.1f}%",
                    ha="center", va="top", fontsize=6.5, fontweight="bold",
                    color="#000000",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.80))

    # Zero-crossing reference line
    ax.axhline(0, color="#444444", lw=0.9, ls=(0, (6, 4)), alpha=0.65, zorder=2)

    # Thin dotted separator between Direct and GEV groups
    if 0 < n_direct < n_m:
        ax.axvline(n_direct - 0.5, color="#AAAAAA", lw=0.75, ls=":", zorder=1)

    ax.set_xlim(-0.6, n_m - 0.4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(models_present, rotation=28, ha="right", fontsize=_FS["tick"])
    ax.set_axisbelow(True)

    if show_ylabel:
        ax.set_ylabel(r"log($Q_{\rm next}$ / $Q_{\rm current}$)",
                      fontsize=_FS["label"])

    # Panel letter — bold, top-left inside axes
    ax.text(0.015, 0.975, letter,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=13, fontweight="bold", color="#111111", zorder=10)

    # Pair label tag — top-right, rounded box
    ax.text(0.975, 0.975, f"Q{t1}→Q{t2}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color="#222222",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.93, linewidth=0.8),
            zorder=10)


def _draw_legend_panel(ax: plt.Axes, models_present: list) -> None:
    ax.axis("off")

    handles = [
        matplotlib.patches.Patch(
            facecolor=MODEL_COLORS.get(m, "#999999"),
            edgecolor="white", alpha=0.82, label=m, linewidth=0.9,
        )
        for m in models_present
    ]
    handles.append(
        matplotlib.lines.Line2D(
            [0], [0], color="#666666", lw=0.85, ls=":",
            marker="v", ms=4.5, markeredgecolor="none",
            label="Min (any crossing)",
        )
    )

    leg = ax.legend(
        handles=handles, frameon=True, loc="upper center",
        fontsize=9.5, title="Model", title_fontsize=10,
        ncol=1, framealpha=1.0, edgecolor="#DDDDDD",
        bbox_to_anchor=(0.5, 0.98),
        borderpad=0.9, handlelength=1.8, handleheight=1.2,
    )
    leg.get_frame().set_linewidth(0.8)


def _make_logincrement_figure(df: pd.DataFrame) -> list[Path]:
    _setup_style()
    if "PUR" not in df["experiment"].astype(str).unique():
        LOG.warning("No PUR data; skipping log-increment figure.")
        return []

    sub = df[df["experiment"].astype(str) == "PUR"].copy()

    # Global y-axis limits across all pairs/models
    all_mins, all_p95s = [], []
    for t1, t2 in ADJ_PAIRS:
        log_col = f"log_gap_Q{t1}_Q{t2}"
        for model in MODEL_ORDER:
            mdf = sub[sub["model"].astype(str) == model]
            if not mdf.empty and log_col in mdf.columns:
                vals = mdf[log_col].dropna().to_numpy(float)
                if len(vals) >= 5:
                    all_mins.append(float(np.nanmin(vals)))
                    all_p95s.append(float(np.percentile(vals, 95)))

    y_min_g = min(all_mins) if all_mins else -0.3
    y_max_g = max(all_p95s) if all_p95s else 0.9
    y_lo = y_min_g - abs(y_min_g) * 0.28 - 0.06
    y_hi = y_max_g * 1.14

    saved: list[Path] = []
    letters = "abcde"

    for scheme_id, colors in _COLOR_SCHEMES.items():
        # 3 rows × 2 cols; 6th cell (axes[2,1]) is hidden
        fig, axes = plt.subplots(3, 2, figsize=(10.5, 13.0), sharey=True)

        for idx, (t1, t2) in enumerate(ADJ_PAIRS):
            row, col = idx // 2, idx % 2
            _plot_pair_panel(
                axes[row, col], sub, t1, t2,
                show_ylabel=(col == 0),
                letter=letters[idx],
                colors=colors,
            )

        # Hide unused 6th panel
        axes[2, 1].axis("off")

        # Shared y limits (sharey propagates to all panels)
        axes[0, 0].set_ylim(y_lo, y_hi)

        fig.tight_layout(rect=(0, 0, 1, 1.0), h_pad=3.5, w_pad=1.5)
        out = OUT_FIG / f"{TAG}_log_increment_scheme{scheme_id}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        saved.append(out)
        LOG.info("Saved log-increment figure (scheme %s): %s", scheme_id, out)

    return saved


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{100 * x:.1f}%"


def _write_report(model_summary: pd.DataFrame, pair_summary: pd.DataFrame, example: pd.DataFrame, fig_path: Path) -> Path:
    def _lookup(model: str, exp: str, col: str) -> float:
        sub = model_summary[
            (model_summary["model"].astype(str) == model) & (model_summary["experiment"].astype(str) == exp)
        ]
        if sub.empty or col not in sub:
            return np.nan
        return float(sub[col].iloc[0])

    ann_pub = _lookup("ANN", "PUB", "any_violation_rate")
    ann_pur = _lookup("ANN", "PUR", "any_violation_rate")
    gev_pub = np.nanmean([_lookup(m, "PUB", "any_violation_rate") for m in GEV_CONSTRAINED_MODELS])
    gev_pur = np.nanmean([_lookup(m, "PUR", "any_violation_rate") for m in GEV_CONSTRAINED_MODELS])

    worst_pair = pd.DataFrame()
    if not pair_summary.empty:
        worst_pair = (
            pair_summary[(pair_summary["model"].astype(str) == "ANN") & (pair_summary["experiment"].astype(str) == "PUR")]
            .sort_values("violation_rate", ascending=False)
            .head(1)
        )
    worst_pair_text = "NA"
    if not worst_pair.empty:
        worst_pair_text = (
            f"{worst_pair.iloc[0]['adjacent_pair']} "
            f"({_fmt_pct(float(worst_pair.iloc[0]['violation_rate']))})"
        )

    example_text = "NA"
    if not example.empty:
        example_text = f"{example.iloc[0]['station_id']} ({example.iloc[0]['experiment']})"

    text = f"""# Monotonicity Interpretation

## What the diagnostic shows

- Direct ANN predictions have no built-in ordering constraint across return periods. Each output head can reduce its own loss while still producing Q_T values that are locally inconsistent, for example Q50 > Q100.
- In the available results, ANN any-crossing rates are {_fmt_pct(ann_pub)} in PUB and {_fmt_pct(ann_pur)} in PUR.
- Averaged across the GEV-constrained variants, any-crossing rates are {_fmt_pct(gev_pub)} in PUB and {_fmt_pct(gev_pur)} in PUR.
- The largest ANN PUR crossing hotspot is {worst_pair_text}.
- The example curve in the figure uses station {example_text}.

## Why the unconstrained ANN crosses

The direct ANN treats the six return-period quantiles as six supervised targets. Even if the observed targets are monotone, the network output layer does not know that Q2 <= Q5 <= Q10 <= Q20 <= Q50 <= Q100 must hold. Small independent errors across adjacent outputs are therefore enough to invert the order, especially for high return periods where sampling uncertainty is larger and the target signal is noisier.

## Why unconstrained tree/kernel models (RF, XGBoost, SVM) rarely cross

These models are also not explicitly constrained, but their inductive biases are more conservative than independent ANN heads. RF averages over many trees whose terminal-leaf predictions are local means of training samples; because adjacent return periods select similar neighborhoods in feature space these averages usually preserve ordering. XGBoost is a gradient-boosted ensemble of shallow trees with regularisation that penalises large residuals; its per-period predictions are also dominated by smoothed local averages. SVM with a kernel maps inputs to a high-dimensional space and finds a maximum-margin hyperplane; the resulting regression surface is globally smooth and changes slowly across the feature space, so adjacent-period predictions rarely diverge sharply. None of these models has a formal monotonicity guarantee, but their ensemble/kernel averaging and limited extrapolation ability make crossings rare and shallow compared to multi-head ANN models.

The ANN-Single model is different: it trains separate neural networks for each return period. The six networks can learn different nonlinear surfaces, respond differently to sparse tail information, and extrapolate differently in PUR regions. This produces larger asynchronous errors between adjacent return periods. Therefore the main issue is not only lack of an explicit monotonicity constraint; it is lack of constraint combined with a flexible extrapolative model class.

## Why the GEV-constrained model improves monotonicity

The GEV-constrained models do not freely predict six unrelated quantiles. They predict distribution parameters and then evaluate all return-period quantiles from the same frequency curve. With a positive scale parameter and valid return probabilities, the resulting quantile function is ordered by construction. The constraint therefore removes an entire class of physically impossible solutions rather than merely penalizing them after prediction.

## How to phrase it in the paper

The monotonicity issue reflects a structural mismatch in the unconstrained ANN: it optimizes pointwise quantile accuracy but does not encode the ordering law of frequency analysis. The low RF crossing rate shows that an unconstrained model can still behave almost monotonically when its inductive bias is dominated by local averaging and weak extrapolation. By contrast, independent ANN return-level models are flexible enough to learn mutually inconsistent response surfaces, particularly in the distribution tail and in ungauged-region extrapolation. The GEV-constrained model improves because the predicted quantiles are generated from a single parametric extreme-value distribution, forcing all return levels to lie on one coherent flood-frequency curve. This constraint acts as a hydrological regularizer and is especially helpful under extrapolative PUR conditions.

## Files

- Figure: `{fig_path}`
- Model summary: `{OUT_DATA / "monotonicity_model_summary.csv"}`
- Pair summary: `{OUT_DATA / "monotonicity_pair_summary.csv"}`
- Station-level metrics: `{OUT_DATA / "monotonicity_station_metrics.csv"}`
"""
    out = OUT_DATA / "monotonicity_interpretation.md"
    out.write_text(text, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    LOG.info("=" * 72)
    LOG.info("%s - start", TAG)
    LOG.info("=" * 72)

    st, heat = _read_inputs()
    st = _add_monotonicity_metrics(st)

    model_summary = _model_summary(st, heat)
    pair_summary = _pair_summary(st)
    climate_summary = _climate_summary(st)
    example = _worst_example_rows(st)

    st.to_csv(OUT_DATA / "monotonicity_station_metrics.csv", index=False)
    model_summary.to_csv(OUT_DATA / "monotonicity_model_summary.csv", index=False)
    pair_summary.to_csv(OUT_DATA / "monotonicity_pair_summary.csv", index=False)
    climate_summary.to_csv(OUT_DATA / "monotonicity_climate_summary.csv", index=False)
    example.to_csv(OUT_DATA / "monotonicity_worst_ann_example.csv", index=False)

    fig_path = _make_figure(model_summary, pair_summary, example)
    try:
        logincrement_figs = _make_logincrement_figure(st)
    except Exception as exc:
        LOG.error("Log-increment figure failed: %s", exc, exc_info=True)
        logincrement_figs = []
    report_path = _write_report(model_summary, pair_summary, example, fig_path)

    LOG.info("Saved station metrics: %s", OUT_DATA / "monotonicity_station_metrics.csv")
    LOG.info("Saved model summary:   %s", OUT_DATA / "monotonicity_model_summary.csv")
    LOG.info("Saved pair summary:    %s", OUT_DATA / "monotonicity_pair_summary.csv")
    LOG.info("Saved 4-panel figure:  %s", fig_path)
    for p in logincrement_figs:
        LOG.info("Saved log-increment figure: %s", p)
    LOG.info("Saved interpretation report: %s", report_path)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
