# -*- coding: utf-8 -*-
"""
FigureS4_GHM_Leakage_Analysis.py  —  Nature/WRR submission-grade figures

Reproduces Figure S4.

Figure contract
───────────────
Core conclusion : GHM calibration leakage does not inflate Hybrid Model
  performance — Δ NSE shows no positive correlation with per-basin leakage
  rate, and the improvement is consistent across all return periods.
  Three-layer causal evidence (Figure 3) further demonstrates that
  performance gains arise from physical knowledge transfer, not leakage.

Two-figure output (1 + 3 panel layout each)
  fig_T2   : T = 2 yr — leakage background + three-layer evidence
  fig_T100 : T = 100 yr — leakage background + three-layer evidence

Panel structure (identical layout, different RP data in b and d)
  (a) Full-width: GHM calibration leakage stacked bar (15 PUR basins)
  (b) Layer 1 — Within-basin ΔNSE dumbbell (contaminated vs clean)
  (c) Layer 2 — Clean-only NSE line chart across all return periods
  (d) Layer 3 — Dose-response scatter: leakage rate vs ΔNSE

Backend    : Python / matplotlib (exclusive)
Export     : SVG (editable text, svg.fonttype=none)  +  PNG 300 dpi
"""
from __future__ import annotations
import logging, sys, warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats as spstats
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.paths import DATA_PROCEED, FIGURE_ROOT, stage_dir

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

# ── Paths ──────────────────────────────────────────────────────────────────
TAG      = "FigureS4_GHM_Leakage_Analysis"
OUT_DATA = stage_dir(DATA_PROCEED, TAG)
OUT_FIG  = stage_dir(FIGURE_ROOT, TAG)

RETURN_PERIODS = [2, 5, 10, 20, 50, 100]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PALETTE & STYLE                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PAL = {
    "blue_main":      "#0F4D92",
    "blue_soft":      "#3775BA",
    "red_strong":     "#B64342",
    "neutral_light":  "#D4D4D4",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
    "neutral_black":  "#1A1A1A",
    "teal":           "#42949E",
    "clean_fill":     "#C8D8EA",   # pale blue for clean stations
    "sand":           "#F0EBE1",
}

# Semantic roles
C_GEVNN  = PAL["neutral_dark"]    # GEV-NN baseline  (dark grey)
C_HYBRID = PAL["blue_main"]       # Hybrid model     (deep blue)
C_LEAK   = PAL["red_strong"]      # contaminated     (red)
C_CLEAN  = PAL["clean_fill"]      # clean stations   (pale blue)
C_CONT   = PAL["red_strong"]      # contaminated group line
C_CLN    = PAL["blue_soft"]       # clean group line

# Continent palette — used for basin-label colors & scatter markers
CONT_PAL = {
    "NA": "#2166AC",   # blue
    "EU": "#B2182B",   # dark red
    "SA": "#E08214",   # amber
    "AU": "#4DAC26",   # green
    "AF": "#42949E",   # teal
}


def _apply_style():
    """Nature / WRR publication-grade rcParams."""
    plt.rcParams.update({
        # ── Fonts ────────────────────────────────────────────────────────────
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype":          "none",
        "pdf.fonttype":          42,
        "font.size":             8.5,
        "axes.labelsize":        8.5,
        "axes.titlesize":        9.0,
        "axes.titleweight":      "bold",
        # ── Spines ───────────────────────────────────────────────────────────
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.linewidth":        0.8,
        "axes.edgecolor":        "#444444",
        "axes.labelcolor":       "#1A1A1A",
        "text.color":            "#1A1A1A",
        # ── Ticks: inward, clean ─────────────────────────────────────────────
        "xtick.labelsize":       8.0,
        "ytick.labelsize":       8.0,
        "xtick.major.size":      3.5,
        "ytick.major.size":      3.5,
        "xtick.major.width":     0.7,
        "ytick.major.width":     0.7,
        "xtick.direction":       "in",
        "ytick.direction":       "in",
        # ── Legend ───────────────────────────────────────────────────────────
        "legend.frameon":        False,
        "legend.fontsize":       8.0,
        "legend.title_fontsize": 8.0,
        "legend.handlelength":   1.4,
        "legend.handleheight":   0.9,
        "legend.labelspacing":   0.35,
        "legend.borderpad":      0.4,
        # ── Lines & patches ──────────────────────────────────────────────────
        "lines.linewidth":       1.4,
        "patch.linewidth":       0.7,
        # ── Grid ─────────────────────────────────────────────────────────────
        "axes.grid":             False,
        # ── Save ─────────────────────────────────────────────────────────────
        "figure.facecolor":      "white",
        "axes.facecolor":        "white",
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.05,
    })


def _lbl(ax, ch: str, x: float = -0.10, y: float = 1.03):
    """Bold non-italic panel label (matches Figure3_ML_Performance.py style)."""
    ax.text(x, y, ch, transform=ax.transAxes,
            fontsize=12, fontweight="bold", fontstyle="normal",
            va="top", ha="left", color="#111111")


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")


def _light_grid(ax, axis: str = "y"):
    ax.set_axisbelow(True)
    kw = dict(linestyle="--", linewidth=0.4, color="#E0E0E0", zorder=0)
    if axis in ("y", "both"):
        ax.yaxis.grid(True, **kw)
    if axis in ("x", "both"):
        ax.xaxis.grid(True, **kw)


def _save(fig, stem: Path):
    """Save PNG only (300 dpi)."""
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved  %s.png", stem.name)


def _make_short_names(basins: list[str]) -> dict[str, str]:
    """
    Map full basin labels to short names: continent prefix + sequential index.
    E.g. "EU_2020018240" -> "EU2", "NA_7020038340" -> "NA1".
    Groups are sorted alphabetically within each continent, then numbered 1-N.
    """
    from collections import defaultdict
    cont_groups: dict[str, list] = defaultdict(list)
    for b in sorted(basins):
        cont = b.split("_")[0] if "_" in b else "XX"
        cont_groups[cont].append(b)
    short: dict[str, str] = {}
    for cont in sorted(cont_groups):
        for i, basin in enumerate(sorted(cont_groups[cont]), 1):
            short[basin] = f"{cont}{i}"
    return short


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING & COMPUTATION                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_all():
    match_df = pd.read_csv(DATA_PROCEED / "05_GHM_LeakageMatch" / "pur_stations_exact_match.csv")
    match_df["station_id"] = match_df["station_id"].astype(str).str.strip()
    contaminated = set(match_df[match_df["matched_ghm_train"]]["station_id"])

    pur_gdf = gpd.read_file(DATA_PROCEED / "05_PUR_Basin_Select/pur_retained_basins.shp")
    if "basin_labe" in pur_gdf.columns:
        pur_gdf = pur_gdf.rename(columns={"basin_labe": "basin_label"})
    PUR_LABELS = sorted(pur_gdf["basin_label"].tolist())

    def load_preds(ftag):
        frames = []
        for f in sorted((DATA_PROCEED / "11_GEV_NN").glob(
                f"predictions_GEV_NN_ST_PUR_*_{ftag}.csv")):
            df = pd.read_csv(f)
            df["station_id"] = df["station_id"].astype(str).str.strip()
            df["basin_label"] = f.stem.split("_PUR_")[1].split("_s")[0]
            frames.append(df)
        raw = pd.concat(frames, ignore_index=True)
        qt  = [f"Q{t}_true" for t in RETURN_PERIODS]
        qp  = [f"Q{t}_pred" for t in RETURN_PERIODS]
        agg = pd.concat([
            raw.groupby(["station_id", "basin_label"])[qt].first(),
            raw.groupby(["station_id", "basin_label"])[qp].median(),
        ], axis=1).reset_index()
        agg["station_id"]   = agg["station_id"].astype(str).str.strip()
        agg["contaminated"] = agg["station_id"].isin(contaminated)
        return agg

    base_df   = load_preds("base")
    hybrid_df = load_preds("+flow")
    LOG.info("Loaded  GEV-NN: %d  |  Hybrid: %d  |  contaminated: %d",
             len(base_df), len(hybrid_df), int(base_df["contaminated"].sum()))
    return PUR_LABELS, base_df, hybrid_df


def _metrics(obs, pred):
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    o, p = obs[mask], pred[mask]
    if len(o) < 3:
        return dict(NSE=np.nan, PBIAS=np.nan, rRMSE=np.nan)
    err = p - o
    d   = float(np.sum((o - o.mean()) ** 2))
    return dict(
        NSE  = float(1 - np.sum(err**2) / d) if d > 0 else np.nan,
        PBIAS= float(100 * np.sum(err) / np.sum(o)),
        rRMSE= float(100 * np.sqrt(np.mean(err**2)) / abs(o.mean())),
    )


def compute_basin_metrics(PUR_LABELS, base_df, hybrid_df, rp: int) -> pd.DataFrame:
    rows = []
    for basin in PUR_LABELS:
        b = base_df[base_df["basin_label"] == basin]
        h = hybrid_df[hybrid_df["basin_label"] == basin]
        if b.empty:
            continue
        tc, pc = f"Q{rp}_true", f"Q{rp}_pred"
        mb = _metrics(b[tc].values, b[pc].values)
        mh = _metrics(h[tc].values, h[pc].values)
        n_total, n_cont = len(b), int(b["contaminated"].sum())
        rows.append({
            "basin":       basin,
            "continent":   basin[:2],
            "n_total":     n_total,
            "n_cont":      n_cont,
            "leakage_pct": round(100.0 * n_cont / n_total, 1),
            **{f"{k}_base":   round(v, 4) for k, v in mb.items()},
            **{f"{k}_hybrid": round(v, 4) for k, v in mh.items()},
            **{f"d{k}": round(mh[k] - mb[k], 4)
               if np.isfinite(mb[k]) and np.isfinite(mh[k]) else np.nan
               for k in ("NSE", "PBIAS", "rRMSE")},
        })
    return (pd.DataFrame(rows)
            .sort_values("leakage_pct", ascending=False)
            .reset_index(drop=True))


def compute_delta_rp(base_df, hybrid_df) -> pd.DataFrame:
    # Align by station_id + basin_label so contaminated labels map correctly
    # regardless of row-order differences between the two DataFrames.
    rows = []
    for rp in RETURN_PERIODS:
        tc, pc = f"Q{rp}_true", f"Q{rp}_pred"
        merged = (
            base_df[["station_id", "basin_label", "contaminated", tc, pc]]
            .merge(
                hybrid_df[["station_id", "basin_label", pc]].rename(
                    columns={pc: "pred_hyb"}),
                on=["station_id", "basin_label"],
            )
        )
        for label, mask in [("Contaminated",  merged["contaminated"]),
                             ("Clean",        ~merged["contaminated"])]:
            sub = merged[mask]
            mb  = _metrics(sub[tc].values, sub[pc].values)
            mh  = _metrics(sub[tc].values, sub["pred_hyb"].values)
            rows.append({"rp": rp, "group": label,
                         "NSE_base":    mb["NSE"],  "NSE_hybrid":   mh["NSE"],
                         "PBIAS_base":  mb["PBIAS"],"PBIAS_hybrid": mh["PBIAS"],
                         "rRMSE_base":  mb["rRMSE"],"rRMSE_hybrid": mh["rRMSE"],
                         "dNSE":   round(mh["NSE"]   - mb["NSE"],   4),
                         "dPBIAS": round(mh["PBIAS"]  - mb["PBIAS"],  4),
                         "drRMSE": round(mh["rRMSE"]  - mb["rRMSE"],  4),
                         })
    return pd.DataFrame(rows)


def compute_within_basin_leakage_test(base_df, hybrid_df, rp: int,
                                      min_n: int = 10) -> dict:
    """
    Layer 1 causal test: within each PUR basin, compare ΔNSE for contaminated
    vs clean stations.  δⱼ = ΔNSE_cont − ΔNSE_clean per basin.
    A non-zero δⱼ would indicate leakage-driven inflation; δⱼ ≈ 0 refutes it.

    Returns a dict with keys:
        per_basin     : DataFrame of per-basin results
        n_basins      : int
        mean_delta    : float
        median_delta  : float
        ci_95         : (lo, hi) tuple
        wilcoxon_p    : float (nan if not enough basins)
        dNSE_cont_mean : float
        dNSE_clean_mean: float
    """
    tc, pc = f"Q{rp}_true", f"Q{rp}_pred"

    # Merge: keep one copy of the true column (tc) + both pred columns.
    # Using hybrid_df's pred only avoids the redundant tc_hyb column.
    merged = (
        base_df[["station_id", "basin_label", "contaminated", tc, pc]]
        .merge(
            hybrid_df[["station_id", "basin_label", pc]].rename(
                columns={pc: "pred_hyb"}),
            on=["station_id", "basin_label"],
        )
    )

    rows = []
    for basin, grp in merged.groupby("basin_label"):
        cont_grp  = grp[grp["contaminated"] == True]
        clean_grp = grp[grp["contaminated"] == False]

        if len(cont_grp) < min_n or len(clean_grp) < min_n:
            continue

        mb_cont  = _metrics(cont_grp[tc].values,  cont_grp[pc].values)
        mh_cont  = _metrics(cont_grp[tc].values,  cont_grp["pred_hyb"].values)
        mb_clean = _metrics(clean_grp[tc].values, clean_grp[pc].values)
        mh_clean = _metrics(clean_grp[tc].values, clean_grp["pred_hyb"].values)

        nse_base_cont  = mb_cont["NSE"]
        nse_hyb_cont   = mh_cont["NSE"]
        nse_base_clean = mb_clean["NSE"]
        nse_hyb_clean  = mh_clean["NSE"]

        if not all(np.isfinite([nse_base_cont, nse_hyb_cont,
                                 nse_base_clean, nse_hyb_clean])):
            continue

        dNSE_cont  = nse_hyb_cont  - nse_base_cont
        dNSE_clean = nse_hyb_clean - nse_base_clean
        delta_j    = dNSE_cont - dNSE_clean   # KEY: excess gain for cont. stations

        rows.append({
            "basin":        basin,
            "continent":    basin[:2],
            "n_cont":       len(cont_grp),
            "n_clean":      len(clean_grp),
            "NSE_base_cont":  round(nse_base_cont,  4),
            "NSE_hyb_cont":   round(nse_hyb_cont,   4),
            "NSE_base_clean": round(nse_base_clean,  4),
            "NSE_hyb_clean":  round(nse_hyb_clean,  4),
            "dNSE_cont":    round(dNSE_cont,  4),
            "dNSE_clean":   round(dNSE_clean, 4),
            "delta":        round(delta_j,    4),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return dict(
            per_basin=df, n_basins=0, min_n=min_n,
            mean_delta=np.nan, median_delta=np.nan,
            ci_95=(np.nan, np.nan), wilcoxon_p=np.nan,
            dNSE_cont_mean=np.nan, dNSE_clean_mean=np.nan,
        )

    deltas = df["delta"].values
    mean_d  = float(np.mean(deltas))
    med_d   = float(np.median(deltas))
    sem_d   = float(spstats.sem(deltas)) if len(deltas) > 1 else np.nan

    if len(deltas) >= 2 and np.isfinite(sem_d) and sem_d > 0:
        ci = spstats.t.interval(0.95, len(deltas) - 1,
                                loc=mean_d, scale=sem_d)
    else:
        ci = (np.nan, np.nan)

    if len(deltas) >= 3:
        try:
            _, wilcoxon_p = spstats.wilcoxon(deltas, alternative="greater")
        except Exception:
            wilcoxon_p = np.nan
    else:
        wilcoxon_p = np.nan

    return dict(
        per_basin       = df,
        n_basins        = len(df),
        min_n           = min_n,
        mean_delta      = mean_d,
        median_delta    = med_d,
        ci_95           = ci,
        wilcoxon_p      = wilcoxon_p,
        dNSE_cont_mean  = float(df["dNSE_cont"].mean()),
        dNSE_clean_mean = float(df["dNSE_clean"].mean()),
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DRAWING PRIMITIVES                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _draw_leakage(ax, t, short_names: dict):
    basins = t["basin"].tolist()
    conts  = t["continent"].tolist()
    n      = len(basins)
    x      = np.arange(n)
    lk     = t["leakage_pct"].values
    cl     = 100 - lk

    ax.bar(x, cl, color=C_CLEAN, linewidth=0, zorder=3)
    ax.bar(x, lk, bottom=cl, color=C_LEAK, linewidth=0, zorder=3)

    # Value labels inside / above bars
    for i, (lv, cv) in enumerate(zip(lk, cl)):
        if lv >= 10:
            ax.text(i, cv + lv / 2, f"{lv:.0f}%",
                    ha="center", va="center", fontsize=7.0,
                    color="white", fontweight="bold")
        elif lv > 0:
            ax.text(i, 101.5, f"{lv:.0f}%",
                    ha="center", va="bottom", fontsize=6.5,
                    color=C_LEAK, fontweight="bold")

    # Short names on x-axis
    ax.set_xticks(x)
    lbls = ax.set_xticklabels(
        [short_names.get(b, b) for b in basins],
        fontsize=8.0, rotation=45, ha="right")
    for lbl, ct in zip(lbls, conts):
        lbl.set_color(CONT_PAL.get(ct, PAL["neutral_dark"]))
        lbl.set_fontweight("bold")

    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylabel("Station fraction (%)")
    ax.axhline(100, color="#CCCCCC", lw=0.8, ls="--", zorder=2)
    _light_grid(ax, axis="y")
    _despine(ax)

    type_handles = [
        Patch(facecolor=C_LEAK,  linewidth=0, label="Leakage"),
        Patch(facecolor=C_CLEAN, linewidth=0, label="Clean"),
    ]
    ax.legend(handles=type_handles, loc="upper right", ncol=1,
              handlelength=1.4, handletextpad=0.5, labelspacing=0.45,
              borderpad=0.5, fontsize=8.5)



def _draw_scatter(ax, t: pd.DataFrame, rp_label: str, short_names: dict):
    """Scatter: per-basin leakage rate vs Δ NSE (Hybrid − GEV-NN)."""
    valid = t[["leakage_pct", "dNSE", "continent",
                "n_total", "basin"]].dropna()
    if valid.empty:
        return
    lk   = valid["leakage_pct"].values.astype(float)
    dn   = valid["dNSE"].values.astype(float)
    sz   = np.clip(valid["n_total"].values / 1.5, 30, 180)
    sc_c = [CONT_PAL.get(c, PAL["neutral_dark"]) for c in valid["continent"]]

    # Small jitter to separate overlapping points (x only; keeps y accurate)
    rng  = np.random.default_rng(42)
    lk_j = lk + rng.uniform(-0.8, 0.8, size=len(lk))

    # Show ALL points — no IQR clipping, no triangle markers
    ax.scatter(lk_j, dn, s=sz,
               c=sc_c, edgecolors="#333333", linewidths=0.55, alpha=0.90, zorder=4)

    from adjustText import adjust_text
    texts = []
    for idx_r, (_, row) in enumerate(valid.iterrows()):
        col  = CONT_PAL.get(row["continent"], "#555")
        name = short_names.get(row["basin"], row["continent"][:2])
        txt  = ax.text(lk_j[idx_r], row["dNSE"], name,
                       fontsize=7.0, color=col, fontweight="bold",
                       zorder=6,
                       bbox=dict(boxstyle="square,pad=0.08", fc="white",
                                 ec="none", alpha=0.65))
        texts.append(txt)
    adjust_text(
        texts, ax=ax,
        expand=(1.8, 2.2),
        force_text=(0.6, 0.9),
        force_points=(0.4, 0.6),
        arrowprops=dict(arrowstyle="-", color="#BBBBBB",
                        lw=0.6, shrinkA=3, shrinkB=3),
        max_move=0.5,
    )

    # OLS + 95 % CI (original un-jittered coordinates)
    if len(lk) >= 4:
        res = spstats.linregress(lk, dn)
        xln = np.linspace(lk.min(), lk.max(), 100)
        yln = res.slope * xln + res.intercept
        n_p, x_m = len(lk), lk.mean()
        se  = np.sqrt(np.sum((dn - (res.slope * lk + res.intercept))**2) / (n_p - 2))
        ci  = spstats.t.ppf(0.975, n_p - 2) * se * np.sqrt(
            1/n_p + (xln - x_m)**2 / np.sum((lk - x_m)**2))
        ax.plot(xln, yln, color="#888888", lw=1.2, ls="--", alpha=0.9, zorder=3)
        ax.fill_between(xln, yln - ci, yln + ci,
                        color="#888888", alpha=0.12, zorder=2)

    pad = (dn.max() - dn.min()) * 0.12
    ax.set_ylim(dn.min() - pad, dn.max() + pad)
    ax.axhline(0, color="#BBBBBB", lw=0.9, ls="--", zorder=1)
    _light_grid(ax, axis="y")
    ax.set_xlabel("Leakage rate per PUR region (%)")
    ax.set_ylabel(r"$\Delta$NSE  (Hybrid $-$ GEV-NN)")
    _despine(ax)




def _draw_within_basin_cmp(ax, wb_result: dict, t_sorted: pd.DataFrame,
                            rp_label: str, short_names: dict,
                            show_ylabels: bool = True):
    """
    Horizontal dumbbell chart: per-basin ΔNSE for contaminated vs clean stations.
    Basins sorted by leakage_pct (high → low) from t_sorted.
    """
    pb = wb_result["per_basin"]

    if pb.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5,
                f"Insufficient data\n(< 10 stations per group)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color=PAL["neutral_mid"])
        ax.set_title(f"Within-basin: ΔNSE cont. vs clean  [{rp_label}]",
                     fontsize=8.5, pad=5)
        return

    # Merge with t_sorted to get leakage_pct for ordering
    order_df = t_sorted[["basin", "leakage_pct"]].copy()
    plot_df  = pb.merge(order_df, on="basin", how="left")
    plot_df  = plot_df.sort_values("leakage_pct", ascending=False).reset_index(drop=True)

    basins  = plot_df["basin"].tolist()
    conts   = plot_df["continent"].tolist()
    d_cont  = plot_df["dNSE_cont"].values.astype(float)
    d_clean = plot_df["dNSE_clean"].values.astype(float)
    n       = len(basins)
    y       = np.arange(n)

    _light_grid(ax, axis="x")
    ax.set_axisbelow(True)

    for i, (yi, dc, dk, cont) in enumerate(zip(y, d_cont, d_clean, conts)):
        if not (np.isfinite(dc) and np.isfinite(dk)):
            continue
        col = CONT_PAL.get(cont, PAL["neutral_mid"])
        ax.plot([dc, dk], [yi, yi],
                color=col, lw=1.6, alpha=0.50,
                solid_capstyle="round", zorder=2)
        ax.scatter(dc, yi, s=60, facecolor="white",
                   edgecolors=C_CONT, linewidths=1.5,
                   marker="o", zorder=4)
        ax.scatter(dk, yi, s=52, color=C_CLN,
                   edgecolors="white", linewidths=0.5,
                   marker="s", zorder=4)

    ax.axvline(0, color="#999999", lw=0.9, ls="--", zorder=2)

    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()
    ax.set_yticks(y)

    if show_ylabels:
        ylbls = ax.set_yticklabels(
            [short_names.get(b, b) for b in basins], fontsize=8.0)
        for lbl, ct in zip(ylbls, conts):
            lbl.set_color(CONT_PAL.get(ct, PAL["neutral_dark"]))
            lbl.set_fontweight("bold")
    else:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks_position("none")
        ax.spines["left"].set_visible(False)

    ax.set_xlabel(r"$\Delta$NSE  (Hybrid $-$ GEV-NN)")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markersize=9,
               markerfacecolor="white", markeredgecolor=C_CONT,
               markeredgewidth=1.5, label="Contaminated"),
        Line2D([0], [0], marker="s", color="none", markersize=9,
               markerfacecolor=C_CLN, markeredgecolor="white",
               markeredgewidth=0.5, label="Clean"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              handlelength=0.9, labelspacing=0.40, borderpad=0.45)
    _despine(ax)




def _draw_clean_perf(ax, drp: pd.DataFrame):
    """
    Paired bar + line chart: GEV-NN vs Hybrid NSE for CLEAN stations only.
    Bars show absolute NSE; a secondary line shows ΔNSE improvement.
    """
    clean = drp[drp["group"] == "Clean"].sort_values("rp").reset_index(drop=True)
    rp_x  = np.array(RETURN_PERIODS)

    clean_idx = clean.set_index("rp")
    nse_base = np.array([clean_idx.loc[rp, "NSE_base"]   if rp in clean_idx.index else np.nan
                         for rp in rp_x], dtype=float)
    nse_hyb  = np.array([clean_idx.loc[rp, "NSE_hybrid"] if rp in clean_idx.index else np.nan
                         for rp in rp_x], dtype=float)
    delta    = nse_hyb - nse_base

    x     = np.arange(len(rp_x))
    w     = 0.32

    # Paired bars
    bars_b = ax.bar(x - w/2, nse_base, width=w, color=C_GEVNN,
                    alpha=0.82, linewidth=0, zorder=3, label="GEV-NN")
    bars_h = ax.bar(x + w/2, nse_hyb,  width=w, color=C_HYBRID,
                    alpha=0.82, linewidth=0, zorder=3, label="Hybrid")

    # ΔNSE as secondary axis line
    ax2 = ax.twinx()
    ax2.plot(x, delta, color="#E07B00", lw=1.8, marker="D",
             ms=5.5, zorder=5, ls="-", label=r"$\Delta$NSE")
    ax2.axhline(0, color="#E07B00", lw=0.6, ls=":", alpha=0.5, zorder=1)
    ax2.set_ylabel(r"$\Delta$NSE (Hybrid $-$ GEV-NN)",
                   color="#E07B00", fontsize=8.0)
    ax2.tick_params(axis="y", colors="#E07B00", labelsize=7.5,
                    direction="in", length=3.5, width=0.7)
    ax2.spines["right"].set_color("#E07B00")
    ax2.spines["right"].set_linewidth(0.8)
    ax2.spines["top"].set_visible(False)

    # Pad secondary axis so zero line sits at a sensible place
    d_pad = max(abs(delta[np.isfinite(delta)]).max() * 0.5, 0.05)
    ax2.set_ylim(-d_pad * 0.4, delta[np.isfinite(delta)].max() + d_pad)

    # x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rp_x])
    ax.set_xlabel("Return period (yr)")
    ax.set_ylabel("NSE  (clean stations only)")

    # y-axis
    all_nse = np.concatenate([nse_base, nse_hyb])
    y_lo = max(0.0, np.nanmin(all_nse) - 0.05)
    y_hi = min(1.0, np.nanmax(all_nse) + 0.08)
    ax.set_ylim(y_lo, y_hi)

    _light_grid(ax, axis="y")
    _despine(ax)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right",
              handlelength=1.1, labelspacing=0.35, borderpad=0.4,
              ncol=1)




# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UNIFIED FIGURE  —  1 + 3 panel layout                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_fig(rp: int, t: pd.DataFrame, wb_test: dict,
              drp: pd.DataFrame, t_scatter: pd.DataFrame):
    """
    Left-right composite figure:
      Left  column : Panel a — GHM leakage stacked bar (full height)
      Right column : Panels b / c / d stacked vertically
        b — Layer 1: within-basin dumbbell (ΔNSE_cont vs ΔNSE_clean)
        c — Layer 2: clean-only NSE across return periods
        d — Layer 3: dose-response scatter (leakage % vs ΔNSE)
    """
    _apply_style()

    all_basins = t["basin"].tolist()
    short_names = _make_short_names(all_basins)

    fig = plt.figure(figsize=(14, 10))
    outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.0, 1.0],
        wspace=0.32,
        left=0.07, right=0.97, top=0.96, bottom=0.10)

    # ── Left: Panel a — full-height leakage stacked bar ───────────────────
    ax_a = fig.add_subplot(outer[0])
    _draw_leakage(ax_a, t, short_names)
    _lbl(ax_a, "a", x=-0.06)

    # ── Right: panels b / c / d stacked vertically ────────────────────────
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[1],
        height_ratios=[1.30, 0.80, 1.05],
        hspace=0.52)
    ax_b = fig.add_subplot(gs_right[0])
    ax_c = fig.add_subplot(gs_right[1])
    ax_d = fig.add_subplot(gs_right[2])

    _draw_within_basin_cmp(ax_b, wb_test, t, f"T = {rp} yr",
                           short_names=short_names, show_ylabels=True)
    _lbl(ax_b, "b", x=-0.10)

    _draw_clean_perf(ax_c, drp)
    _lbl(ax_c, "c", x=-0.10)

    _draw_scatter(ax_d, t_scatter, f"T = {rp} yr", short_names=short_names)
    _lbl(ax_d, "d", x=-0.10)

    return fig


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    LOG.info("=== %s ===", TAG)
    PUR_LABELS, base_df, hybrid_df = load_all()

    t2   = compute_basin_metrics(PUR_LABELS, base_df, hybrid_df, rp=2)
    t100 = compute_basin_metrics(PUR_LABELS, base_df, hybrid_df, rp=100)
    drp  = compute_delta_rp(base_df, hybrid_df)

    wb_all = {}
    for rp in RETURN_PERIODS:
        wb_all[rp] = compute_within_basin_leakage_test(base_df, hybrid_df, rp)
        res = wb_all[rp]
        LOG.info("  Within-basin T=%d: n_basins=%d, mean_δ=%.3f, Wilcoxon p=%.3f",
                 rp, res["n_basins"],
                 res["mean_delta"] if (res["mean_delta"] is not None
                                       and np.isfinite(res["mean_delta"]))
                 else np.nan,
                 res["wilcoxon_p"] if (isinstance(res["wilcoxon_p"], float)
                                       and np.isfinite(res["wilcoxon_p"]))
                 else np.nan)
        if not res["per_basin"].empty:
            res["per_basin"].to_csv(
                OUT_DATA / f"within_basin_test_T{rp}.csv", index=False)

    LOG.info("\nBasin summary T=2:")
    for _, r in t2.iterrows():
        LOG.info("  %-25s  leak=%5.1f%%  NSE %.3f->%.3f  PBIAS %.1f->%.1f  rRMSE %.1f->%.1f",
                 r["basin"], r["leakage_pct"],
                 r["NSE_base"], r["NSE_hybrid"],
                 r["PBIAS_base"], r["PBIAS_hybrid"],
                 r["rRMSE_base"], r["rRMSE_hybrid"])
    LOG.info("\nBasin summary T=100:")
    for _, r in t100.iterrows():
        LOG.info("  %-25s  leak=%5.1f%%  NSE %.3f->%.3f  PBIAS %.1f->%.1f  rRMSE %.1f->%.1f",
                 r["basin"], r["leakage_pct"],
                 r["NSE_base"], r["NSE_hybrid"],
                 r["PBIAS_base"], r["PBIAS_hybrid"],
                 r["rRMSE_base"], r["rRMSE_hybrid"])

    t2["return_period"]   = 2
    t100["return_period"] = 100
    pd.concat([t2, t100], ignore_index=True).to_csv(
        OUT_DATA / "basin_summary_T2_T100.csv", index=False)

    # Delete old three-figure output files
    for stem in ["fig1_T2", "fig2_T100", "fig3_evidence"]:
        for ext in ("svg", "png"):
            p = OUT_FIG / f"{TAG}_{stem}.{ext}"
            if p.exists():
                p.unlink()
                LOG.info("Deleted old figure: %s", p.name)

    # Build two composite figures (T=2 and T=100)
    fig_t2 = build_fig(2, t2, wb_all[2], drp, t_scatter=t2)
    _save(fig_t2, OUT_FIG / f"{TAG}_fig_T2")

    fig_t100 = build_fig(100, t100, wb_all[100], drp, t_scatter=t100)
    _save(fig_t100, OUT_FIG / f"{TAG}_fig_T100")

    LOG.info("Done -> %s", OUT_FIG)


if __name__ == "__main__":
    main()
