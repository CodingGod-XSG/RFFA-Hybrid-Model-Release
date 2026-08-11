from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import importlib
import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kstest, genextreme
from scipy.special import gamma as sp_gamma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl


@dataclass
class GEVFitConfig:
    nc_file: Path
    out_csv: Path
    out_fig: Path
    return_periods: List[int]
    min_years: int
    n_sample: int
    seed: int
    xi_clip_min: float
    xi_clip_max: float
    method: str = "mle"  # "mle" or "lmom"
    csv_name: str = "gev_station_params.csv"
    normalize_to_mm_day: bool = False
    area_xlsx: Optional[Path] = None
    unit_scale_factor: float = 86.4


def infer_base_dir(script_file: str | Path) -> Path:
    script = Path(script_file).resolve()
    for p in script.parents:
        if (p / "code").exists() and (p / "data").exists():
            return p
    return script.parents[1]


def load_base_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        yaml_mod = importlib.import_module("yaml")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to read configs/base.yaml. Install with: pip install pyyaml"
        ) from exc
    with config_path.open("r", encoding="utf-8") as f:
        return yaml_mod.safe_load(f) or {}


def _resolve_nc_file_path(nc_file: Path, base_cfg: dict, section: str) -> Path:
    if nc_file.exists():
        return nc_file

    outputs = base_cfg.get("outputs", {})
    paths_cfg = base_cfg.get("paths", {})
    stats_cfg = base_cfg.get("station_statistics", {})
    preprocess_tag = str(stats_cfg.get("preprocess_tag", "00_GRDC-Caravan-Process")).strip() or "00_GRDC-Caravan-Process"

    candidates: List[Path] = [nc_file, nc_file.parent / preprocess_tag / nc_file.name]

    # For simulation section, avoid silently falling back to observed proceed outputs.
    if section == "sim_gev_fit":
        streamflow_sec = base_cfg.get("streamflow_process", {})
        sim_nc = str(streamflow_sec.get("nc_file", "")).strip()
        if sim_nc:
            candidates.append(Path(sim_nc))

        raw_sim_dir = Path(paths_cfg.get("project_root", "")) / "data" / "raw" / "Sim-Dis"
        if raw_sim_dir.exists():
            candidates.extend(sorted(raw_sim_dir.glob("*.nc")))

        for c in candidates:
            if c.exists():
                return c

        checked = "\n  - " + "\n  - ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Cannot locate simulation NC file for section '{section}'. Tried:{checked}"
        )

    for key in ["nc_feat_35", "nc_feat_alias", "nc_35", "nc_dedup", "nc_merge"]:
        val = str(outputs.get(key, "")).strip()
        if not val:
            continue
        p = Path(val)
        candidates.append(p)
        candidates.append(p.parent / preprocess_tag / p.name)

    proceed_dir_s = str(paths_cfg.get("proceed_dir", "")).strip()
    if proceed_dir_s:
        proceed_dir = Path(proceed_dir_s)
        if proceed_dir.exists():
            for name in [
                nc_file.name,
                "4_Cara-GRDC-35.nc",
                "4_Cara-GRDC.nc",
                "3_Cara-GRDC-35.nc",
            ]:
                hits = sorted(proceed_dir.rglob(name))
                if hits:
                    candidates.extend(hits[:3])

    seen = set()
    uniq = []
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            uniq.append(c)

    for c in uniq:
        if c.exists():
            return c

    checked = "\n  - " + "\n  - ".join(str(c) for c in uniq[:12])
    raise FileNotFoundError(
        f"Cannot locate GEV input NC file. Tried:{checked}"
    )


def build_gev_fit_config(base_dir: Path, base_cfg: dict, section: str) -> GEVFitConfig:
    sec = base_cfg.get(section, {})
    raw_nc = Path(sec.get("nc_file", base_dir / "data" / "proceed" / "Caravan-GRDC" / "4_Cara-GRDC-35.nc"))
    nc_file = _resolve_nc_file_path(raw_nc, base_cfg, section=section)
    xi_clip_min = float(sec.get("xi_clip_min", -0.5))
    xi_clip_max = float(sec.get("xi_clip_max", 0.5))
    if xi_clip_min >= xi_clip_max:
        raise ValueError(f"Invalid xi clip range: [{xi_clip_min}, {xi_clip_max}]. Require xi_clip_min < xi_clip_max")

    if section == "sim_gev_fit":
        default_out_csv = base_dir / "data" / "proceed" / "Caravan-GRDC" / "04_Sim_GEV-Fit"
        default_out_fig = base_dir / "figures" / "Caravan-GRDC" / "04_Sim_GEV-Fit"
        default_area_xlsx = base_dir / "data" / "proceed" / "Caravan-GRDC" / "station_locations.xlsx"
    else:
        default_out_csv = base_dir / "data" / "proceed" / "Caravan-GRDC" / "01_GEV-Fit"
        default_out_fig = base_dir / "figures" / "Caravan-GRDC" / "01_GEV-Fit"
        default_area_xlsx = None

    return GEVFitConfig(
        nc_file=nc_file,
        out_csv=Path(sec.get("out_csv_dir", default_out_csv)),
        out_fig=Path(sec.get("out_fig_dir", default_out_fig)),
        return_periods=[int(x) for x in sec.get("return_periods", [2, 5, 10, 20, 50, 100])],
        min_years=int(sec.get("min_years", 5)),
        n_sample=int(sec.get("n_sample", 24)),
        seed=int(sec.get("seed", 42)),
        xi_clip_min=xi_clip_min,
        xi_clip_max=xi_clip_max,
        method=str(sec.get("method", "mle")),
        csv_name=str(sec.get("csv_name", "gev_station_params.csv")),
        normalize_to_mm_day=bool(sec.get("normalize_to_mm_day", False)),
        area_xlsx=Path(sec.get("area_xlsx", default_area_xlsx)) if default_area_xlsx is not None else None,
        unit_scale_factor=float(sec.get("unit_scale_factor", 86.4)),
    )


def ensure_output_dirs(cfg: GEVFitConfig) -> None:
    for d in [cfg.out_csv, cfg.out_fig]:
        d.mkdir(parents=True, exist_ok=True)


def configure_plot_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def lmom_gev(data: np.ndarray, xi_clip_min: float, xi_clip_max: float) -> Tuple[float, float, float]:
    data = np.sort(data[np.isfinite(data) & (data > 0)])
    n = len(data)
    if n < 4:
        raise ValueError(f"Too few points for L-moments: {n}")

    idx = np.arange(n, dtype=float)
    b0 = np.mean(data)
    b1 = float(np.sum(idx[1:] / (n - 1) * data[1:])) / n
    b2 = float(np.sum(idx[2:] * (idx[2:] - 1) / ((n - 1) * (n - 2)) * data[2:])) / n

    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    if l2 <= 0:
        raise ValueError("L2 <= 0 - degenerate series")
    t3 = l3 / l2

    c = 2.0 / (3.0 + t3) - np.log(2.0) / np.log(3.0)
    xi = 7.8590 * c + 2.9554 * c ** 2
    xi = float(np.clip(xi, xi_clip_min, xi_clip_max))

    g1 = sp_gamma(1.0 + xi)
    if abs(xi) < 1e-6 or abs(1.0 - 2.0 ** (-xi)) < 1e-10:
        sigma = l2 / np.log(2.0)
        mu = l1 - sigma * 0.5772156649
        return float(mu), float(abs(sigma)), 0.0

    sigma = l2 * xi / ((1.0 - 2.0 ** (-xi)) * g1)
    mu = l1 - sigma * (1.0 - g1) / xi
    return float(mu), float(abs(sigma)), float(xi)


def mle_gev(data: np.ndarray, min_years: int, xi_clip_min: float, xi_clip_max: float) -> Tuple[float, float, float]:
    data = data[np.isfinite(data) & (data > 0)]
    if len(data) < min_years:
        raise ValueError(f"Too few valid years: {len(data)}")

    try:
        mu0, sigma0, xi0 = lmom_gev(data, xi_clip_min, xi_clip_max)
    except Exception:
        mu0, sigma0, xi0 = float(np.mean(data)), float(np.std(data) + 1e-9), 0.1

    try:
        c_fit, loc_fit, scale_fit = genextreme.fit(
            data, -xi0, loc=mu0, scale=sigma0, method="MLE"
        )
        mu = float(loc_fit)
        sigma = float(abs(scale_fit))
        xi = float(np.clip(-c_fit, xi_clip_min, xi_clip_max))
    except Exception:
        mu, sigma, xi = mu0, sigma0, xi0

    return mu, sigma, xi


def gev_quantile(mu: float, sigma: float, xi: float, T: int) -> float:
    p = 1.0 - 1.0 / T
    if abs(xi) < 1e-6:
        return mu + sigma * (-np.log(-np.log(p)))
    return mu + sigma / xi * ((-np.log(p)) ** (-xi) - 1.0)


def gev_cdf(x: np.ndarray, mu: float, sigma: float, xi: float) -> np.ndarray:
    t = (x - mu) / sigma
    if abs(xi) < 1e-6:
        return np.exp(-np.exp(-t))
    z = 1.0 + xi * t
    z = np.clip(z, 1e-10, None)
    return np.exp(-(z ** (-1.0 / xi)))


def ks_gev(data: np.ndarray, mu: float, sigma: float, xi: float) -> float:
    _, pval = kstest(data, lambda x: gev_cdf(x, mu, sigma, xi))
    return float(pval)


def _normalize_sim_streamflow_to_mm_day(
    sf: xr.DataArray,
    station_ids: np.ndarray,
    station_dim: str,
    cfg: GEVFitConfig,
    log: logging.Logger | None = None,
) -> xr.DataArray:
    def _resolve_area_xlsx_path(raw_path: Optional[Path]) -> Optional[Path]:
        if raw_path is None:
            return None
        if raw_path.exists():
            return raw_path

        # Common relocation: .../Caravan-GRDC/station_locations.xlsx -> .../Caravan-GRDC/03_Streamflow-Process/station_locations.xlsx
        candidates = [
            raw_path,
            raw_path.parent / "03_Streamflow-Process" / raw_path.name,
        ]

        try:
            proceed_dir = cfg.out_csv.parent
            candidates.extend(
                [
                    proceed_dir / raw_path.name,
                    proceed_dir / "03_Streamflow-Process" / raw_path.name,
                    proceed_dir / "station_locations.xlsx",
                    proceed_dir / "03_Streamflow-Process" / "station_locations.xlsx",
                ]
            )
        except Exception:
            pass

        seen = set()
        uniq = []
        for c in candidates:
            s = str(c)
            if s not in seen:
                seen.add(s)
                uniq.append(c)
        for c in uniq:
            if c.exists():
                return c

        # Last resort: scan proceed directory for station_locations.xlsx.
        try:
            proceed_dir = cfg.out_csv.parent
            hits = list(proceed_dir.rglob("station_locations.xlsx"))
            if hits:
                return hits[0]
        except Exception:
            pass
        return raw_path

    if cfg.area_xlsx is None:
        raise ValueError("normalize_to_mm_day=True requires area_xlsx in config")
    area_xlsx = _resolve_area_xlsx_path(cfg.area_xlsx)
    if area_xlsx is None or not area_xlsx.exists():
        raise FileNotFoundError(f"Area xlsx not found: {cfg.area_xlsx}")
    if area_xlsx != cfg.area_xlsx and log is not None:
        log.warning("  area_xlsx not found at configured path; using fallback: %s", area_xlsx)

    area_df = pd.read_excel(area_xlsx, usecols=["station_id", "area_km2"])
    area_df["station_id"] = area_df["station_id"].astype(str).str.strip()
    area_map = dict(zip(area_df["station_id"], area_df["area_km2"]))

    stn_str = np.array([str(s).strip() for s in station_ids], dtype=object)
    areas = np.array([area_map.get(s, np.nan) for s in stn_str], dtype=float)
    valid = np.isfinite(areas) & (areas > 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError(
            "No station IDs matched valid area_km2 records; cannot normalize simulated flow"
        )

    if log is not None:
        log.info(
            "  Normalizing simulated streamflow to mm/day with %d/%d valid areas (factor=%.3f/area_km2)",
            n_valid,
            len(station_ids),
            cfg.unit_scale_factor,
        )

    scale = np.where(valid, cfg.unit_scale_factor / areas, np.nan)
    scale_da = xr.DataArray(scale, coords={station_dim: sf[station_dim]}, dims=[station_dim])
    return sf * scale_da


def _load_ams_and_stations(ds: xr.Dataset, cfg: GEVFitConfig, log: logging.Logger | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load AMS matrix and station IDs from a dataset.

    Supported inputs:
    1) `ann_max_streamflow` already present (station x year).
    2) Daily `streamflow` with `time` coordinate (station/merit x time),
       converted to annual maxima on the fly.
    """
    if "ann_max_streamflow" in ds.variables:
        da = ds["ann_max_streamflow"]
        if len(da.dims) != 2:
            raise ValueError("ann_max_streamflow must be 2-D (station_dim, year)")
        station_dim = da.dims[0]
        station_ids = ds.coords[station_dim].values if station_dim in ds.coords else np.arange(da.shape[0])
        return da.values, station_ids

    if "streamflow" in ds.variables and "time" in ds.coords:
        sf = ds["streamflow"]
        if "time" not in sf.dims:
            raise ValueError("streamflow must include 'time' dimension")

        station_dims = [d for d in sf.dims if d != "time"]
        if len(station_dims) != 1:
            raise ValueError(f"Cannot infer station dimension from streamflow dims: {sf.dims}")
        station_dim = station_dims[0]
        station_ids = ds.coords[station_dim].values if station_dim in ds.coords else np.arange(sf.shape[0])

        if cfg.normalize_to_mm_day:
            sf = _normalize_sim_streamflow_to_mm_day(sf, station_ids, station_dim, cfg, log=log)

        sf_pos = sf.where(sf > 0)
        ams_da = sf_pos.groupby("time.year").max("time", skipna=True)
        ams_da = ams_da.transpose(station_dim, "year")

        if log is not None:
            log.info("  ann_max_streamflow not found; built AMS from daily streamflow via yearly maxima")

        return ams_da.values, station_ids

    raise KeyError(
        "No variable named 'ann_max_streamflow', and cannot derive AMS because "
        "'streamflow' + 'time' are not both available."
    )


def fit_all_stations(cfg: GEVFitConfig, log: logging.Logger) -> pd.DataFrame:
    log.info(f"Reading {cfg.nc_file.name} ...")
    log.info(f"  Xi clip range: [{cfg.xi_clip_min}, {cfg.xi_clip_max}]")
    if cfg.normalize_to_mm_day:
        log.info("  Sim flow normalization: ENABLED (to mm/day)")
        log.info(f"  Area table: {cfg.area_xlsx}")
    else:
        log.info("  Sim flow normalization: DISABLED")
    ds = xr.open_dataset(cfg.nc_file)
    amf, stn = _load_ams_and_stations(ds, cfg, log=log)

    def _get_var(name):
        if name in ds:
            return ds[name].values
        return np.full(len(stn), np.nan)

    lat  = _get_var("static_gauge_lat")
    lon  = _get_var("static_gauge_lon")
    area = _get_var("static_area")
    ds.close()

    n_st = amf.shape[0]
    log.info(f"  Stations: {n_st}  |  Years: {amf.shape[1]}")

    records = []
    for i in range(n_st):
        series = amf[i, :]
        series = series[np.isfinite(series) & (series > 0)]
        n_valid = len(series)
        rec = {
            "station_id": stn[i],
            "lat": float(lat[i]),
            "lon": float(lon[i]),
            "area_km2": float(area[i]),
            "n_years": n_valid,
            "mu": np.nan,
            "sigma": np.nan,
            "xi": np.nan,
            "fit_ok": False,
            "ks_pvalue": np.nan,
        }
        for T in cfg.return_periods:
            rec[f"Q{T}"] = np.nan

        if n_valid < cfg.min_years:
            records.append(rec)
            continue

        try:
            if cfg.method == "lmom":
                mu, sigma, xi = lmom_gev(series, cfg.xi_clip_min, cfg.xi_clip_max)
            else:
                mu, sigma, xi = mle_gev(series, cfg.min_years, cfg.xi_clip_min, cfg.xi_clip_max)
            rec.update({"mu": mu, "sigma": sigma, "xi": xi, "fit_ok": True})
            rec["ks_pvalue"] = ks_gev(series, mu, sigma, xi)
            for T in cfg.return_periods:
                rec[f"Q{T}"] = gev_quantile(mu, sigma, xi, T)
        except Exception:
            rec["fit_ok"] = False

        records.append(rec)

    df = pd.DataFrame(records)
    ok = df["fit_ok"].sum()
    log.info(f"  Fit OK: {ok}/{n_st}  ({100 * ok / n_st:.1f}%)")
    return df


def plot_freq_curves(df: pd.DataFrame, cfg: GEVFitConfig, log: logging.Logger) -> None:
    log.info("  Plotting frequency curves ...")
    ds = xr.open_dataset(cfg.nc_file)
    amf, stn = _load_ams_and_stations(ds, cfg, log=log)
    ds.close()

    good_idx = df.index[df["fit_ok"]].tolist()
    if len(good_idx) == 0:
        log.warning("  No fitted stations - skipping frequency curves.")
        return

    rng = np.random.default_rng(cfg.seed)
    sample = rng.choice(good_idx, size=min(cfg.n_sample, len(good_idx)), replace=False)
    ncols = 4
    nrows = (len(sample) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8), dpi=150)
    axes = np.array(axes).flatten()

    T_plot = np.logspace(np.log10(1.01), np.log10(200), 200)
    T_marks = cfg.return_periods

    for k, idx in enumerate(sample):
        ax = axes[k]
        row = df.loc[idx]
        si = np.where(stn == row["station_id"])[0][0]
        data = amf[si, :]
        data = data[np.isfinite(data) & (data > 0)]
        n = len(data)

        emp_sorted = np.sort(data)
        emp_T = (n + 1) / (np.arange(n, 0, -1))

        mu, sigma, xi = float(row["mu"]), float(row["sigma"]), float(row["xi"])
        q_curve = np.array([gev_quantile(mu, sigma, xi, t) for t in T_plot])

        ax.semilogx(T_plot, q_curve, color="#1565C0", lw=1.8, label="GEV fit")
        ax.scatter(emp_T, emp_sorted, s=18, color="#C62828", zorder=5, alpha=0.7, label="Empirical")

        for T in T_marks:
            q = gev_quantile(mu, sigma, xi, T)
            ax.axvline(T, color="gray", lw=0.6, ls=":", alpha=0.5)
            ax.plot(T, q, "^", color="#2E7D32", ms=5, zorder=6)

        ax.set_xlabel("Return period (yr)", fontsize=8)
        ax.set_ylabel("Flow (m3/s)", fontsize=8)
        ax.set_title(f"{str(row['station_id'])[:14]}  n={n}  xi={xi:.3f}", fontsize=7.5, fontweight="bold")
        ax.tick_params(labelsize=6.5)
        if k == 0:
            ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.25)

    for j in range(len(sample), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"GEV Frequency Curves - {len(sample)} random stations", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(cfg.out_fig / "freq_curves_sample.png", dpi=300)
    plt.close(fig)
    log.info(f"  Saved: {cfg.out_fig / 'freq_curves_sample.png'}")


def plot_param_distributions(df: pd.DataFrame, cfg: GEVFitConfig, log: logging.Logger) -> None:
    log.info("  Plotting parameter distributions ...")
    fitted = df[df["fit_ok"]]
    if fitted.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=200)
    param_info = [
        ("mu", r"$\mu$ (location, mm/day)", "#1565C0"),
        ("sigma", r"$\sigma$ (scale, mm/day)", "#C62828"),
        ("xi", r"$\xi$ (shape)", "#2E7D32"),
    ]
    for ax, (col, label, color) in zip(axes, param_info):
        vals = fitted[col].dropna()
        if col == "xi":
            # For shape parameter, keep full configured clip range visible.
            clipped = vals
            p1, p99 = cfg.xi_clip_min, cfg.xi_clip_max
        else:
            p1, p99 = np.percentile(vals, [1, 99])
            clipped = vals[(vals >= p1) & (vals <= p99)]

        ax.hist(clipped, bins=60, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(clipped.median(), color="k", lw=1.5, ls="--", label=f"Median={clipped.median():.3f}")

        if col == "xi":
            n_min = int(np.sum(np.isclose(vals.values, cfg.xi_clip_min)))
            n_max = int(np.sum(np.isclose(vals.values, cfg.xi_clip_max)))
            ax.axvline(cfg.xi_clip_min, color="#616161", lw=1.0, ls=":", alpha=0.9)
            ax.axvline(cfg.xi_clip_max, color="#616161", lw=1.0, ls=":", alpha=0.9)
            ax.text(
                0.02,
                0.95,
                f"clip=[{cfg.xi_clip_min:.2f}, {cfg.xi_clip_max:.2f}]\n"
                f"at_min={n_min}, at_max={n_max}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )
            ax.set_xlim(cfg.xi_clip_min - 0.05, cfg.xi_clip_max + 0.05)

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"Distribution of {label}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"GEV Parameter Distributions  (n={len(fitted)} stations)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(cfg.out_fig / "gev_param_distributions.png", dpi=300)
    plt.close(fig)
    log.info(f"  Saved: {cfg.out_fig / 'gev_param_distributions.png'}")


def plot_ks_distribution(df: pd.DataFrame, cfg: GEVFitConfig, log: logging.Logger) -> None:
    log.info("  Plotting K-S p-value distribution ...")
    fitted = df[df["fit_ok"] & df["ks_pvalue"].notna()]
    if fitted.empty:
        return

    pvals = fitted["ks_pvalue"]
    frac_ok = (pvals >= 0.05).mean()

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    ax.hist(pvals, bins=40, color="#7B1FA2", alpha=0.75, edgecolor="none")
    ax.axvline(0.05, color="red", lw=1.5, ls="--", label=f"p=0.05  ({100 * frac_ok:.1f}% >= 0.05)")
    ax.set_xlabel("K-S test p-value (GEV fit)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Goodness-of-fit: K-S test p-values across all stations", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.out_fig / "ks_pvalue_distribution.png", dpi=300)
    plt.close(fig)
    log.info(f"  Saved: {cfg.out_fig / 'ks_pvalue_distribution.png'}")
