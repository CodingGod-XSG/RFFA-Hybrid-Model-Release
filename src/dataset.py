# -*- coding: utf-8 -*-
"""
4_12_base/dataset.py
DatasetBuilder: NC + GEV CSV + Flow CSV  →  data dict

Returns
-------
dict with keys:
    X_base     (N, 36)  standardised static features (NOT yet scaled)
    X_flow     (N, 12)  standardised flow features   (NOT yet scaled)
    X_full     (N, 48)  concatenation of above
    q_true     (N, 6)   GEV quantiles [Q2…Q100] in m³/s
    ams        (N, T_max) annual-max timeseries (padded/masked)
    ams_mask   (N, T_max) bool mask
    gev_params (N, 3)   [mu, sigma, xi]
    lat        (N,)     float64
    lon        (N,)     float64
    stations   (N,)     str
    feat_base  list of base feature names (len=actual loaded)
    feat_flow  list of flow feature names (len=actual loaded)
"""
from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr

LOG = logging.getLogger(__name__)

# ── canonical column lists ──────────────────────────────────────────────────
FEATURE_COLS = [
    "static_p_mean", "static_aridity_FAO_PM", "static_seasonality_FAO_PM",
    "static_frac_snow", "static_hft_ix_s93", "static_area",
    "static_ele_mt_sav", "static_slp_dg_sav", "static_soc_th_sav",
    "static_inu_pc_smx", "static_cly_pc_sav", "static_snd_pc_sav",
    "static_for_pc_sse", "static_crp_pc_sse", "static_dor_pc_pva",
    "static_ppd_pk_sav", "static_urb_pc_sse",
    "static_high_prec_freq", "static_high_prec_dur",
    "static_low_prec_freq",  "static_low_prec_dur",
    "static_sgr_dk_sav", "static_gwt_cm_sav", "static_snw_pc_syr",
    "static_lka_pc_sse", "static_run_mm_syr", "static_cmi_ix_syr",
    "static_ele_mt_smn", "static_inu_pc_slt", "static_wet_pc_s09",
    "prec_roll3d_ann_max_mean", "prec_roll5d_ann_max_mean",
    "prec_roll7d_ann_max_mean", "prec_roll30d_ann_max_mean",
    "temp_roll7d_ann_max", "temp_roll30d_ann_max",
]
LOG_TRANSFORM_BASE = {"static_area"}

FLOW_COLS = [
    "ams_mean", "ams_std", "ams_cv",
    "roll3d_ann_max_mean", "roll5d_ann_max_mean",
    "roll7d_ann_max_mean", "roll30d_ann_max_mean",
    "flow_mean", "flow_std", "flow_cv",
    "high_flow_freq", "seasonality_ratio",
]
LOG_TRANSFORM_FLOW = {
    "ams_mean", "ams_std",
    "roll3d_ann_max_mean", "roll5d_ann_max_mean",
    "roll7d_ann_max_mean", "roll30d_ann_max_mean",
    "flow_mean", "flow_std",
}

RETURN_PERIODS = [2, 5, 10, 20, 50, 100]
AMS_VAR = "ann_max_streamflow"
LAT_VAR = "static_gauge_lat"
LON_VAR = "static_gauge_lon"


def _impute_nan(X: np.ndarray) -> np.ndarray:
    for j in range(X.shape[1]):
        m = ~np.isfinite(X[:, j])
        if m.any():
            col_med = np.nanmedian(X[:, j])
            if not np.isfinite(col_med):
                col_med = 0.0
            X[m, j] = col_med
    return X


class DatasetBuilder:
    def __init__(self, nc_path: Path, gev_csv: Path, flow_csv: Path):
        self.nc_path  = Path(nc_path)
        self.gev_csv  = Path(gev_csv)
        self.flow_csv = Path(flow_csv)

    def build(self) -> dict:
        LOG.info("DatasetBuilder: loading inputs …")

        # ── NC ──────────────────────────────────────────────────────────────
        ds = xr.open_dataset(self.nc_path, engine="netcdf4")
        stations_nc = ds.coords["station"].values.astype(str)

        # ── GEV CSV ─────────────────────────────────────────────────────────
        gev_df = pd.read_csv(self.gev_csv)
        gev_df = gev_df[gev_df["fit_ok"] == True].copy()
        gev_df.set_index("station_id", inplace=True)
        gev_df.index = gev_df.index.astype(str).str.strip()

        # ── Flow features CSV ────────────────────────────────────────────────
        if not self.flow_csv.exists():
            raise FileNotFoundError(f"Flow CSV not found: {self.flow_csv}")
        flow_df = pd.read_csv(self.flow_csv).set_index("station_id")
        flow_df.index = flow_df.index.astype(str).str.strip()

        # ── Intersection ─────────────────────────────────────────────────────
        common = sorted(set(stations_nc) & set(gev_df.index) & set(flow_df.index))
        LOG.info(f"  NC:{len(stations_nc)}  GEV:{len(gev_df)}  "
                 f"Flow:{len(flow_df)}  common:{len(common)}")
        if len(common) == 0:
            ds.close()
            raise ValueError(
                "No overlapping station_id across NC, GEV CSV and flow CSV. "
                "Check station_id format/dtype in inputs."
            )

        stn_to_i = {s: i for i, s in enumerate(stations_nc)}
        nc_idx   = np.array([stn_to_i[s] for s in common])

        # ── Static features ──────────────────────────────────────────────────
        avail_base = [f for f in FEATURE_COLS if f in ds]
        X_base = np.column_stack(
            [ds[f].values[nc_idx] for f in avail_base]
        ).astype(np.float32)
        for j, f in enumerate(avail_base):
            if f in LOG_TRANSFORM_BASE:
                X_base[:, j] = np.log1p(np.abs(X_base[:, j]))
        X_base = _impute_nan(X_base)

        # ── Flow features ────────────────────────────────────────────────────
        avail_flow = [c for c in FLOW_COLS if c in flow_df.columns]
        X_flow = flow_df.loc[common, avail_flow].values.astype(np.float32)
        for j, f in enumerate(avail_flow):
            if f in LOG_TRANSFORM_FLOW:
                X_flow[:, j] = np.log1p(np.abs(X_flow[:, j]))
        X_flow = _impute_nan(X_flow)

        LOG.info(f"  base features:{X_base.shape[1]}  flow features:{X_flow.shape[1]}")

        # ── Labels & AMS ─────────────────────────────────────────────────────
        ams_all  = ds[AMS_VAR].values[nc_idx]           # (N, T_max)
        ams_mask = np.isfinite(ams_all)
        ams_all  = np.nan_to_num(ams_all, nan=0.0).astype(np.float32)

        gev_sub    = gev_df.loc[common]
        q_true     = gev_sub[[f"Q{T}" for T in RETURN_PERIODS]].values.astype(np.float32)
        gev_params = gev_sub[["mu", "sigma", "xi"]].values.astype(np.float32)
        lat = ds[LAT_VAR].values[nc_idx].astype(np.float64)
        lon = ds[LON_VAR].values[nc_idx].astype(np.float64)
        ds.close()

        # ── Quality filter ───────────────────────────────────────────────────
        good = ((gev_params[:, 1] >= 1e-3)
                & (gev_params[:, 0] > 0)
                & np.all(q_true > 0, axis=1)
                & np.all(np.isfinite(q_true), axis=1))
        n_bad = (~good).sum()
        if n_bad:
            LOG.warning(f"  Removing {n_bad} pathological stations")
        ig = np.where(good)[0]

        X_base, X_flow   = X_base[ig],   X_flow[ig]
        q_true           = q_true[ig]
        gev_params       = gev_params[ig]
        ams_all, ams_mask = ams_all[ig], ams_mask[ig]
        lat, lon         = lat[ig], lon[ig]
        common           = [common[i] for i in ig]

        X_full = np.concatenate([X_base, X_flow], axis=1).astype(np.float32)

        LOG.info(f"  Final: {len(common)} stations | "
                 f"X_base={X_base.shape[1]}  X_full={X_full.shape[1]}")

        return dict(
            X_base     = X_base,
            X_flow     = X_flow,
            X_full     = X_full,
            q_true     = q_true,
            ams        = ams_all,
            ams_mask   = ams_mask,
            gev_params = gev_params,
            lat        = lat,
            lon        = lon,
            stations   = np.array(common),
            feat_base  = avail_base,
            feat_flow  = avail_flow,
        )
