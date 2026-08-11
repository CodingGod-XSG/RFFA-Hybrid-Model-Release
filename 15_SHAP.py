# -*- coding: utf-8 -*-
"""
15_SHAP.py
Compute and cache SHAP outputs for PUB models.

This script only does computation and persistence.
No figures are generated here.

Outputs
-------
data/proceed/Caravan-GRDC/15_SHAP/
  shap_cache_{model}_{flow}.npz
  shap_importance_{model}_{flow}.csv
  shap_cache_index.csv
  log.txt
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import torch

warnings.filterwarnings("ignore")

# ------------------------------- Paths/config -------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

from src.paths import DATA_PROCEED, MODEL_ROOT, stage_dir

PROC = DATA_PROCEED

MODEL_DIRS = {
    "RF": MODEL_ROOT / "06_RF",
    "ANN": MODEL_ROOT / "09_ANN",
    "GEV-NN": MODEL_ROOT / "11_GEV_NN",
}

COMPUTE_TAG = "15_SHAP"
OUT_DATA = stage_dir(PROC, COMPUTE_TAG)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DATA / "log.txt", mode="w", encoding="utf-8"),
    ],
)
LOG = logging.getLogger(__name__)

from src.dataset import DatasetBuilder
from src.models import ANNSingle, GEVNNSingleTower
from src.splits import pub_split

# --------------------------------- Config ----------------------------------
NC_PATH = PROC / "02_Data-Clean" / "4_Cara-GRDC-35_cleaned.nc"
GEV_CSV = PROC / "01_GEV-Fit" / "gev_station_params.csv"
FLOW_CSV = PROC / "03_Streamflow-Process" / "sim_flow_features_per_station.csv"

SEED = 42
DEVICE = torch.device("cpu")
N_SHAP_BG = 200
N_SHAP_TEST = 500
RETURN_PERIODS = [2, 5, 10, 20, 50, 100]


# ------------------------------- SHAP helpers -------------------------------
def _flow_tag(use_flow: bool) -> str:
    return "+flow" if use_flow else "base"


def _flow_key(use_flow: bool) -> str:
    return "flow" if use_flow else "base"


def _load_scaler(model_dir: Path, flow_tag: str, seed: int):
    candidates = list(model_dir.glob(f"scaler_PUB_*_{flow_tag}_s{seed}.joblib"))
    if not candidates:
        candidates = list(model_dir.glob(f"scaler_PUB_*_{flow_tag}*.joblib"))
    if not candidates:
        raise FileNotFoundError(
            f"No scaler found in {model_dir} for flow={flow_tag} seed={seed}"
        )
    return joblib.load(candidates[0])


def _normalize_shap_matrix(shap_values, n_samples: int, n_features: int, context: str) -> np.ndarray:
    arr = np.asarray(shap_values)

    if isinstance(shap_values, list):
        arr = np.stack([np.asarray(v) for v in shap_values], axis=0)

    while arr.ndim > 2:
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            arr = arr.mean(axis=2)
            continue
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            arr = arr.mean(axis=0)
            continue
        if arr.shape[0] == n_samples and arr.shape[2] == n_features:
            arr = arr.mean(axis=1)
            continue

        reduce_axis = next(
            (ax for ax, dim in enumerate(arr.shape) if dim not in (n_samples, n_features)),
            0,
        )
        arr = arr.mean(axis=reduce_axis)

    if arr.ndim != 2:
        raise ValueError(f"{context}: unexpected SHAP ndim={arr.ndim}, shape={arr.shape}")

    if arr.shape[0] == n_features and arr.shape[1] == n_samples:
        arr = arr.T

    if arr.shape[0] != n_samples or arr.shape[1] != n_features:
        raise ValueError(
            f"{context}: cannot align SHAP shape {arr.shape} to expected "
            f"(N={n_samples}, F={n_features})"
        )

    return arr.astype(np.float32, copy=False)


def shap_rf(model_dir: Path, data: dict, tr_idx, te_idx, use_flow: bool):
    flow_tag = _flow_tag(use_flow)
    model_paths = sorted(model_dir.glob(f"RF_PUB_RF_{flow_tag}_s{SEED}.joblib"))
    if not model_paths:
        model_paths = sorted(model_dir.glob(f"RF_PUB_RF_{flow_tag}*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"RF model not found in {model_dir}")

    rfs = joblib.load(model_paths[0])
    scaler = _load_scaler(model_dir, flow_tag, SEED)

    X_all = data["X_full"] if use_flow else data["X_base"]
    X_te = scaler.transform(X_all[te_idx])

    rng = np.random.RandomState(42)
    te_sub = rng.choice(len(te_idx), size=min(N_SHAP_TEST, len(te_idx)), replace=False)
    X_sub = X_te[te_sub]

    sv_list = []
    for i, rf in enumerate(rfs):
        explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
        sv = explainer.shap_values(X_sub)
        sv = _normalize_shap_matrix(
            sv,
            n_samples=X_sub.shape[0],
            n_features=X_sub.shape[1],
            context=f"RF[{i}]",
        )
        sv_list.append(sv)

    sv_mean = np.mean(np.stack(sv_list, axis=0), axis=0)
    return sv_mean, X_sub


def shap_gev_nn(model_dir: Path, data: dict, tr_idx, te_idx, use_flow: bool):
    flow_tag = _flow_tag(use_flow)
    pt_paths = sorted(model_dir.glob(f"PUB_GEV_NN_ST_{flow_tag}_s{SEED}.pt"))
    if not pt_paths:
        pt_paths = sorted(model_dir.glob(f"PUB_GEV_NN_ST_{flow_tag}*.pt"))
    if not pt_paths:
        raise FileNotFoundError(f"GEV-NN model not found in {model_dir}")

    scaler = _load_scaler(model_dir, flow_tag, SEED)
    X_all = data["X_full"] if use_flow else data["X_base"]
    X_tr = scaler.transform(X_all[tr_idx])
    X_te = scaler.transform(X_all[te_idx])

    in_dim = X_tr.shape[1]
    model = GEVNNSingleTower(in_dim, dropout=0.0).to(DEVICE)
    model.load_state_dict(torch.load(pt_paths[0], map_location=DEVICE))
    model.eval()

    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(tr_idx), size=min(N_SHAP_BG, len(tr_idx)), replace=False)
    te_idx_sub = rng.choice(len(te_idx), size=min(N_SHAP_TEST, len(te_idx)), replace=False)

    X_bg = torch.FloatTensor(X_tr[bg_idx]).to(DEVICE)
    X_sub = torch.FloatTensor(X_te[te_idx_sub]).to(DEVICE)

    class _QWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            mu, sig, xi = out
            from src.models import gev_quantile_torch

            qs = torch.stack([gev_quantile_torch(mu, sig, xi, t) for t in RETURN_PERIODS], dim=1)
            return torch.log(qs.clamp(min=1e-6))

    wrapper = _QWrapper(model).to(DEVICE)
    wrapper.eval()

    explainer = shap.GradientExplainer(wrapper, X_bg)
    sv = explainer.shap_values(X_sub)
    sv = _normalize_shap_matrix(
        sv,
        n_samples=X_sub.shape[0],
        n_features=X_sub.shape[1],
        context="GEV-NN",
    )

    return sv, X_te[te_idx_sub]


def shap_ann(model_dir: Path, data: dict, tr_idx, te_idx, use_flow: bool):
    flow_tag = _flow_tag(use_flow)
    scaler = _load_scaler(model_dir, flow_tag, SEED)
    X_all = data["X_full"] if use_flow else data["X_base"]
    X_tr = scaler.transform(X_all[tr_idx])
    X_te = scaler.transform(X_all[te_idx])

    in_dim = X_tr.shape[1]
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(tr_idx), size=min(N_SHAP_BG, len(tr_idx)), replace=False)
    te_sub = rng.choice(len(te_idx), size=min(N_SHAP_TEST, len(te_idx)), replace=False)
    X_bg = torch.FloatTensor(X_tr[bg_idx]).to(DEVICE)
    X_sub_t = torch.FloatTensor(X_te[te_sub]).to(DEVICE)

    single_candidates = []
    single_candidates.extend(sorted(model_dir.glob(f"PUB_ANN_Single_{flow_tag}_s{SEED}.pt")))
    single_candidates.extend(sorted(model_dir.glob(f"PUB_ANN_Direct_{flow_tag}_s{SEED}.pt")))
    single_candidates.extend(sorted(model_dir.glob(f"PUB_ANN*_{flow_tag}_s{SEED}.pt")))
    single_candidates = list(dict.fromkeys(single_candidates))

    if single_candidates:
        model = ANNSingle(in_dim, dropout=0.0).to(DEVICE)
        model.load_state_dict(torch.load(single_candidates[0], map_location=DEVICE))
        model.eval()

        explainer = shap.GradientExplainer(model, X_bg)
        sv = explainer.shap_values(X_sub_t)
        sv = _normalize_shap_matrix(
            sv,
            n_samples=X_sub_t.shape[0],
            n_features=X_sub_t.shape[1],
            context="ANN[single-file]",
        )
        return sv, X_te[te_sub]

    sv_list = []
    for t in RETURN_PERIODS:
        pt_paths = sorted(model_dir.glob(f"PUB_ANN_Single_T{t}_{flow_tag}_s{SEED}.pt"))
        if not pt_paths:
            pt_paths = sorted(model_dir.glob(f"PUB_ANN*T{t}*{flow_tag}*s{SEED}.pt"))
        if not pt_paths:
            pt_paths = sorted(model_dir.glob(f"PUB_ANN*{flow_tag}*s{SEED}*T{t}*.pt"))
        if not pt_paths:
            pt_paths = sorted(model_dir.glob(f"PUB_ANN_Single_{flow_tag}_s{SEED}_Q{t}.pt"))
        if not pt_paths:
            pt_paths = sorted(model_dir.glob(f"PUB_ANN*{flow_tag}*s{SEED}*Q{t}*.pt"))
        if not pt_paths:
            LOG.warning("ANN model for T=%d not found, skipping", t)
            continue

        model = ANNSingle(in_dim, dropout=0.0).to(DEVICE)
        model.load_state_dict(torch.load(pt_paths[0], map_location=DEVICE))
        model.eval()

        explainer = shap.GradientExplainer(model, X_bg)
        sv = explainer.shap_values(X_sub_t)
        sv = _normalize_shap_matrix(
            sv,
            n_samples=X_sub_t.shape[0],
            n_features=X_sub_t.shape[1],
            context=f"ANN[T={t}]",
        )
        sv_list.append(sv)

    if not sv_list:
        raise RuntimeError(
            "No ANN models found for SHAP computation. "
            f"Checked single-file and per-T naming under: {model_dir}"
        )

    sv_mean = np.mean(np.stack(sv_list, axis=0), axis=0)
    return sv_mean, X_te[te_sub]


# ------------------------------ Persistence ---------------------------------
def _feat_label_map() -> dict[str, str]:
    return {
        "static_p_mean": "Mean precip.",
        "static_aridity_FAO_PM": "Aridity index",
        "static_seasonality_FAO_PM": "Precip. seasonality",
        "static_frac_snow": "Snow fraction",
        "static_high_prec_freq": "High-prec. freq.",
        "static_high_prec_dur": "High-prec. duration",
        "static_low_prec_freq": "Low-prec. freq.",
        "static_low_prec_dur": "Low-prec. duration",
        "static_cmi_ix_syr": "Climate moisture index",
        "static_run_mm_syr": "Mean runoff depth",
        "static_area": "Catchment area",
        "static_ele_mt_sav": "Mean elevation",
        "static_slp_dg_sav": "Mean slope",
        "static_sgr_dk_sav": "Stream gradient",
        "static_ele_mt_smn": "Min elevation",
        "static_soc_th_sav": "Soil organic carbon",
        "static_cly_pc_sav": "Clay fraction",
        "static_snd_pc_sav": "Sand fraction",
        "static_for_pc_sse": "Forest cover",
        "static_crp_pc_sse": "Cropland fraction",
        "static_urb_pc_sse": "Urban fraction",
        "static_ppd_pk_sav": "Population density",
        "static_gwt_cm_sav": "Groundwater table depth",
        "static_snw_pc_syr": "Snow cover",
        "static_lka_pc_sse": "Lake fraction",
        "static_hft_ix_s93": "Human footprint",
        "static_dor_pc_pva": "Degree of regulation",
        "static_inu_pc_smx": "Max inundation extent",
        "static_inu_pc_slt": "Long-term inundation",
        "static_wet_pc_s09": "Wetland fraction",
        "prec_roll3d_ann_max_mean": "3-day precip. max",
        "prec_roll5d_ann_max_mean": "5-day precip. max",
        "prec_roll7d_ann_max_mean": "7-day precip. max",
        "prec_roll30d_ann_max_mean": "30-day precip. max",
        "temp_roll7d_ann_max": "7-day temp. max",
        "temp_roll30d_ann_max": "30-day temp. max",
        "ams_mean": "Sim. AMS mean",
        "ams_std": "Sim. AMS std",
        "ams_cv": "Sim. AMS CV",
        "roll3d_ann_max_mean": "Sim. 3-day max",
        "roll5d_ann_max_mean": "Sim. 5-day max",
        "roll7d_ann_max_mean": "Sim. 7-day max",
        "roll30d_ann_max_mean": "Sim. 30-day max",
        "flow_mean": "Sim. mean flow",
        "flow_std": "Sim. flow std",
        "flow_cv": "Sim. flow CV",
        "high_flow_freq": "High-flow freq.",
        "seasonality_ratio": "Flow seasonality",
    }


def _get_label(fname: str) -> str:
    return _feat_label_map().get(fname, fname.replace("static_", ""))


def save_shap_csv(sv: np.ndarray, feat_names: list[str], model: str, flow_tag: str) -> Path:
    cols = [_get_label(f) for f in feat_names]
    df = pd.DataFrame(np.abs(sv), columns=cols)
    summary = df[cols].mean().reset_index()
    summary.columns = ["feature", "mean_abs_shap"]
    summary = summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    model_key = model.replace("-", "_")
    flow_key = "flow" if flow_tag == "+flow" else "base"
    path = OUT_DATA / f"shap_importance_{model_key}_{flow_key}.csv"
    summary.to_csv(path, index=False)
    return path


def save_shap_cache(sv: np.ndarray, x_sub: np.ndarray, feat_names: list[str], model: str, flow_tag: str) -> Path:
    model_key = model.replace("-", "_")
    flow_key = "flow" if flow_tag == "+flow" else "base"
    out_path = OUT_DATA / f"shap_cache_{model_key}_{flow_key}.npz"

    np.savez_compressed(
        out_path,
        shap_values=np.asarray(sv, dtype=np.float32),
        x_sub=np.asarray(x_sub, dtype=np.float32),
        feature_names=np.asarray(feat_names, dtype=object),
        model=np.asarray([model], dtype=object),
        flow_tag=np.asarray([flow_tag], dtype=object),
        seed=np.asarray([SEED], dtype=np.int32),
    )
    return out_path


# ---------------------------------- Main -----------------------------------
def main() -> None:
    LOG.info("=" * 60)
    LOG.info("15_SHAP compute-only pipeline start")
    LOG.info("=" * 60)

    data = DatasetBuilder(NC_PATH, GEV_CSV, FLOW_CSV).build()
    n = len(data["X_base"])
    tr_idx, val_idx, te_idx = pub_split(n, SEED)
    LOG.info("PUB split: train=%d val=%d test=%d", len(tr_idx), len(val_idx), len(te_idx))

    jobs = [
        ("RF", False, shap_rf),
        ("ANN", False, shap_ann),
        ("GEV-NN", False, shap_gev_nn),
        ("GEV-NN", True, shap_gev_nn),
    ]

    index_rows: list[dict] = []

    for model_name, use_flow, fn in jobs:
        flow_tag = _flow_tag(use_flow)
        feat_names = data["feat_base"] + (data["feat_flow"] if use_flow else [])
        LOG.info("Computing SHAP: %s (%s)", model_name, flow_tag)

        try:
            sv, x_sub = fn(MODEL_DIRS[model_name], data, tr_idx, te_idx, use_flow)
            cache_path = save_shap_cache(sv, x_sub, feat_names, model_name, flow_tag)
            csv_path = save_shap_csv(sv, feat_names, model_name, flow_tag)

            index_rows.append(
                {
                    "model": model_name,
                    "flow_tag": flow_tag,
                    "n_samples": int(np.asarray(x_sub).shape[0]),
                    "n_features": int(np.asarray(x_sub).shape[1]),
                    "cache_file": cache_path.name,
                    "importance_csv": csv_path.name,
                }
            )
            LOG.info("Saved cache: %s", cache_path)
            LOG.info("Saved summary: %s", csv_path)
        except Exception as exc:
            LOG.error("%s (%s) failed: %s", model_name, flow_tag, exc)

    if not index_rows:
        LOG.error("No SHAP cache was produced. Please check model files and naming patterns.")
        return

    idx_path = OUT_DATA / "shap_cache_index.csv"
    pd.DataFrame(index_rows).to_csv(idx_path, index=False)
    LOG.info("Saved cache index: %s", idx_path)
    LOG.info("Done. SHAP cache output -> %s", OUT_DATA)


if __name__ == "__main__":
    main()
