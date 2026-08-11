# -*- coding: utf-8 -*-
"""
4_12_base/evaluation.py
Metrics, per-model eval wrappers, and permutation importance.

Public API
----------
metrics(y_true, y_pred) → dict
eval_gev(model, device, X, q_true) → (metrics_dict_per_T, q_pred [N,6])
eval_ann(model, device, X, q_true) → (metrics_dict_per_T, q_pred [N,6])
eval_ann_single(models, device, X, q_true) → (metrics_dict_per_T, q_pred [N,6])
eval_rf(rfs, X, q_true)            → (metrics_dict_per_T, q_pred [N,6])
permutation_importance_nn(model_type, model, device, X, q_true,
                           feat_names, n_repeats=5)
    → DataFrame(feature_name, importance_mean, importance_std)
permutation_importance_rf(rfs, X, q_true, feat_names, n_repeats=5)
    → DataFrame(feature_name, importance_mean, importance_std)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
try:
    from .models import (
        RETURN_PERIODS, gev_quantile_np, gumbel_quantile_np
    )
except ImportError:
    from models import (
        RETURN_PERIODS, gev_quantile_np, gumbel_quantile_np
    )

Q10_IDX = RETURN_PERIODS.index(10)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = float(1.0 - ss_res / max(ss_tot, 1e-12))
    rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rrmse  = rmse / max(float(np.mean(np.abs(y_true))), 1e-12) * 100.0
    pbias  = 100.0 * float(np.sum(y_pred - y_true)) / max(float(np.sum(y_true)), 1e-12)
    mrat   = float(np.median(y_pred / np.where(y_true != 0, y_true, 1e-12)))
    # KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    std_obs  = float(np.std(y_true,  ddof=1)) if len(y_true)  > 1 else 1e-12
    std_pred = float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 0.0
    mean_obs = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    r_val = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    alpha = std_pred / max(std_obs, 1e-12)
    beta  = mean_pred / max(abs(mean_obs), 1e-12)
    kge   = float(1.0 - np.sqrt((r_val - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))
    return dict(R2=r2, RMSE=rmse, rRMSE=rrmse, PBIAS=pbias, MedianRatio=mrat, KGE=kge)


def _metrics_all_T(q_true: np.ndarray, q_pred: np.ndarray) -> dict:
    return {f"Q{T}": metrics(q_true[:, j], q_pred[:, j])
            for j, T in enumerate(RETURN_PERIODS)}


# ─── GEV model eval ─────────────────────────────────────────────────────────

def eval_gev(model, device, X: np.ndarray,
             q_true: np.ndarray) -> tuple[dict, np.ndarray]:
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X).to(device))
    mt  = model.TYPE
    if mt in ("gev_st", "gev_tt"):
        mu, sig, xi = (v.cpu().numpy() for v in out)
        q_pred = np.column_stack(
            [gev_quantile_np(mu, sig, xi, T) for T in RETURN_PERIODS]
        )
    else:   # gumbel_st / gumbel_dt
        mu, sig = (v.cpu().numpy() for v in out)
        q_pred = np.column_stack(
            [gumbel_quantile_np(mu, sig, T) for T in RETURN_PERIODS]
        )
    return _metrics_all_T(q_true, q_pred), q_pred


# ─── ANN-Direct eval ────────────────────────────────────────────────────────

def eval_ann(model, device, X: np.ndarray,
             q_true: np.ndarray) -> tuple[dict, np.ndarray]:
    model.eval()
    with torch.no_grad():
        log_pred = model(torch.FloatTensor(X).to(device)).cpu().numpy()
    q_pred = np.exp(log_pred)     # shape (N, 6)
    return _metrics_all_T(q_true, q_pred), q_pred


def eval_ann_single(models: list, device, X: np.ndarray,
                    q_true: np.ndarray) -> tuple[dict, np.ndarray]:
    """Eval 6 independent single-output ANNs."""
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            log_pred = model(torch.FloatTensor(X).to(device)).cpu().numpy()  # (N,1)
        preds.append(np.exp(log_pred[:, 0]))
    q_pred = np.column_stack(preds)   # (N, 6)
    return _metrics_all_T(q_true, q_pred), q_pred


# ─── RF eval ────────────────────────────────────────────────────────────────

def eval_sklearn_ensemble(rfs: list, X: np.ndarray,
            q_true: np.ndarray) -> tuple[dict, np.ndarray]:
    """Works for RF, XGBoost, and SVM since all expose .predict(X)."""
    q_pred = np.column_stack([np.exp(rf.predict(X)) for rf in rfs])
    return _metrics_all_T(q_true, q_pred), q_pred

eval_rf = eval_sklearn_ensemble


# ─── Permutation importance ──────────────────────────────────────────────────

def _nn_r2_q10(model, device, X: np.ndarray, q_true: np.ndarray) -> float:
    mt = model.TYPE
    if mt == "ann_direct":
        _, q_pred = eval_ann(model, device, X, q_true)
    else:
        _, q_pred = eval_gev(model, device, X, q_true)
    m = metrics(q_true[:, Q10_IDX], q_pred[:, Q10_IDX])
    return m["R2"]


def _rf_r2_q10(rfs: list, X: np.ndarray, q_true: np.ndarray) -> float:
    _, q_pred = eval_rf(rfs, X, q_true)
    return metrics(q_true[:, Q10_IDX], q_pred[:, Q10_IDX])["R2"]


def permutation_importance_nn(model, device,
                               X: np.ndarray,
                               q_true: np.ndarray,
                               feat_names: list[str],
                               n_repeats: int = 5) -> pd.DataFrame:
    base_r2 = _nn_r2_q10(model, device, X, q_true)
    imps = []
    rng  = np.random.RandomState(0)
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp     = X.copy()
            perm   = rng.permutation(X.shape[0])
            Xp[:, j] = Xp[perm, j]
            drops.append(base_r2 - _nn_r2_q10(model, device, Xp, q_true))
        imps.append(drops)
    arr = np.array(imps)    # (n_feat, n_repeats)
    return pd.DataFrame({
        "feature_name":    feat_names,
        "importance_mean": arr.mean(axis=1),
        "importance_std":  arr.std(axis=1),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def permutation_importance_rf(rfs: list,
                               X: np.ndarray,
                               q_true: np.ndarray,
                               feat_names: list[str],
                               n_repeats: int = 5) -> pd.DataFrame:
    base_r2 = _rf_r2_q10(rfs, X, q_true)
    imps = []
    rng  = np.random.RandomState(0)
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp     = X.copy()
            perm   = rng.permutation(X.shape[0])
            Xp[:, j] = Xp[perm, j]
            drops.append(base_r2 - _rf_r2_q10(rfs, Xp, q_true))
        imps.append(drops)
    arr = np.array(imps)
    return pd.DataFrame({
        "feature_name":    feat_names,
        "importance_mean": arr.mean(axis=1),
        "importance_std":  arr.std(axis=1),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
