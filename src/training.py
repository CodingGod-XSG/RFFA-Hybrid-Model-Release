# -*- coding: utf-8 -*-
"""
4_12_base/training.py
Generic NN training loop + RF wrapper.

Public API
----------
train_nn(model, device, tr_dl, val_dl, compute_loss_fn,
         num_epochs, patience, lr, weight_decay, grad_clip,
         nll_lambda, nll_warmup, tag)
    → training history dict

make_dataloader(X, targets, batch_size, shuffle, extra=None)
    → DataLoader

train_rf(X_tr, q_tr, n_estimators, max_features, n_jobs, seed, min_samples_leaf, max_depth)
    → list of 6 fitted RandomForestRegressor (one per return period)
train_xgb(X_tr, q_tr, n_estimators, learning_rate, max_depth, subsample,
          colsample_bytree, min_child_weight, n_jobs, seed)
    → list of 6 fitted XGBRegressor (one per return period)
train_svm(X_tr, q_tr, c, epsilon, gamma, seed)
    → list of 6 fitted SVR (one per return period)
"""
from __future__ import annotations
import logging
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

try:
    from .models import RETURN_PERIODS
except ImportError:
    from models import RETURN_PERIODS

LOG = logging.getLogger(__name__)


def make_dataloader(X: np.ndarray,
                    targets: np.ndarray,
                    batch_size: int,
                    shuffle: bool,
                    extra: list[np.ndarray] | None = None) -> DataLoader:
    """Build a DataLoader from numpy arrays.

    Parameters
    ----------
    X       : (N, F)  float32
    targets : (N, *) or (N,)
    extra   : additional arrays (e.g. ams, ams_mask) each (N, *)
    """
    tensors = [torch.FloatTensor(X), torch.FloatTensor(targets)]
    if extra:
        for a in extra:
            if a.dtype == bool:
                tensors.append(torch.BoolTensor(a))
            else:
                tensors.append(torch.FloatTensor(a.astype(np.float32)))
    ds = TensorDataset(*tensors)
    n  = len(X)
    bs = min(batch_size, max(1, n // 2)) if n < batch_size else batch_size
    return DataLoader(ds, bs, shuffle=shuffle,
                      drop_last=(shuffle and n >= batch_size))


def train_nn(model: nn.Module,
             device: torch.device,
             tr_dl: DataLoader,
             val_dl: DataLoader,
             compute_loss_fn,
             num_epochs: int   = 500,
             patience: int     = 30,
             lr: float         = 1e-3,
             weight_decay: float = 1e-4,
             grad_clip: float  = 5.0,
             nll_lambda: float = 0.01,
             nll_warmup: int   = 5,
             tag: str          = "") -> dict:
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch  = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15, min_lr=1e-6)
    best_val, pat, best_state = 1e30, 0, {}
    hist = {"train": [], "val": []}

    for ep in range(1, num_epochs + 1):
        lam = nll_lambda * min(1.0, ep / max(nll_warmup, 1))

        model.train()
        tr_losses = []
        for batch in tr_dl:
            batch = [b.to(device) for b in batch]
            loss = compute_loss_fn(model, batch, lam)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                batch = [b.to(device) for b in batch]
                val_losses.append(compute_loss_fn(model, batch, lam).item())

        tr_avg  = float(np.mean(tr_losses))
        val_avg = float(np.mean(val_losses))
        hist["train"].append(tr_avg)
        hist["val"].append(val_avg)
        sch.step(val_avg)

        if val_avg < best_val:
            best_val   = val_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1

        if ep <= 3 or ep % 50 == 0 or pat == 0:
            LOG.info(f"    [{tag}] ep{ep:3d}  tr={tr_avg:.5f}  "
                     f"val={val_avg:.5f}  pat={pat}")
        if pat >= patience:
            LOG.info(f"    [{tag}] early stop at ep{ep}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    return hist


def train_rf(X_tr: np.ndarray,
             q_tr: np.ndarray,
             n_estimators: int   = 300,
             max_features: str | int | float = "sqrt",
             min_samples_leaf: int = 1,
             max_depth: int | None = None,
             n_jobs: int         = -1,
             seed: int           = 42) -> list:
    """Train 6 independent RF regressors, one per return period (log-space)."""
    rfs = []
    for j, T in enumerate(RETURN_PERIODS):
        log_y = np.log(q_tr[:, j].clip(min=1e-6))
        rf    = RandomForestRegressor(
            n_estimators     = n_estimators,
            max_features     = max_features,
            min_samples_leaf = min_samples_leaf,
            max_depth        = max_depth,
            n_jobs           = n_jobs,
            random_state     = seed)
        rf.fit(X_tr, log_y)
        rfs.append(rf)
    return rfs


def train_xgb(X_tr: np.ndarray,
              q_tr: np.ndarray,
              n_estimators: int = 500,
              learning_rate: float = 0.05,
              max_depth: int = 6,
              subsample: float = 0.8,
              colsample_bytree: float = 0.8,
              min_child_weight: float = 5.0,
              n_jobs: int = -1,
              seed: int = 42) -> list:
    """Train 6 independent XGBoost regressors, one per return period (log-space)."""
    try:
        XGBRegressor = getattr(importlib.import_module("xgboost"), "XGBRegressor")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for train_xgb. Install with: pip install xgboost"
        ) from exc

    xgbs = []
    for j, _ in enumerate(RETURN_PERIODS):
        log_y = np.log(q_tr[:, j].clip(min=1e-6))
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
        )
        model.fit(X_tr, log_y)
        xgbs.append(model)
    return xgbs


def train_svm(X_tr: np.ndarray,
              q_tr: np.ndarray,
              c: float = 10.0,
              epsilon: float = 0.1,
              gamma: str = "scale",
              seed: int = 42,
              ) -> list:
    """Train 6 independent SVR models, one per return period (log-space)."""
    svms = []
    n = X_tr.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    X_use = X_tr[perm]

    for j, _ in enumerate(RETURN_PERIODS):
        y = q_tr[:, j]
        log_y = np.log(y.clip(min=1e-6))[perm]
        model = SVR(
            kernel="rbf",
            C=c,
            epsilon=epsilon,
            gamma=gamma,
        )
        model.fit(X_use, log_y)
        svms.append(model)
    return svms
