# -*- coding: utf-8 -*-
"""
src/_runner.py
Common experiment runner used by the 06_RF.py-13_GEV_NN_NLL.py model-training scripts.

Public API
----------
run_experiment(cfg)  →  None
    Runs PUB + PUR for one model; saves all outputs.

cfg is a RunConfig(dataclass-like) with fields:
    model_name      str   e.g. "GEV-NN-ST"
    model_class     type  NN model or None (RF)
    is_rf           bool
    loss_fn         callable(model, batch, lambda) → scalar loss  (None for RF)
    data            dict  from DatasetBuilder.build()
    seeds           list  e.g. [42, 123, 456]
    basin_csv       Path
    out_data        Path
    out_fig         Path
    out_model       Path
    device          torch.device
    # hyperparams
    batch_size      int
    num_epochs      int
    patience        int
    lr              float
    weight_decay    float
    grad_clip       float
    nll_lambda      float
    nll_warmup      int
    dropout         float
    model_hidden_dims list[int] | None
    rf_n_estimators int
    rf_max_features str | int | float
    rf_min_samples_leaf int
    rf_max_depth    int | None
    xgb_n_estimators int
    xgb_learning_rate float
    xgb_max_depth   int
    xgb_subsample   float
    xgb_colsample_bytree float
    xgb_min_child_weight float
    svm_c           float
    svm_epsilon     float
    svm_gamma       str
"""
from __future__ import annotations
import logging, time, dataclasses, random, hashlib
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from .models import RETURN_PERIODS
    from .splits import pub_split, pur_splits, pur_train_val_split
    from .training import train_nn, train_rf, train_xgb, train_svm, make_dataloader
    from .evaluation import (eval_gev, eval_ann, eval_sklearn_ensemble)
except ImportError:
    # Backward-compatible path when imported as top-level modules.
    from models import RETURN_PERIODS
    from splits import pub_split, pur_splits, pur_train_val_split
    from training import train_nn, train_rf, train_xgb, train_svm, make_dataloader
    from evaluation import (eval_gev, eval_ann, eval_sklearn_ensemble)

LOG = logging.getLogger(__name__)
SEEDS_DEFAULT = [42, 123, 456]


@dataclasses.dataclass
class RunConfig:
    model_name:      str
    model_class:     object          # NN class or None
    is_rf:           bool
    loss_fn:         Optional[Callable]
    data:            dict
    seeds:           list
    basin_csv:       Path
    out_data:        Path
    out_fig:         Path
    out_model:       Path
    device:          object
    is_xgb:          bool  = False
    is_svm:          bool  = False
    split_seeds:     Optional[list] = None
    batch_size:      int   = 256
    num_epochs:      int   = 500
    patience:        int   = 30
    lr:              float = 1e-3
    weight_decay:    float = 1e-4
    grad_clip:       float = 5.0
    nll_lambda:      float = 0.01
    nll_warmup:      int   = 5
    dropout:         float = 0.1
    model_hidden_dims: Optional[list[int]] = None
    rf_n_estimators: int   = 300
    rf_max_features: str | int | float = "sqrt"
    rf_min_samples_leaf: int = 5
    rf_max_depth:    int | None = None
    xgb_n_estimators: int = 500
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: float = 5.0
    svm_c:          float = 10.0
    svm_epsilon:    float = 0.1
    svm_gamma:      str   = "scale"
    min_pur_stations:int   = 50
    use_pure_nll:    bool  = False
    is_ann_single:   bool  = False
    save_pur_models: bool  = False    # True → also save NN weights for every PUR fold


def _stable_seed(base_seed: int, context: str) -> int:
    msg = f"{base_seed}|{context}".encode("utf-8")
    digest = hashlib.sha256(msg).hexdigest()
    return int(digest[:8], 16)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _X(data: dict, use_flow: bool) -> np.ndarray:
    return data["X_full"] if use_flow else data["X_base"]


def _feat_names(data: dict, use_flow: bool) -> list[str]:
    return data["feat_base"] + (data["feat_flow"] if use_flow else [])


def _eval_model(cfg: RunConfig, model_or_rfs, device, X_te, q_te):
    if cfg.is_rf or cfg.is_xgb or cfg.is_svm:
        return eval_sklearn_ensemble(model_or_rfs, X_te, q_te)
    if cfg.is_ann_single:
        try:
            from .evaluation import eval_ann_single
        except ImportError:
            from evaluation import eval_ann_single
        return eval_ann_single(model_or_rfs, device, X_te, q_te)
    mt = model_or_rfs.TYPE
    if mt == "ann_direct":
        return eval_ann(model_or_rfs, device, X_te, q_te)
    return eval_gev(model_or_rfs, device, X_te, q_te)


def _rows_from_metrics(met: dict, experiment: str, fold: str,
                        seed: int, model: str, use_flow: bool) -> list[dict]:
    rows = []
    for T_key, m in met.items():
        rows.append(dict(
            experiment    = experiment,
            fold          = fold,
            seed          = seed,
            model         = model,
            use_flow      = use_flow,
            return_period = int(T_key[1:]),   # "Q10" → 10
            R2            = m["R2"],
            RMSE          = m["RMSE"],
            rRMSE         = m["rRMSE"],
            PBIAS         = m["PBIAS"],
            MedianRatio   = m["MedianRatio"],
        ))
    return rows


def _pred_df(stations, lat, lon, q_true, q_pred) -> pd.DataFrame:
    d = {"station_id": stations, "lat": lat, "lon": lon}
    for j, T in enumerate(RETURN_PERIODS):
        d[f"Q{T}_true"] = q_true[:, j]
    for j, T in enumerate(RETURN_PERIODS):
        d[f"Q{T}_pred"] = q_pred[:, j]
    return pd.DataFrame(d)


def _gev_params_df(stations, lat, lon, model, device, X) -> pd.DataFrame | None:
    """
    Run GEV/Gumbel model forward and return predicted parameters as a DataFrame.
    Returns None for ANN-Direct and RF (no distributional parameters).
    Columns: station_id, lat, lon, mu, sigma, xi
    For Gumbel models xi is fixed to 0.
    """
    if isinstance(model, list):
        return None
    if not hasattr(model, "TYPE"):
        return None
    mt = model.TYPE
    if mt == "ann_direct":
        return None
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X).to(device))
    if mt in ("gev_st", "gev_tt"):
        mu  = out[0].cpu().numpy()
        sig = out[1].cpu().numpy()
        xi  = out[2].cpu().numpy()
    else:   # gumbel_st / gumbel_dt  (xi ≡ 0)
        mu  = out[0].cpu().numpy()
        sig = out[1].cpu().numpy()
        xi  = np.zeros_like(mu)
    return pd.DataFrame({
        "station_id": stations,
        "lat":        lat,
        "lon":        lon,
        "mu":         mu,
        "sigma":      sig,
        "xi":         xi,
    })


def _train_one(cfg: RunConfig, X_tr, X_val, q_tr, q_val,
               ams_tr, mask_tr, ams_val, mask_val, gev_tr,
               scaler, seed: int, tag: str):
    """Train & return model (NN or RF list)."""
    if cfg.is_rf:
        return train_rf(X_tr, q_tr,
                    n_estimators     = cfg.rf_n_estimators,
                    max_features     = cfg.rf_max_features,
                    min_samples_leaf = cfg.rf_min_samples_leaf,
                    max_depth        = cfg.rf_max_depth,
                    n_jobs           = -1, seed=seed)

    if cfg.is_xgb:
        return train_xgb(X_tr, q_tr,
                         n_estimators     = cfg.xgb_n_estimators,
                         learning_rate    = cfg.xgb_learning_rate,
                         max_depth        = cfg.xgb_max_depth,
                         subsample        = cfg.xgb_subsample,
                         colsample_bytree = cfg.xgb_colsample_bytree,
                         min_child_weight = cfg.xgb_min_child_weight,
                         n_jobs           = -1, seed=seed)

    if cfg.is_svm:
        return train_svm(X_tr, q_tr,
                         c                = cfg.svm_c,
                         epsilon          = cfg.svm_epsilon,
                         gamma            = cfg.svm_gamma,
                         seed             = seed)

    if cfg.is_ann_single:
        try:
            from .models import ANNDirect
        except ImportError:
            from models import ANNDirect

        models = []
        in_dim = X_tr.shape[1]
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        def _make_loss(idx):
            def compute_loss(model, batch, lam):
                xb, qb = batch[0], batch[1]
                log_pred = model(xb)[:, 0]
                log_true = torch.log(qb[:, idx].clamp(min=1e-6))
                return nn.functional.mse_loss(log_pred, log_true)
            return compute_loss
        try:
            for j, T in enumerate(RETURN_PERIODS):
                _set_global_seed(seed + j)
                tr_dl = make_dataloader(X_tr, q_tr, cfg.batch_size, True)
                val_dl = make_dataloader(X_val, q_val, cfg.batch_size, False)
                m = ANNDirect(in_dim, cfg.dropout, n_out=1,
                              hidden_dims=cfg.model_hidden_dims).to(cfg.device)
                log_q_med = float(np.median(np.log(q_tr[:, j].clip(min=1e-6))))
                with torch.no_grad():
                    m.net[-1].bias[0] = log_q_med

                train_nn(m, cfg.device, tr_dl, val_dl, _make_loss(j),
                         num_epochs=cfg.num_epochs, patience=cfg.patience,
                         lr=cfg.lr, weight_decay=cfg.weight_decay,
                         grad_clip=cfg.grad_clip, nll_lambda=0.0,
                         nll_warmup=0, tag=f"{tag}|Q{T}")
                models.append(m)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        return models

    in_dim = X_tr.shape[1]
    if cfg.model_hidden_dims is None:
        model = cfg.model_class(in_dim, cfg.dropout).to(cfg.device)
    else:
        model = cfg.model_class(in_dim, cfg.dropout, hidden_dims=cfg.model_hidden_dims).to(cfg.device)
    mt = model.TYPE
    if mt == "ann_direct":
        log_q_meds = np.median(np.log(q_tr.clip(min=1e-6)), axis=0)
        model.init_bias(log_q_meds)
    else:
        model.init_bias(float(np.median(gev_tr[:, 0])),
                        float(np.median(gev_tr[:, 1])))

    def compute_loss(m, batch, lam):
        if mt == "ann_direct":
            xb, qb = batch[0], batch[1]
            log_pred = m(xb)
            log_true = torch.log(qb.clamp(min=1e-6))
            return nn.functional.mse_loss(log_pred, log_true)
        else:
            xb, qb, ab, mb = batch[0], batch[1], batch[2], batch[3]
            out = m(xb)
            # import here to avoid circular at module level
            try:
                from .models import (gev_quantile_torch, gumbel_quantile_torch,
                                     gev_nll_torch, gumbel_nll_torch)
            except ImportError:
                from models import (gev_quantile_torch, gumbel_quantile_torch,
                                    gev_nll_torch, gumbel_nll_torch)
            import torch.nn.functional as F
            if mt in ("gev_st", "gev_tt"):
                mu, sig, xi = out
                q_pred = torch.stack(
                    [gev_quantile_torch(mu, sig, xi, T) for T in RETURN_PERIODS], 1)
                q_loss = F.mse_loss(torch.log(q_pred.clamp(1e-6)),
                                     torch.log(qb.clamp(1e-6)))
                nll    = gev_nll_torch(mu, sig, xi, ab, mb)
            else:   # gumbel
                mu, sig = out
                q_pred = torch.stack(
                    [gumbel_quantile_torch(mu, sig, T) for T in RETURN_PERIODS], 1)
                q_loss = F.mse_loss(torch.log(q_pred.clamp(1e-6)),
                                     torch.log(qb.clamp(1e-6)))
                nll    = gumbel_nll_torch(mu, sig, ab, mb)
            if cfg.use_pure_nll:
                return nll
            return q_loss + lam * nll

    if mt == "ann_direct":
        tr_dl  = make_dataloader(X_tr,  q_tr,  cfg.batch_size, True)
        val_dl = make_dataloader(X_val, q_val, cfg.batch_size, False)
    else:
        tr_dl  = make_dataloader(X_tr,  q_tr,  cfg.batch_size, True,
                                  extra=[ams_tr, mask_tr])
        val_dl = make_dataloader(X_val, q_val, cfg.batch_size, False,
                                  extra=[ams_val, mask_val])

    train_nn(model, cfg.device, tr_dl, val_dl, compute_loss,
             num_epochs   = cfg.num_epochs,
             patience     = cfg.patience,
             lr           = cfg.lr,
             weight_decay = cfg.weight_decay,
             grad_clip    = cfg.grad_clip,
             nll_lambda   = cfg.nll_lambda,
             nll_warmup   = cfg.nll_warmup,
             tag          = tag)
    return model


# ─── Main runner ─────────────────────────────────────────────────────────────

def run_experiment(cfg: RunConfig):
    t0      = time.time()
    data    = cfg.data
    N       = len(data["X_base"])
    all_idx = np.arange(N)
    mn      = cfg.model_name
    dev     = cfg.device

    for d in [cfg.out_data, cfg.out_fig, cfg.out_model]:
        d.mkdir(parents=True, exist_ok=True)

    LOG.info(f"\n{'='*60}")
    split_seeds = cfg.split_seeds if cfg.split_seeds else cfg.seeds
    LOG.info(f"{mn}  N={N}  split_seeds={split_seeds}  train_seeds={cfg.seeds}")

    all_met_rows       = []
    all_met_rows_train = []   # training-set metrics (same structure, experiment="PUB_train"/"PUR_train")

    # ════════════════════════════════════════════════════════
    #  PUB
    # ════════════════════════════════════════════════════════
    LOG.info("\n── PUB ──")
    for split_seed in split_seeds:
        tr_idx, val_idx, te_idx = pub_split(N, split_seed)
        LOG.info(f"  split_seed={split_seed}  tr={len(tr_idx)}  val={len(val_idx)}  te={len(te_idx)}")

        for use_flow in (False, True):
            flow_tag = "+flow" if use_flow else "base"
            train_seeds = cfg.seeds if (cfg.is_rf or cfg.is_xgb or cfg.is_svm) else [
                _stable_seed(split_seed, f"PUB|{flow_tag}")
            ]
            for train_seed in train_seeds:
                _set_global_seed(train_seed)
                X_all   = _X(data, use_flow)
                scaler  = StandardScaler().fit(X_all[tr_idx])
                X_tr    = scaler.transform(X_all[tr_idx])
                X_val   = scaler.transform(X_all[val_idx])
                X_te    = scaler.transform(X_all[te_idx])
                q_tr, q_val, q_te = (data["q_true"][i]
                                      for i in (tr_idx, val_idx, te_idx))
                ams_tr,  mask_tr  = data["ams"][tr_idx],  data["ams_mask"][tr_idx]
                ams_val, mask_val = data["ams"][val_idx], data["ams_mask"][val_idx]
                gev_tr            = data["gev_params"][tr_idx]
                tag = f"PUB|{mn}|{flow_tag}|split{split_seed}|train{train_seed}"

                model = _train_one(cfg, X_tr, X_val, q_tr, q_val,
                                   ams_tr, mask_tr, ams_val, mask_val,
                                   gev_tr, scaler, train_seed, tag)

                if cfg.is_rf or cfg.is_xgb or cfg.is_svm:
                    seed_tag = f"s{train_seed}_split{split_seed}"
                    metric_seed = train_seed
                else:
                    seed_tag = f"s{split_seed}"
                    metric_seed = split_seed

                # ── save model ──────────────────────────────────
                safe = f"{mn.replace('-','_')}_{flow_tag}_{seed_tag}"
                if cfg.is_rf or cfg.is_xgb or cfg.is_svm:
                    pref = mn.replace('-', '_')
                    joblib.dump(model, cfg.out_model / f"{pref}_PUB_{safe}.joblib")
                else:
                    if cfg.is_ann_single:
                        for j, T in enumerate(RETURN_PERIODS):
                            torch.save(model[j].state_dict(),
                                       cfg.out_model / f"PUB_{safe}_Q{T}.pt")
                    else:
                        torch.save(model.state_dict(),
                                   cfg.out_model / f"PUB_{safe}.pt")
                joblib.dump(scaler, cfg.out_model / f"scaler_PUB_{safe}.joblib")

                # ── eval + metrics ───────────────────────────────
                met, q_pred = _eval_model(cfg, model, dev, X_te, q_te)
                all_met_rows.extend(_rows_from_metrics(
                    met, "PUB", "PUB", metric_seed, mn, use_flow))

                met_tr, _ = _eval_model(cfg, model, dev, X_tr, q_tr)
                all_met_rows_train.extend(_rows_from_metrics(
                    met_tr, "PUB_train", "PUB", metric_seed, mn, use_flow))

                # ── predictions CSV ───────────────────────────────
                pf = _pred_df(data["stations"][te_idx],
                              data["lat"][te_idx], data["lon"][te_idx],
                              q_te, q_pred)
                pf.to_csv(
                    cfg.out_data
                    / f"predictions_{mn.replace('-','_')}_PUB_PUB_{seed_tag}_{flow_tag}.csv",
                    index=False,
                )

                # ── GEV parameters CSV (GEV/Gumbel models only) ───────
                gp = _gev_params_df(data["stations"][te_idx],
                                    data["lat"][te_idx], data["lon"][te_idx],
                                    model, dev, X_te)
                if gp is not None:
                    gp.to_csv(
                        cfg.out_data
                        / f"gev_params_{mn.replace('-','_')}_PUB_PUB_{seed_tag}_{flow_tag}.csv",
                        index=False,
                    )

    # ════════════════════════════════════════════════════════
    #  PUR
    # ════════════════════════════════════════════════════════
    LOG.info("\n── PUR ──")
    if not cfg.basin_csv.exists():
        LOG.warning(f"  basin CSV not found: {cfg.basin_csv}  – skipping PUR")
    else:
        folds = pur_splits(cfg.basin_csv, data["stations"], min_fold_n=cfg.min_pur_stations)
        LOG.info(f"  {len(folds)} folds")
        for fold_label, te_idx in sorted(folds.items()):
            for split_seed in split_seeds:
                tr_idx, val_idx = pur_train_val_split(all_idx, te_idx, split_seed)
                if len(tr_idx) == 0 or len(val_idx) == 0:
                    LOG.warning(
                        f"  Skip PUR fold={fold_label} split_seed={split_seed}: "
                        f"empty train/val (tr={len(tr_idx)}, val={len(val_idx)})"
                    )
                    continue
                for use_flow in (False, True):
                    flow_tag  = "+flow" if use_flow else "base"
                    train_seeds = cfg.seeds if (cfg.is_rf or cfg.is_xgb or cfg.is_svm) else [
                        _stable_seed(split_seed, f"PUR|{fold_label}|{flow_tag}")
                    ]
                    for train_seed in train_seeds:
                        _set_global_seed(train_seed)
                        X_all   = _X(data, use_flow)
                        scaler  = StandardScaler().fit(X_all[tr_idx])
                        X_tr    = scaler.transform(X_all[tr_idx])
                        X_val   = scaler.transform(X_all[val_idx])
                        X_te    = scaler.transform(X_all[te_idx])
                        q_tr    = data["q_true"][tr_idx]
                        q_val   = data["q_true"][val_idx]
                        q_te    = data["q_true"][te_idx]
                        ams_tr,  mask_tr  = data["ams"][tr_idx],  data["ams_mask"][tr_idx]
                        ams_val, mask_val = data["ams"][val_idx], data["ams_mask"][val_idx]
                        gev_tr            = data["gev_params"][tr_idx]
                        safe_fold = fold_label.replace(" ", "_")
                        tag = f"PUR|{fold_label}|{mn}|{flow_tag}|split{split_seed}|train{train_seed}"

                        model = _train_one(cfg, X_tr, X_val, q_tr, q_val,
                                           ams_tr, mask_tr, ams_val, mask_val,
                                           gev_tr, scaler, train_seed, tag)

                        if cfg.is_rf or cfg.is_xgb or cfg.is_svm:
                            seed_tag = f"s{train_seed}_split{split_seed}"
                            metric_seed = train_seed
                        else:
                            seed_tag = f"s{split_seed}"
                            metric_seed = split_seed

                        # ── save scaler (always) + model (optional) ──
                        safe_s = f"{mn.replace('-','_')}_PUR_{safe_fold}_{flow_tag}_{seed_tag}"
                        joblib.dump(scaler, cfg.out_model / f"scaler_{safe_s}.joblib")
                        if cfg.save_pur_models:
                            if cfg.is_rf or cfg.is_xgb or cfg.is_svm:
                                pref = mn.replace('-', '_')
                                joblib.dump(model, cfg.out_model / f"{pref}_{safe_s}.joblib")
                            else:
                                if cfg.is_ann_single:
                                    for j, T in enumerate(RETURN_PERIODS):
                                        torch.save(model[j].state_dict(),
                                                   cfg.out_model / f"{safe_s}_Q{T}.pt")
                                else:
                                    torch.save(model.state_dict(),
                                               cfg.out_model / f"{safe_s}.pt")

                        met, q_pred = _eval_model(cfg, model, dev, X_te, q_te)
                        all_met_rows.extend(_rows_from_metrics(
                            met, "PUR", fold_label, metric_seed, mn, use_flow))

                        met_tr, _ = _eval_model(cfg, model, dev, X_tr, q_tr)
                        all_met_rows_train.extend(_rows_from_metrics(
                            met_tr, "PUR_train", fold_label, metric_seed, mn, use_flow))

                        pf = _pred_df(data["stations"][te_idx],
                                      data["lat"][te_idx], data["lon"][te_idx],
                                      q_te, q_pred)
                        pf.to_csv(
                            cfg.out_data
                            / f"predictions_{mn.replace('-','_')}_PUR_{safe_fold}_{seed_tag}_{flow_tag}.csv",
                            index=False,
                        )

                        # ── GEV parameters CSV (GEV/Gumbel models only) ───
                        gp = _gev_params_df(data["stations"][te_idx],
                                            data["lat"][te_idx], data["lon"][te_idx],
                                            model, dev, X_te)
                        if gp is not None:
                            gp.to_csv(
                                cfg.out_data
                                / f"gev_params_{mn.replace('-','_')}_PUR_{safe_fold}_{seed_tag}_{flow_tag}.csv",
                                index=False,
                            )

    # ── save combined metrics CSV ────────────────────────────
    df_met = pd.DataFrame(all_met_rows)
    df_met.to_csv(cfg.out_data / f"metrics_{mn.replace('-','_')}.csv", index=False)

    if all_met_rows_train:
        df_met_train = pd.DataFrame(all_met_rows_train)
        df_met_train.to_csv(cfg.out_data / f"metrics_{mn.replace('-','_')}_train.csv", index=False)
        LOG.info(f"\n  Saved {len(df_met)} test rows + {len(df_met_train)} train rows  |  "
                 f"runtime={(time.time()-t0)/60:.1f} min")
    else:
        LOG.info(f"\n  Saved {len(df_met)} metric rows  |  "
                 f"runtime={(time.time()-t0)/60:.1f} min")
