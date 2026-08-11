# -*- coding: utf-8 -*-
"""13_GEV_NN_NLL.py  --  GEV-NN-ST trained with pure NLL loss (model label: GEV-NN-NLL)
               x base/+flow x PUB/PUR x 3-seed

Architecture:  GEVNNSingleTower  (identical to 11_GEV_NN.py)
               [256, 128] -> (mu, sigma, xi) -> GEV quantiles

Loss:          Pure Negative Log-Likelihood only
               L = NLL(mu, sigma, xi | AMS)
               No MSE term on quantile predictions.

Motivation:
    11_GEV_NN.py uses  L = MSE(log Q) + lambda * NLL.
    This ablation removes the MSE component entirely, forcing the model to
    learn GEV parameters exclusively from the likelihood signal.
    Expected behaviour:
      + Better distributional calibration (parameters statistically optimal)
      - Potentially lower R2 on specific quantiles (no direct supervision)
    Comparing with 11_GEV_NN.py isolates the contribution of the quantile MSE term.

All data / features / splits / hyperparams identical to 11_GEV_NN.py.
"""
import sys, logging
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

from src.paths import DATA_PROCEED, FIGURE_ROOT, MODEL_ROOT, stage_dir

import torch
from src.dataset import DatasetBuilder
from src._runner import RunConfig, run_experiment
from src.models import GEVNNSingleTower

NC_PATH = DATA_PROCEED / "02_Data-Clean" / "4_Cara-GRDC-35_cleaned.nc"
GEV_CSV = DATA_PROCEED / "01_GEV-Fit" / "gev_station_params.csv"
FLOW_CSV = DATA_PROCEED / "03_Streamflow-Process" / "sim_flow_features_per_station.csv"
BASIN_CSV = DATA_PROCEED / "05_PUR_Basin_Select" / "station_basin_assignment.csv"

TAG = "13_GEV_NN_NLL"
OUT_DATA = stage_dir(DATA_PROCEED, TAG)
OUT_FIG = stage_dir(FIGURE_ROOT, TAG)
OUT_MODEL = stage_dir(MODEL_ROOT, TAG)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(OUT_DATA / "log.txt", mode="w", encoding="utf-8")])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456]
SPLIT_SEEDS = [42, 123, 456]
MIN_PUR_STATIONS = 50
HIDDEN_DIMS = [256, 128]

if __name__ == "__main__":
    data = DatasetBuilder(NC_PATH, GEV_CSV, FLOW_CSV).build()
    cfg = RunConfig(
        model_name   = "GEV-NN-NLL",
        model_class  = GEVNNSingleTower,
        is_rf        = False,
        loss_fn      = None,         # not used; compute_loss defined inside _runner
        data         = data,
        seeds        = SEEDS,
        split_seeds  = SPLIT_SEEDS,
        basin_csv    = BASIN_CSV,
        out_data     = OUT_DATA,
        out_fig      = OUT_FIG,
        out_model    = OUT_MODEL,
        min_pur_stations = MIN_PUR_STATIONS,
        device       = DEVICE,
        model_hidden_dims = HIDDEN_DIMS,
        # ── pure NLL: no MSE term, NLL warmup irrelevant ──────────────────
        use_pure_nll = True,
        nll_lambda   = 1.0,
        nll_warmup   = 0,
        # ── architecture / optimiser (same as 11_GEV_NN.py) ──────────────
        batch_size   = 256,
        num_epochs   = 500,
        patience     = 30,
        lr           = 1e-3,
        weight_decay = 1e-4,
        grad_clip    = 5.0,
        dropout      = 0.1,
    )
    run_experiment(cfg)
