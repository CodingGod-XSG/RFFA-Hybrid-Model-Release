# -*- coding: utf-8 -*-
"""12_GEV_NN_MSE.py  –  GEV-NN-ST MSE-only ablation (model label: GEV-NN-MSE)  x base/+flow x PUB/PUR x 3-seed.

This script keeps the same GEV-NN architecture as 11_GEV_NN.py, but disables
the NLL term by setting nll_lambda=0.0, so training objective becomes
pure quantile MSE in log space.
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

TAG = "12_GEV_NN_MSE"
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
        model_name   = "GEV-NN-MSEOnly",
        model_class  = GEVNNSingleTower,
        is_rf        = False,
        loss_fn      = None,
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
        # MSE-only ablation: disable likelihood regularization.
        nll_lambda   = 0.0,
        nll_warmup   = 0,
        batch_size   = 256,
        num_epochs   = 500,
        patience     = 30,
        lr           = 1e-3,
        weight_decay = 1e-4,
        grad_clip    = 5.0,
        dropout      = 0.1,
    )
    run_experiment(cfg)
