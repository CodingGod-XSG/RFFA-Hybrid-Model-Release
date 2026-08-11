# -*- coding: utf-8 -*-
"""09_ANN.py  –  ANN-Single (6 independent ANNs, one per T)  ±flow × PUB/PUR × 3-seed

Experiment B: ablation of joint prediction.
Trains 6 completely independent ANNs (same architecture as ANNDirect backbone),
each specialising on a single return period.  Compared against ANN-Joint (10_ANN_Joint.py)
to measure the value of joint multi-output learning.
"""
import sys, logging
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

from src.paths import DATA_PROCEED, FIGURE_ROOT, MODEL_ROOT, stage_dir

import torch
from src.dataset import DatasetBuilder
from src._runner import RunConfig, run_experiment
from src.models import ANNSingle

# ── Paths ────────────────────────────────────────────────────────────────────
NC_PATH = DATA_PROCEED / "02_Data-Clean" / "4_Cara-GRDC-35_cleaned.nc"
GEV_CSV = DATA_PROCEED / "01_GEV-Fit" / "gev_station_params.csv"
FLOW_CSV = DATA_PROCEED / "03_Streamflow-Process" / "sim_flow_features_per_station.csv"
BASIN_CSV = DATA_PROCEED / "05_PUR_Basin_Select" / "station_basin_assignment.csv"

TAG = "09_ANN"
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
        model_name    = "ANN-Single",
        model_class   = ANNSingle,
        is_rf         = False,
        is_ann_single = True,
        loss_fn       = None,
        data          = data,
        seeds         = SEEDS,
        split_seeds   = SPLIT_SEEDS,
        basin_csv     = BASIN_CSV,
        out_data      = OUT_DATA,
        out_fig       = OUT_FIG,
        out_model     = OUT_MODEL,
        min_pur_stations = MIN_PUR_STATIONS,
        device        = DEVICE,
        model_hidden_dims = HIDDEN_DIMS,
        dropout       = 0.1,
        batch_size    = 256,
        num_epochs    = 500,
        patience      = 30,
        lr            = 1e-3,
        weight_decay  = 1e-4,
        grad_clip     = 5.0,
        nll_lambda    = 0.01,
        nll_warmup    = 5,
    )
    run_experiment(cfg)
