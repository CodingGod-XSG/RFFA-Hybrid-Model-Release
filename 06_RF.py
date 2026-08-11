# -*- coding: utf-8 -*-
"""06_RF.py  –  Random Forest (6 independent RFs per T)  ±flow × PUB/PUR
RF uses fixed random_state per run; multi-seed is supported for robustness.
"""
import sys, logging, argparse, os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

from src.paths import DATA_PROCEED, FIGURE_ROOT, MODEL_ROOT, stage_dir
from src.dataset import DatasetBuilder
from src._runner import RunConfig, run_experiment

NC_PATH = DATA_PROCEED / "02_Data-Clean" / "4_Cara-GRDC-35_cleaned.nc"
GEV_CSV = DATA_PROCEED / "01_GEV-Fit" / "gev_station_params.csv"
FLOW_CSV = DATA_PROCEED / "03_Streamflow-Process" / "sim_flow_features_per_station.csv"
BASIN_CSV = DATA_PROCEED / "05_PUR_Basin_Select" / "station_basin_assignment.csv"

TAG = "06_RF"
OUT_DATA = stage_dir(DATA_PROCEED, TAG)
OUT_FIG = stage_dir(FIGURE_ROOT, TAG)
OUT_MODEL = stage_dir(MODEL_ROOT, TAG)
DEFAULT_SEEDS = [42, 123, 456]
SPLIT_SEEDS = [42, 123, 456]
MIN_PUR_STATIONS = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(OUT_DATA / "log.txt", mode="w", encoding="utf-8")])


def _parse_seeds_text(text: str) -> list[int]:
    vals = []
    for tok in str(text).replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    out = []
    seen = set()
    for s in vals:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def resolve_seeds() -> list[int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds, e.g. 42,123,456")
    args, _ = parser.parse_known_args()

    raw = args.seeds if args.seeds is not None else os.environ.get("RF_SEEDS", "")
    if str(raw).strip() == "":
        return DEFAULT_SEEDS.copy()

    try:
        seeds = _parse_seeds_text(raw)
        return seeds if seeds else DEFAULT_SEEDS.copy()
    except Exception:
        logging.warning("Invalid --seeds/RF_SEEDS=%r, fallback to default %s", raw, DEFAULT_SEEDS)
        return DEFAULT_SEEDS.copy()

if __name__ == "__main__":
    import torch
    seeds = resolve_seeds()
    logging.info("RF run seeds: %s", seeds)
    if len(seeds) == 1:
        logging.warning("Only one seed configured; set --seeds 42,123,456 for multi-seed robustness.")

    data = DatasetBuilder(NC_PATH, GEV_CSV, FLOW_CSV).build()
    cfg = RunConfig(
        model_name       = "RF",
        model_class      = None,
        is_rf            = True,
        loss_fn          = None,
        data             = data,
        seeds            = seeds,
        split_seeds      = SPLIT_SEEDS,
        basin_csv        = BASIN_CSV,
        out_data         = OUT_DATA,
        out_fig          = OUT_FIG,
        out_model        = OUT_MODEL,
        min_pur_stations = MIN_PUR_STATIONS,
        device           = torch.device("cpu"),
        rf_n_estimators  = 300,
        rf_max_features  = "sqrt",
        rf_min_samples_leaf = 5,
        rf_max_depth        = None,
    )
    run_experiment(cfg)
