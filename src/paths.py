"""Central definition of the on-disk directory layout used throughout this repository.

Every script resolves its input/output locations from :data:`REPO_ROOT`, so the
pipeline runs unmodified regardless of where the repository is cloned. There is
no external config file: to point the pipeline at a different data location,
edit the constants below (or override them by setting the ``RFFA_DATA_ROOT``
environment variable before running a script).

Directory layout expected under ``REPO_ROOT`` (see README.md for details on
how to obtain each raw dataset):

    data/raw/Caravan/...                     Caravan dataset (Kratzert et al., 2023)
    data/raw/GRDC-Caravan/...                GRDC-Caravan extension (Farber et al., 2025)
    data/raw/Sim-Dis/...                     GHM simulated streamflow (Ji et al., 2025)
    data/raw/Hydrosheds/...                  HydroBASINS Level-2 shapefiles
    data/raw/ClimateZone/...                 Koeppen climate zone shapefile
    data/proceed/Caravan-GRDC/<stage>/...    intermediate CSV/NetCDF outputs, one folder per pipeline stage
    figures/Caravan-GRDC/<stage>/...         diagnostic and manuscript figures, one folder per stage
    models/Caravan-GRDC/<stage>/...          trained model weights/scalers, one folder per stage
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("RFFA_DATA_ROOT", Path(__file__).resolve().parent.parent))

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCEED = REPO_ROOT / "data" / "proceed" / "Caravan-GRDC"
FIGURE_ROOT = REPO_ROOT / "figures" / "Caravan-GRDC"
MODEL_ROOT = REPO_ROOT / "models" / "Caravan-GRDC"


def stage_dir(base: Path, tag: str) -> Path:
    """Return ``base / tag``, creating it (and parents) if it does not exist."""
    p = base / tag
    p.mkdir(parents=True, exist_ok=True)
    return p
