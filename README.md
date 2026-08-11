# RFFA-Hybrid-Model

Code accompanying **"Hybrid Deep Learning for Design Flood Estimation in Ungauged Regions: Integrating Extreme Value Theory and Hydrological Knowledge"** (submitted to *Water Resources Research*).

The repository implements a hybrid deep-learning framework for regional flood frequency analysis (RFFA) that combines:

1. **Deep learning** (ANN / GEV-NN) instead of classical machine learning (RF, SVM, XGBoost), for better spatial extrapolation;
2. **Extreme value theory constraints**, embedding the GEV distribution analytically in the network so all design-flood quantiles (Q2-Q100) are derived from one set of physically constrained parameters, guaranteeing monotonic ordering;
3. **Process-based hydrological knowledge**, augmenting catchment descriptors with flow statistics simulated by a globally calibrated hydrological model (GHM; δHBV2-δMC2, Ji et al., 2025).

Models are evaluated under two spatial-validation regimes: **PUB** (Prediction in Ungauged Basins — random hold-out, spatial interpolation) and **PUR** (Prediction in Ungauged Regions — 15 geographic hold-out basins, spatial extrapolation).

## Repository layout

```
01_DataPreparation.py                  Stage 1-3: merge Caravan + GRDC-Caravan, fit at-site GEV, quality-filter stations
02_SimulatedStreamflowProcessing.py    Stage 4-5: extract GHM-simulated flow statistics, fit GEV to simulated AMS
03_SpatialSplit_PUB_PUR.py             Stage 6-7: PUR basin assignment (15 hold-out regions) and PUB random split
04_GHM_Benchmark.py                    Stage 8: observed-vs-simulated GEV comparison (standalone GHM benchmark, Sec. 3.2.4)
05_GHM_LeakageMatch.py                 Stage 9: matches PUR test stations against GHM calibration/test station lists

06_RF.py / 07_SVM.py / 08_XGBoost.py   Classical ML baselines (Sec. 3.2.3)
09_ANN.py / 10_ANN_Joint.py            ANN baselines: independent per-quantile vs. joint multi-output (Sec. 3.2.2)
11_GEV_NN.py / 12_GEV_NN_MSE.py / 13_GEV_NN_NLL.py   GEV-constrained neural network and loss-function ablations (Sec. 3.2.1)
14_RunAllModels.py               Orchestrator: runs a group of the training scripts above as subprocesses
15_SHAP.py                       SHAP feature-attribution analysis (used by Figure 6c)

Figure3_ML_Performance.py                Figure 3  - ML vs. DL performance under PUB/PUR
Figure4_GEV_Constrained_Performance.py   Figure 4  - monotonicity violations and GEV-constrained model performance
Figure5_HybridModel_Performance.py       Figure 5  - hybrid model vs. GEV-NN
Figure6_GHM_Performance.py               Figure 6  - GHM benchmark performance and SHAP feature importance
FigureS1_Monotonicity_Explanation.py     Figure S1 - log-increment distributions explaining monotonicity violations
FigureS2S3_FeatureSpace_PUB_PUR.py       Figures S2-S3 - PCA / Mahalanobis-distance covariate-shift diagnostics
FigureS4_GHM_Leakage_Analysis.py         Figure S4 - GHM calibration-leakage sensitivity analysis

src/
    paths.py             central definition of the relative directory layout (no external config file)
    dataset.py            DatasetBuilder — assembles the model input matrix from NC/CSV sources
    splits.py              pub_split / pur_splits / pur_train_val_split
    models.py               GEV-NN / ANN architectures, GEV quantile & NLL functions (torch and numpy)
    training.py             training loops for NN / RF / XGBoost / SVM
    evaluation.py           NSE / PBIAS / rRMSE metrics and per-model evaluation helpers
    _runner.py              shared PUB+PUR training/evaluation orchestration used by all 06_-13_ model-training scripts
    gev_fit_common.py       shared GEV fitting (L-moments + MLE) used by stages 2 and 5
    pur_precheck.py         standalone PUR-fold sanity-check utilities (not called by the numbered pipeline)
```

Scripts are numbered in pipeline order; run them in ascending order the first time. Each stage script writes its outputs into its own subfolder (named after the stage) so later stages and figure scripts can find them automatically — no manual path editing is required between stages.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python >= 3.10. `xgboost`, `shap`, and `geopandas`/`shapely` are only needed by the specific scripts that use them (08_XGBoost.py; 15_SHAP.py; 03_SpatialSplit_PUB_PUR.py, 04_GHM_Benchmark.py, 05_GHM_LeakageMatch.py, and the figure scripts that plot maps) — the rest of the pipeline runs without them.

## Data

No data is bundled in this repository (see `.gitignore`: `data/`, `figures/`, and `models/` are all untracked). Place the raw inputs under `data/raw/` in the layout below, relative to the repository root; every script resolves its paths automatically from `src/paths.py` (set the `RFFA_DATA_ROOT` environment variable if you want to point the whole pipeline at a different root without moving the code).

```
data/raw/Caravan/...            Caravan dataset (Kratzert et al., 2023) — https://doi.org/10.5281/zenodo.7540792
data/raw/GRDC-Caravan/...       GRDC-Caravan extension (Farber et al., 2025) — https://doi.org/10.5281/zenodo.14006282
data/raw/Sim-Dis/...            GHM-simulated streamflow (delta HBV2-delta MC2; Ji et al., 2025) — https://doi.org/10.5281/zenodo.17417150
                                 including ModelTrainingStation.csv / ModelTestStation.csv (GHM calibration/test station lists, for 05_GHM_LeakageMatch.py)
data/raw/Hydrosheds/...         HydroBASINS Level-2 shapefiles — https://www.hydrosheds.org/products/hydrobasins
data/raw/ClimateZone/...        Koeppen climate-zone shapefile (5-class merge), used by the figure scripts
```

Intermediate outputs are written to `data/proceed/Caravan-GRDC/<stage>/`, figures to `figures/Caravan-GRDC/<stage>/`, and trained models to `models/Caravan-GRDC/<stage>/`.

## Reproducing the pipeline

```bash
# 1. Data preparation
python 01_DataPreparation.py
python 02_SimulatedStreamflowProcessing.py
python 03_SpatialSplit_PUB_PUR.py
python 04_GHM_Benchmark.py
python 05_GHM_LeakageMatch.py

# 2. Model training (PUB + 15-region PUR, 3 seeds each, base and +flow feature sets)
python 14_RunAllModels.py --group all
python 15_SHAP.py

# 3. Manuscript figures
python Figure3_ML_Performance.py
python Figure4_GEV_Constrained_Performance.py
python Figure5_HybridModel_Performance.py
python Figure6_GHM_Performance.py
python FigureS1_Monotonicity_Explanation.py
python FigureS2S3_FeatureSpace_PUB_PUR.py
python FigureS4_GHM_Leakage_Analysis.py
```

`14_RunAllModels.py --group all` trains RF, SVM, XGBoost, ANN, ANN-Joint, GEV-NN, GEV-NN-MSE, and GEV-NN-NLL in sequence (`--group baseline` / `--group gev` run subsets; `--only <script...>` runs specific scripts; `--pass-arg <arg>` forwards CLI arguments such as `--seeds` to every child script). Each of the eight model scripts can also be run individually, e.g. `python 11_GEV_NN.py`.

## Correspondence between scripts and manuscript figures

| Script | Figure |
|---|---|
| `Figure3_ML_Performance.py` | Figure 3 — ML vs. DL performance under PUB and PUR |
| `Figure4_GEV_Constrained_Performance.py` | Figure 4 — monotonicity violations and GEV-constrained model performance |
| `Figure5_HybridModel_Performance.py` | Figure 5 — hybrid model vs. GEV-NN |
| `Figure6_GHM_Performance.py` | Figure 6 — GHM benchmark performance and SHAP feature importance |
| `FigureS1_Monotonicity_Explanation.py` | Figure S1 — mechanism behind monotonicity violations |
| `FigureS2S3_FeatureSpace_PUB_PUR.py` | Figures S2-S3 — attribute-space covariate-shift diagnostics |
| `FigureS4_GHM_Leakage_Analysis.py` | Figure S4 — GHM calibration-leakage sensitivity analysis |

## Notes on this release

- All scripts resolve data paths relative to the repository root via `src/paths.py`; there is no external YAML/JSON configuration file to edit.
- Every pipeline script (01-15) writes its outputs into a subfolder named after its own filename (e.g. `06_RF.py` writes to `06_RF/`), so filenames and on-disk output folders always match — the earlier `13_XGBoost`/`14_SVM` naming mismatch that existed in the original research code has been eliminated.
- `01_DataPreparation.py`, `02_SimulatedStreamflowProcessing.py`, and `03_SpatialSplit_PUB_PUR.py` each combine multiple original processing stages; internally they still write to the historical per-stage subfolder names (`00_GRDC-Caravan-Process`, `01_GEV-Fit`, `02_Data-Clean`, `03_Streamflow-Process`, `04_Sim_GEV-Fit`, `05_PUR_Basin_Select`, `06_PUB_Station_Select`) so that intermediate outputs stay individually identifiable — these sub-stage names are unrelated to the top-level script numbering above.
- `04_GHM_Benchmark.py` and `05_GHM_LeakageMatch.py` were consolidated from exploratory analysis scripts for this release; their outputs are required by `Figure6_GHM_Performance.py` and `FigureS4_GHM_Leakage_Analysis.py` respectively.
- `src/pur_precheck.py` is a standalone diagnostic utility for sanity-checking PUR fold assignments; it is not invoked by any numbered script.

## Citation

If you use this code, please cite the associated manuscript (citation details to be added upon publication).

## License

Released under the MIT License — see `LICENSE`.
