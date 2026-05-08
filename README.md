# Adaptive Doubly Robust Proxy Causal Learning

This repository contains the code used for an anonymous NeurIPS 2026 submission on adaptive doubly robust proxy causal learning. The submitted package focuses on the source code and synthetic simulation pipeline needed to reproduce the paper's synthetic results.

## Repository Contents

- `src/`: implementation of the proposed doubly robust proxy causal learning neural network together with the benchmark methods used in the paper. 
- `Simulations/DRNMEPCL/`: main simulation scripts for the proposed method.
- `Simulations/DRKPV/` and `Simulations/PKDR/`: benchmark simulation scripts.
- `Simulations/AnalyzeSimulationResults/`: Jupyter notebooks used to aggregate CSV outputs and generate the figures/tables. This folder also includes the ablation-study analyses.
- `Simulations/Results/`: output directory where the simulation scripts write CSV files. Precomputed CSV files are not included in the submission.

## Setup

Use Python 3.12 and install the required packages:

```bash
pip install -r requirements.txt
```

## Reproducing Synthetic Results

Run the simulation scripts from inside the corresponding simulation folder.

### Example 1: Synthetic low-dimensional experiment

```bash
cd Simulations/DRNMEPCL
python Synthetic_Low_Dim_Experiment_w_SGD_log_cosh.py
```

This produces a CSV file under `Simulations/Results/Synthetic_Low_Dim_Experiment/`. Then go to `Simulations/AnalyzeSimulationResults/1-SyntheticLowDim_Experiment.ipynb` for visualization and summary plots.

### Example 2: Synthetic high-dimensional experiment

```bash
cd Simulations/DRNMEPCL
python Synthetic_High_Dim_Experiment_w_SGD_log_cosh_V2.py --size 2000
```

This produces a CSV file under `Simulations/Results/Synthetic_High_Dim_Experiment_NewVersion/`. Repeat similarly for the other sample sizes used in the paper, then use `Simulations/AnalyzeSimulationResults/2-SyntheticHighDim_NewVersion_Experiment.ipynb`.

Other simulations follow the same pattern. For example:

- `python Synthetic_CATE_Experiment_w_SGD_log_cosh.py` writes to `Simulations/Results/CATE_Synthetic_Experiment/`; analyze with `Simulations/AnalyzeSimulationResults/4-Synthetic_CATE_Experiment.ipynb`.
- `python run_proxy_misspecification_six_settings.py --data_size 2000` writes to `Simulations/Results/Proxy_Misspecification_6Settings/`; analyze with `Simulations/AnalyzeSimulationResults/6-Misspecification_Analysis.ipynb`.
- Benchmark scripts in `Simulations/DRKPV/` and `Simulations/PKDR/` can be run analogously, and their CSV outputs are analyzed through the notebooks in `Simulations/AnalyzeSimulationResults/`.
