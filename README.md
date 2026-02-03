# GPredicting Chronic Pain States from Smartphone-Derived Digital Biomarkers Using Semi-Supervised GP-VIB

This repository contains the code to **reproduce the experiments from a paper currently under review**, titled:

**"Predicting Chronic Pain States from Smartphone-Derived Digital Biomarkers Using Semi-Supervised GP-VIB"**

The code implements **Variational Information Bottleneck (VIB) with Gaussian Processes** for time-series classification, supporting both supervised and semi-supervised settings.

**Note:** This repository is anonymized for the review process. Author names and affiliations are removed.

## Features

- Implementation of **GP-VIB** and **Semi-Supervised GP-VIB (SS-GP-VIB)** models.
- Hyperparameter optimization via **Optuna** integrated with **Hydra**.
- Evaluation using **best hyperparameters** from CSV or Optuna studies.
- Experiment logs organized under `logs/` and results under `results/`.

## Repository Structure

```
.
├── src/
│   ├── train.py                    # Training script
│   ├── eval.py                     # Evaluation with best hyperparameters
│   ├── main.py                     # Orchestrates hyperparameter search and evaluation
│   └── hyper_parameters_search.py  # Function for running Optuna sweeps
├── results/                        # Optuna databases and CSVs with best hyperparameters
├── logs/                           # Logs for all experiment runs
├── configs/                        # Hydra configuration files
├── best_hyperparameters/           # Containing csvs with best hyperparameters
└── README.md
```

## Installation

```bash
# Create env
conda create -n ss_gp_vib python=3.10 -y
conda activate ss_gp_vib

# Always upgrade pip first
pip install --upgrade pip setuptools wheel

# Install deps
pip install -r requirements.txt
```

## Usage

### 1. Run Hyperparameter Search and Evaluation

The main script `src/main.py` supports running hyperparameter search, evaluation, or both:

```bash
# Run hyperparameter search for all dataset/model combinations, then evaluate
python src/main.py --hparams_search

# Run evaluation only using CSV with precomputed best hyperparameters
python src/main.py --use_csv
```

### 2. Configuration Options

- **Datasets:** `hmnist` and semi-supervised subsets (`ss_hmnist`).
- **Models:** `gp-vib` for full datasets, `ss-gp-vib` for semi-supervised subsets.
- **Hyperparameters:** Can be loaded from **Optuna database** or **CSV files**. Controlled by `datamodule`, `size`, `num_labeled`, and `is_data_missing`.

### 3. Logging and Results

- **Logs:** stored under `logs/`.
- **Results / Optuna databases:** stored under `results/`.
- Each run creates separate folders for clarity.
