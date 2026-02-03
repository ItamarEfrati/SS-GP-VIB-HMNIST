import os
import subprocess
import optuna
import pandas as pd
from pathlib import Path
import argparse
from pyrootutils import pyrootutils

from hyper_parameters_search import search_hyper_parameters

# Setup root
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
os.chdir(root)

RESULTS_DIR = 'results'  # Optuna DB folder
RUNS_DIR = Path("logs")  # Logs folder

# CSV paths
CSV_NO_MISSING = Path("best_hyperparameters/hmnist_full_top.csv")
CSV_MISSING = Path("best_hyperparameters/hmnist_missing_top.csv")

# Datamodule + size combinations
COMBINATIONS = [
    ("ss_hmnist", "full_0.01", "0.01"),
    ("ss_hmnist", "full_0.02", "0.02"),
    ("ss_hmnist", "full_0.1", "0.1"),
    ("ss_hmnist", "full_0.2", "0.2"),
    ("hmnist", "full", "1"),
    ("ss_hmnist", "ss_0.01", "0.01"),
    ("ss_hmnist", "ss_0.02", "0.02"),
    ("ss_hmnist", "ss_0.1", "0.1"),
    ("ss_hmnist", "ss_0.2", "0.2"),
]


def load_hparams_csv(size: str, is_data_missing: bool) -> dict:
    """Load hyperparameters from CSV depending on is_data_missing flag."""
    csv_path = CSV_MISSING if is_data_missing else CSV_NO_MISSING
    df = pd.read_csv(csv_path).set_index("size")

    if size not in df.index:
        raise ValueError(f"Size '{size}' not found in CSV {csv_path}")

    row = df.loc[size]
    return {
        "model.beta": float(row["beta"]),
        "model.data_beta": float(row["data_beta"]),
        "model.entropy_coef": float(row["entropy_coef"]),
        "model.reconstruction_coef": float(row["reconstruction_coef"]),
    }


def load_hparams_study(datamodule: str, size: str, model: str, is_data_missing: bool) -> dict:
    """Load best hyperparameters from Optuna study."""
    db_name = f"{datamodule}_{model}_{size}_is_data_missing_{is_data_missing}.db"
    db_path = RESULTS_DIR + '/' + db_name
    study_name = f"{datamodule}_{model}_{size}_is_data_missing_{is_data_missing}"
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
    return study.best_params


def run_eval(datamodule: str, size: str, model: str, num_labeled: float, is_data_missing: bool, use_csv: bool):
    """Run train.py with hyperparameters loaded either from CSV or Optuna."""
    if use_csv:
        hparams = load_hparams_csv(size, is_data_missing)
    else:
        hparams = load_hparams_study(datamodule, size, model, is_data_missing)

    cmd = [
        "python", "src/eval.py",
        f"datamodule={datamodule}",
        f"size={size}",
        f"model={model}",
        f"num_labeled={num_labeled}",
        f"datamodule.is_data_missing={is_data_missing}"
    ]

    # Add hyperparameters
    for k, v in hparams.items():
        cmd.append(f"{k}={v}")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run GP-VIB experiments with or without hparams search")
    parser.add_argument(
        "--hparams_search",
        action="store_true",
        help="Run hyperparameter search before evaluation",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="Load best hyperparameters from CSV instead of Optuna studies",
    )
    args = parser.parse_args()

    if args.hparams_search:
        print("Running hyperparameter search for all combinations...")
        search_hyper_parameters()

    for datamodule, size, num_labeled in COMBINATIONS:
        model = "ss-gp-vib" if size.startswith("ss_") else "gp-vib"

        # Run twice: is_data_missing False and True
        for is_missing in [False, True]:
            run_eval(datamodule, size, model, num_labeled, is_missing, use_csv=args.use_csv)


if __name__ == "__main__":
    main()
