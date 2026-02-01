# this file acts as a robust starting point for launching hydra runs and multiruns
# can be run from any place
import os

import hydra
import pyrootutils
import pandas as pd

from omegaconf import DictConfig

# project root setup
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


def _get_results_filename(cfg: DictConfig) -> str:
    """
    Build CSV filename based on dataset type and num_label.
    """
    datamodule = cfg.datamodule._target_ if "_target_" in cfg.datamodule else str(cfg.datamodule)

    # full dataset
    if "hmnist/hmnist" in datamodule:
        name = "full"
    else:
        num_label = cfg.datamodule.num_label
        name = f"ss_{num_label}"

    return f"results_{name}.csv"


@hydra.main(version_base="1.2", config_path=os.path.join(root, "configs"), config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    from src.tasks.eval_task import evaluate

    metric_dict = {}
    num_runs = 1 if cfg.get("seed") else 1

    for i in range(num_runs):
        metric_dict[i] = evaluate(cfg.copy())

    new_dict = {"run": list(range(1, num_runs + 1))}
    for k in metric_dict[0].keys():
        new_dict[k] = [float(v[k]) for v in metric_dict.values()]

    df = pd.DataFrame.from_dict(new_dict).set_index("run")

    mean = df.drop(columns="seed", errors="ignore").mean()
    mean["seed"] = "-"
    mean.name = "mean"

    std = df.drop(columns="seed", errors="ignore").std()
    std["seed"] = "-"
    std.name = "std"

    df = pd.concat([df, mean.to_frame().T, std.to_frame().T])

    results_dir = os.path.join(root, "results")
    os.makedirs(results_dir, exist_ok=True)

    filename = _get_results_filename(cfg)
    df.to_csv(os.path.join(results_dir, filename))


if __name__ == "__main__":
    main()
