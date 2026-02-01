import wandb
from collections import defaultdict
from tqdm import tqdm

PROJECT = "ss_gp_vib"
GROUP = "hmnist_missing_top"

if __name__ == "__main__":
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={"group": GROUP})

    params_by_tag = defaultdict(list)

    for run in tqdm(runs):
        tag = run.metadata["args"][-2]
        if tag in params_by_tag.keys(): continue
        cfg = run.config
        params_by_tag[tag].append({
            "beta": cfg.get("model/beta"),
            "data_beta": cfg.get("model/data_beta"),
            "entropy_coef": cfg.get("model/entropy_coef"),
            "reconstruction_coef": cfg.get("model/reconstruction_coef"),
        })

    for tag, params in params_by_tag.items():
        print(f"\nTag: {tag}")
        for i, p in enumerate(params):
            print(
                f"  Run {i}: "
                f"beta={p['beta']}, "
                f"data_beta={p['data_beta']}, "
                f"entropy_coef={p['entropy_coef']}, "
                f"reconstruction_coef={p['reconstruction_coef']}"
            )
