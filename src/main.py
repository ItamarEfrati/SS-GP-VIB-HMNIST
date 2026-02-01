from pathlib import Path
import pandas as pd
import subprocess

def run_eval_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        overrides = []

        for col, val in row.items():
            if pd.isna(val):
                continue

            # skip size for now (handled separately)
            if col == "size":
                continue

            overrides.append(f"model.{col}={val}")

        # ---- handle size logic ----
        size = row["size"]

        # Add datamodule_name as the raw size string
        overrides.append(f"datamodule_name={size}")

        if size == "full":
            overrides.append("datamodule=hmnist/hmnist")
        else:
            overrides.append("datamodule=hmnist/ss_hmnist")

            # extract percentage (e.g. full_0.01 → 0.01, ss_0.2 → 0.2)
            try:
                num_label = float(size.split("_")[-1])
            except ValueError:
                raise ValueError(f"Cannot parse num_label from size='{size}'")

            overrides.append(f"datamodule.num_labeled={num_label}")
            overrides.append(f"num_labeled={num_label}")

        cmd = [
            "python",
            "src/eval.py",
            *overrides
        ]

        print(f"\nRunning eval for row {idx}")
        print("Command:", " ".join(cmd))

        subprocess.run(cmd, check=True)
