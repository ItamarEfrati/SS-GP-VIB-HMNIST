import concurrent.futures
import subprocess

# List of scripts to run
scripts = [
    # ss_hmnist semi supervised
    # "python src/train.py datamodule=ss_hmnist size=ss_0.01 num_labeled=0.01 model=ss-gp-vib hparams_search=ss-gp-vib datamodule.is_data_missing=False",
    # "python src/train.py datamodule=ss_hmnist size=ss_0.01 num_labeled=0.01 model=ss-gp-vib hparams_search=ss-gp-vib datamodule.is_data_missing=True",

    # "python src/train.py datamodule=ss_hmnist size=ss_0.02 num_labeled=0.02 model=ss-gp-vib hparams_search=ss-gp-vib datamodule.is_data_missing=False",
    # "python src/train.py datamodule=ss_hmnist size=ss_0.02 num_labeled=0.02 model=ss-gp-vib hparams_search=ss-gp-vib datamodule.is_data_missing=True",
    #
    # "python src/train.py datamodule=ss_hmnist size=ss_0.1 num_labeled=0.1 model=ss-gp-vib hparams_search=ss-gp-vib datamodule.is_data_missing=False",
    # "python src/train.py datamodule=ss_hmnist size=ss_0.1 num_labeled=0.1 model=ss-gp-vib hparams_search=ss-gp-vib datamodule.is_data_missing=True",
    #
    # "python src/train.py datamodule=ss_hmnist size=ss_0.2 num_labeled=0.2 model=ss-gp-vib hparams_search=ss-gp-vib datamodule.is_data_missing=False",
    # "python src/train.py datamodule=ss_hmnist size=ss_0.2 num_labeled=0.2 model=ss-gp-vib hparams_search=ss-gp-vib datamodule.is_data_missing=True",

    # ss_hmnist full sizes (gp-vib)
    # "python src/train.py datamodule=ss_hmnist size=full_0.01 num_labeled=0.01 model=gp-vib hparams_search=gp-vib datamodule.is_data_missing=False",
    # "python src/train.py datamodule=ss_hmnist size=full_0.01 num_labeled=0.01 model=gp-vib hparams_search=gp-vib datamodule.is_data_missing=True",

    "python src/train.py datamodule=ss_hmnist size=full_0.02 num_labeled=0.02 model=gp-vib hparams_search=gp-vib datamodule.is_data_missing=False",
    "python src/train.py datamodule=ss_hmnist size=full_0.02 num_labeled=0.02 model=gp-vib hparams_search=gp-vib datamodule.is_data_missing=True",

    # "python src/train.py datamodule=ss_hmnist size=full_0.1 num_labeled=0.1 model=gp-vib hparams_search=gp-vib datamodule.is_data_missing=False",
    # "python src/train.py datamodule=ss_hmnist size=full_0.1 num_labeled=0.1 model=gp-vib hparams_search=gp-vib datamodule.is_data_missing=True",

    "python src/train.py datamodule=ss_hmnist size=full_0.2 num_labeled=0.2 model=gp-vib hparams_search=gp-vib datamodule.is_data_missing=False",
    "python src/train.py datamodule=ss_hmnist size=full_0.2 num_labeled=0.2 model=gp-vib hparams_search=gp-vib datamodule.is_data_missing=True",

    # hmnist full size
    # "python src/train.py datamodule=hmnist size=full model=gp-vib hparams_search=gp-vib datamodule.is_data_missing=False",
    # "python src/train.py datamodule=hmnist size=full model=gp-vib hparams_search=gp-vib datamodule.is_data_missing=True",
]


# Function to run a script
def run_script(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    print(f"Completed: {command} with return code {result.returncode}")
    return result.returncode


# Main execution
def search_hyper_parameters():
    max_concurrent = 2  # Limit the number of concurrent processes

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(run_script, script): script for script in scripts}

        # Wait for the results as they complete
        for future in concurrent.futures.as_completed(futures):
            script = futures[future]
            try:
                return_code = future.result()
                if return_code != 0:
                    print(f"Script {script} failed with return code {return_code}")
            except Exception as e:
                print(f"Script {script} generated an exception: {e}")


if __name__ == "__main__":
    search_hyper_parameters()
