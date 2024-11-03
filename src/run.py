import concurrent.futures
import subprocess

# List of scripts to run
gpu = [1]
full_run = {
    "gp-vib-full": "python src/train_tsc.py "
                   "datamodule=tsc/ucr "
                   "datamodule.dataset_name={} "
                   "size=full "
                   "num_labeled=1 "
                   "model=tsc/gpvib_e_inception_d_flatten "
                   "hparams_search=tsc/gp_vib "
                   f"trainer.devices={gpu} "
                   "callbacks=default",
    "gp-vib-full_0.4": "python src/train_tsc.py "
                       "datamodule=tsc/ssl_ucr "
                       "datamodule.dataset_name={} "
                       "size=full_0.4 "
                       "num_labeled=0.4 "
                       "model=tsc/gpvib_e_inception_d_flatten "
                       "hparams_search=tsc/gp_vib "
                       f"trainer.devices={gpu} "
                       "callbacks=default",
    "gp-vib-full_0.25": "python src/train_tsc.py "
                        "datamodule=tsc/ssl_ucr "
                        "datamodule.dataset_name={} "
                        "size=full_0.25 "
                        "num_labeled=0.25 "
                        "model=tsc/gpvib_e_inception_d_flatten "
                        "hparams_search=tsc/gp_vib "
                        f"trainer.devices={gpu} "
                        "callbacks=default",
    "gp-vib-full_0.1": "python src/train_tsc.py "
                       "datamodule=tsc/ssl_ucr "
                       "datamodule.dataset_name={} "
                       "size=full_0.1 "
                       "num_labeled=0.1 "
                       "model=tsc/gpvib_e_inception_d_flatten "
                       "hparams_search=tsc/gp_vib "
                       f"trainer.devices={gpu} "
                       "callbacks=default",
    "gp-vib-full_0.05": "python src/train_tsc.py "
                        "datamodule=tsc/ssl_ucr "
                        "datamodule.dataset_name={} "
                        "size=full_0.05 "
                        "num_labeled=0.05 "
                        "model=tsc/gpvib_e_inception_d_flatten "
                        "hparams_search=tsc/gp_vib "
                        f"trainer.devices={gpu} "
                        "callbacks=default",
    "gp-vib-full_0.01": "python src/train_tsc.py "
                        "datamodule=tsc/ssl_ucr "
                        "datamodule.dataset_name={} "
                        "size=full_0.01 "
                        "num_labeled=0.01 "
                        "model=tsc/gpvib_e_inception_d_flatten "
                        "hparams_search=tsc/gp_vib "
                        f"trainer.devices={gpu} "
                        "callbacks=default",
}

ssl_run = {
    # "ss-gp-vib-0.4": "python src/train_tsc.py "
    #                  "datamodule=tsc/ssl_uea "
    #                  "datamodule.dataset_name={} "
    #                  "size=ss_0.4 "
    #                  "num_labeled=0.4 "
    #                  "model=tsc/ss_gpvib_e_inception_d_flatten "
    #                  "hparams_search=tsc/ss_gp_vib_reconstruct "
    #                  f"trainer.devices={gpu} "
    #                  "callbacks=default",
    # "ss-gp-vib_0.25": "python src/train_tsc.py "
    #                   "datamodule=tsc/ssl_uea "
    #                   "datamodule.dataset_name={} "
    #                   "size=ss_0.25 "
    #                   "num_labeled=0.25 "
    #                   "model=tsc/ss_gpvib_e_inception_d_flatten "
    #                   "hparams_search=tsc/ss_gp_vib_reconstruct "
    #                   f"trainer.devices={gpu} "
    #                   "callbacks=default",
    # "ss-vib_0.1": "python src/train_tsc.py "
    #               "datamodule=tsc/ssl_uea "
    #               "datamodule.dataset_name={} "
    #               "size=ss_0.1 "
    #               "num_labeled=0.1 "
    #               "model=tsc/ss_gpvib_e_inception_d_flatten "
    #               "hparams_search=tsc/ss_gp_vib_reconstruct "
    #               f"trainer.devices={gpu} "
    #               "callbacks=default",
    "ss-vib_0.05": "python src/train_tsc.py "
                   "datamodule=tsc/ssl_uea "
                   "datamodule.dataset_name={} "
                   "size=ss_0.05 "
                   "num_labeled=0.05 "
                   "model=tsc/ss_gpvib_e_inception_d_flatten "
                   "hparams_search=tsc/ss_gp_vib_reconstruct "
                   f"trainer.devices={gpu} "
                   "callbacks=default",
    # "ss-vib-ss_0.01": "python src/train_tsc.py "
    #                   "datamodule=tsc/ssl_uea "
    #                   "datamodule.dataset_name={} "
    #                   "size=ss_0.01 "
    #                   "num_labeled=0.01 "
    #                   "model=tsc/ss_gpvib_e_inception_d_flatten "
    #                   "hparams_search=tsc/ss_gp_vib_reconstruct "
    #                   f"trainer.devices={gpu} "
    #                   "callbacks=default"
}


full_har = {
    "gp-vib-full": "python src/train_tsc.py "
                   "datamodule=har/har "
                   "datamodule.dataset_name=HAR "
                   "size=full "
                   "num_labeled=1 "
                   "model=tsc/gpvib_e_inception_d_flatten "
                   "hparams_search=tsc/gp_vib "
                   f"trainer.devices={gpu} "
                   "callbacks=default",
    "gp-vib-full_0.4": "python src/train_tsc.py "
                       "datamodule=har/ssl_har "
                       "datamodule.dataset_name=HAR "
                       "size=full_0.4 "
                       "num_labeled=0.4 "
                       "model=tsc/gpvib_e_inception_d_flatten "
                       "hparams_search=tsc/gp_vib "
                       f"trainer.devices={gpu} "
                       "callbacks=default",
    "gp-vib-full_0.25": "python src/train_tsc.py "
                        "datamodule=har/ssl_har "
                        "datamodule.dataset_name=HAR "
                        "size=full_0.25 "
                        "num_labeled=0.25 "
                        "model=tsc/gpvib_e_inception_d_flatten "
                        "hparams_search=tsc/gp_vib "
                        f"trainer.devices={gpu} "
                        "callbacks=default",
    "gp-vib-full_0.1": "python src/train_tsc.py "
                       "datamodule=har/ssl_har "
                       "datamodule.dataset_name=HAR "
                       "size=full_0.1 "
                       "num_labeled=0.1 "
                       "model=tsc/gpvib_e_inception_d_flatten "
                       "hparams_search=tsc/gp_vib "
                       f"trainer.devices={gpu} "
                       "callbacks=default",
    "gp-vib-full_0.05": "python src/train_tsc.py "
                        "datamodule=har/ssl_har "
                        "datamodule.dataset_name=HAR "
                        "size=full_0.05 "
                        "num_labeled=0.05 "
                        "model=tsc/gpvib_e_inception_d_flatten "
                        "hparams_search=tsc/gp_vib "
                        f"trainer.devices={gpu} "
                        "callbacks=default",
    "gp-vib-full_0.01": "python src/train_tsc.py "
                        "datamodule=har/ssl_har "
                        "datamodule.dataset_name=HAR "
                        "size=full_0.01 "
                        "num_labeled=0.01 "
                        "model=tsc/gpvib_e_inception_d_flatten "
                        "hparams_search=tsc/gp_vib "
                        f"trainer.devices={gpu} "
                        "callbacks=default",
    }

ssl_har = {
    "ss-gp-vib_0.4": "python src/train_tsc.py "
                     "datamodule=har/ssl_har "
                     "datamodule.dataset_name=HAR "
                     "size=ss_0.4 "
                     "num_labeled=0.4 "
                     "model=tsc/ss_gpvib_e_inception_d_flatten "
                     "hparams_search=tsc/ss_gp_vib_reconstruct "
                     f"trainer.devices={gpu} "
                     "callbacks=default",
    "ss-gp-vib_0.25": "python src/train_tsc.py "
                      "datamodule=har/ssl_har "
                      "datamodule.dataset_name=HAR "
                      "size=ss_0.25 "
                      "num_labeled=0.25 "
                      "model=tsc/ss_gpvib_e_inception_d_flatten "
                      "hparams_search=tsc/ss_gp_vib_reconstruct "
                      f"trainer.devices={gpu} "
                      "callbacks=default",
    "ss-vib-0.1": "python src/train_tsc.py "
                  "datamodule=har/ssl_har "
                  "datamodule.dataset_name=HAR "
                  "size=ss_0.1 "
                  "num_labeled=0.1 "
                  "model=tsc/ss_gpvib_e_inception_d_flatten "
                  "hparams_search=tsc/ss_gp_vib_reconstruct "
                  f"trainer.devices={gpu} "
                  "callbacks=default",
    "ss_gp-vib_0.05": "python src/train_tsc.py "
                      "datamodule=har/ssl_har "
                      "datamodule.dataset_name=HAR "
                      "size=ss_0.05 "
                      "num_labeled=0.05 "
                      "model=tsc/ss_gpvib_e_inception_d_flatten "
                      "hparams_search=tsc/ss_gp_vib_reconstruct "
                      f"trainer.devices={gpu} "
                      "callbacks=default",
    "ss-gp-vib_0.01": "python src/train_tsc.py "
                      "datamodule=har/ssl_har "
                      "datamodule.dataset_name=HAR "
                      "size=ss_0.01 "
                      "num_labeled=0.01 "
                      "model=tsc/ss_gpvib_e_inception_d_flatten "
                      "hparams_search=tsc/ss_gp_vib_reconstruct "
                      f"trainer.devices={gpu} "
                      "callbacks=default"
}



s = list(ssl_run.values())

# datasets = ['Wafer', 'FordA', 'FordB']
datasets = ['UWaveGestureLibrary']
scripts = []

for dataset in datasets:
    for sc in s:
        scripts.append(sc.format(dataset))


# Function to run a script


def run_script(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    print(f"Completed: {command} with return code {result.returncode}")
    return result.returncode


# Main execution
def main():
    max_concurrent = 1  # Limit the number of concurrent processes

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
    main()
