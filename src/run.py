import concurrent.futures
import subprocess

# List of scripts to run

d = {
    # "gp-vib-full": "python src/train_tsc.py "
    #                "datamodule=tsc/uea "
    #                "datamodule.dataset_name={} "
    #                "size=full "
    #                "num_labeled=1 "
    #                "model=tsc/gpvib_e_inception_d_flatten "
    #                "hparams_search=tsc/gp_vib "
    #                "callbacks=default",
    # "gp-vib-full_0.4": "python src/train_tsc.py "
    #                    "datamodule=tsc/ssl_uea "
    #                    "datamodule.dataset_name={} "
    #                    "size=full_0.4 "
    #                    "num_labeled=0.4 "
    #                    "model=tsc/gpvib_e_inception_d_flatten "
    #                    "hparams_search=tsc/gp_vib "
    #                    "callbacks=default",
    # "gp-vib-full_0.25": "python src/train_tsc.py "
    #                     "datamodule=tsc/ssl_uea "
    #                     "datamodule.dataset_name={} "
    #                     "size=full_0.25 "
    #                     "num_labeled=0.25 "
    #                     "model=tsc/gpvib_e_inception_d_flatten "
    #                     "hparams_search=tsc/gp_vib "
    #                     "callbacks=default",
    # "gp-vib-full_0.1": "python src/train_tsc.py "
    #                    "datamodule=tsc/ssl_uea "
    #                    "datamodule.dataset_name={} "
    #                    "size=full_0.1 "
    #                    "num_labeled=0.1 "
    #                    "model=tsc/gpvib_e_inception_d_flatten "
    #                    "hparams_search=tsc/gp_vib "
    #                    "callbacks=default",
    # "gp-vib-full_0.07": "python src/train_tsc.py "
    #                     "datamodule=tsc/ssl_uea "
    #                     "datamodule.dataset_name={} "
    #                     "size=full_0.07 "
    #                     "num_labeled=0.07 "
    #                     "model=tsc/gpvib_e_inception_d_flatten "
    #                     "hparams_search=tsc/gp_vib "
    #                     "callbacks=default",
    # "gp-vib-full_0.05": "python src/train_tsc.py "
    #                     "datamodule=tsc/ssl_uea "
    #                     "datamodule.dataset_name={} "
    #                     "size=full_0.05 "
    #                     "num_labeled=0.05 "
    #                     "model=tsc/gpvib_e_inception_d_flatten "
    #                     "hparams_search=tsc/gp_vib "
    #                     "callbacks=default",
    # "ss-gp-vib-0.4": "python src/train_tsc.py "
    #                  "datamodule=tsc/ssl_uea "
    #                  "datamodule.dataset_name={} "
    #                  "size=ss_0.4 "
    #                  "num_labeled=0.4 "
    #                  "model=tsc/ss_gpvib_e_inception_d_flatten "
    #                  "hparams_search=tsc/gp_vib "
    #                  "callbacks=default",
    # "ss-gp-vib_0.25": "python src/train_tsc.py "
    #                   "datamodule=tsc/ssl_uea "
    #                   "datamodule.dataset_name={} "
    #                   "size=ss_0.25 "
    #                   "num_labeled=0.25 "
    #                   "model=tsc/ss_gpvib_e_inception_d_flatten "
    #                   "hparams_search=tsc/ss_gp_vib "
    #                   "callbacks=default",
    # "ss-vib_0.1": "python src/train_tsc.py "
    #               "datamodule=tsc/ssl_uea "
    #               "datamodule.dataset_name={} "
    #               "size=ss_0.1 "
    #               "num_labeled=0.1 "
    #               "model=tsc/ss_gpvib_e_inception_d_flatten "
    #               "hparams_search=tsc/ss_gp_vib "
    #               "callbacks=default",
    # "ss-vib_0.07": "python src/train_tsc.py "
    #                "datamodule=tsc/ssl_uea "
    #                "datamodule.dataset_name={} "
    #                "size=ss_0.07 "
    #                "num_labeled=0.07 "
    #                "model=tsc/ss_gpvib_e_inception_d_flatten "
    #                "hparams_search=tsc/ss_gp_vib "
    #                "callbacks=default",
    "ss-vib-ss_0.05": "python src/train_tsc.py "
                      "datamodule=tsc/ssl_uea "
                      "datamodule.dataset_name={} "
                      "size=ss_0.05 "
                      "num_labeled=0.05 "
                      "model=tsc/ss_gpvib_e_inception_d_flatten "
                      "hparams_search=tsc/ss_gp_vib "
                      "callbacks=default "
                      "suffix=_both",
    "ss-vib-ss_0.05_only_rocon": "python src/train_tsc.py "
                                 "datamodule=tsc/ssl_uea "
                                 "datamodule.dataset_name={} "
                                 "size=ss_0.05 "
                                 "num_labeled=0.05 "
                                 "model=tsc/ss_gpvib_e_inception_d_flatten "
                                 "hparams_search=tsc/ss_gp_vib_reconstruct "
                                 "callbacks=default "
                                 "suffix=_recon",
    "ss-vib-ss_0.05_only_triplet": "python src/train_tsc.py "
                                   "datamodule=tsc/ssl_uea "
                                   "datamodule.dataset_name={} "
                                   "size=ss_0.05 "
                                   "num_labeled=0.05 "
                                   "model=tsc/ss_gpvib_e_inception_d_flatten "
                                   "hparams_search=tsc/ss_gp_vib_triplet "
                                   "callbacks=default "
                                   "suffix=_triplet"
}

s = list(d.values())

datasets = ['PenDigits', 'FaceDetection', 'Handwriting', 'LSST', 'UWaveGestureLibrary']
datasets = ['Handwriting', 'PenDigits']

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
    max_concurrent = 6  # Limit the number of concurrent processes

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
