# this file acts as a robust starting point for launching hydra runs and multiruns
# can be run from any place
import os.path
from collections import defaultdict

import hydra
import pandas as pd
import pyrootutils
import torch
from omegaconf import DictConfig

from utils import get_metric_value

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

DATASETS_UCR_2018 = ["AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ", "ArrowHead", "BME", "Car", "CBF",
                     "Chinatown", "ChlorineConcentration", "CinCECGTorso", "Computers", "CricketX", "CricketY",
                     "CricketZ", "Crop", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup",
                     "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "DodgerLoopGame", "DodgerLoopWeekend",
                     "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices", "EOGHorizontalSignal",
                     "EOGVerticalSignal", "EthanolLevel", "FaceAll", "FacesUCR", "Fish", "FordA", "FordB",
                     "FreezerRegularTrain", "FreezerSmallTrain", "GesturePebbleZ1", "GesturePebbleZ2", "GunPoint",
                     "GunPointAgeSpan", "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "Ham", "HandOutlines",
                     "Haptics", "Herring", "HouseTwenty", "InlineSkate", "InsectEPGRegularTrain", "InsectEPGSmallTrain",
                     "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Mallat",
                     "Meat", "MedicalImages", "MelbournePedestrian", "MiddlePhalanxOutlineAgeGroup",
                     "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MixedShapesRegularTrain",
                     "MixedShapesSmallTrain", "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2",
                     "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme", "PLAID", "Plane", "PowerCons",
                     "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW",
                     "RefrigerationDevices", "ScreenType", "SemgHandGenderCh2", "SemgHandMovementCh2",
                     "SemgHandSubjectCh2", "ShapeletSim", "SmallKitchenAppliances", "SmoothSubspace",
                     "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf",
                     "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG",
                     "TwoPatterns", "UMD", "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY",
                     "UWaveGestureLibraryZ", "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"]

IS_HYPER_SEARCH = False


@hydra.main(version_base="1.2", config_path=os.path.join(root, "configs"), config_name="train_tsc.yaml")
def main(cfg: DictConfig) -> float:
    # imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.tasks.train_ts_task import evaluate

    torch.set_float32_matmul_precision('high')
    num_runs = 1 if cfg.get("seed") else 1
    seeds = [int.from_bytes(os.urandom(4), byteorder='little', signed=False) for _ in range(num_runs)]
    for dataset_name in [DATASETS_UCR_2018[3]]:
        dataset_name = "SyntheticControl"
        dataset_metric_dict = defaultdict(list)
        for i, seed in enumerate(seeds):
            temp_conf = cfg.copy()
            temp_conf['seed'] = seed
            run_dict = evaluate(temp_conf, dataset_name)
            dataset_metric_dict['dataset'].append(dataset_name)

            for k, v in run_dict.items():
                dataset_metric_dict[k].append(float(v))
            if IS_HYPER_SEARCH:
                return run_dict['test_Accuracy']
            df = pd.DataFrame.from_dict(dataset_metric_dict).set_index(['seed', 'dataset'])
            median = df.groupby('dataset').median()
            median.index = pd.MultiIndex.from_tuples(list(map(lambda x: ('median', x), median.index)))
            mean = df.groupby('dataset').mean()
            mean.index = pd.MultiIndex.from_tuples(list(map(lambda x: ('mean', x), mean.index)))
            std = df.groupby('dataset').std()
            std.index = pd.MultiIndex.from_tuples(list(map(lambda x: ('std', x), std.index)))
            df = pd.concat([df, median], axis=0)
            df = pd.concat([df, mean], axis=0)
            df = pd.concat([df, std], axis=0)
            df.index.names = ['seed', 'dataset']
            df.to_csv(os.path.join(cfg.paths.output_dir, f'results_{dataset_name}.csv'))
        return get_metric_value(metric_dict=run_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
