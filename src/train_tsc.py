# this file acts as a robust starting point for launching hydra runs and multiruns
# can be run from any place
import os.path
from collections import defaultdict

import hydra
import pandas as pd
import pyrootutils
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

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

DATASETS_UEA = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'Cricket', 'DuckDuckGeese',
                'Epilepsy', 'EthanolConcentration', 'ERing', 'FaceDetection', 'FingerMovements',
                'HandMovementDirection', 'Handwriting', 'Heartbeat', 'Libras', 'LSST', 'MotorImagery', 'NATOPS',
                'PenDigits', 'PEMS-SF', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
                'StandWalkJump', 'UWaveGestureLibrary', 'EigenWorms']


@hydra.main(version_base="1.2", config_path=os.path.join(root, "configs"), config_name="train_tsc.yaml")
def main(cfg: DictConfig) -> float:
    from src.tasks.train_ts_task import evaluate
    torch.set_float32_matmul_precision('high')
    if cfg.seed:
        seeds = [cfg.seed]
    else:
        num_runs = cfg.num_runs
        seeds = [int.from_bytes(os.urandom(4), byteorder='little', signed=False) for _ in range(num_runs)]
    datamodule_name = HydraConfig.get().runtime.choices.datamodule
    is_hyper_search = 'hparams_search' in HydraConfig.get().runtime.choices.keys()
    datasets = DATASETS_UCR_2018 if datamodule_name == 'tsc/ucr' else DATASETS_UEA

    if cfg.datamodule.dataset_name:
        datasets = [cfg.datamodule.dataset_name]

    for dataset_name in datasets:
        dataset_metric_dict = defaultdict(list)
        for i, seed in enumerate(seeds):
            temp_conf = cfg.copy()

            temp_conf['seed'] = seed
            run_dict = evaluate(temp_conf, dataset_name)
            dataset_metric_dict['dataset'].append(dataset_name)

            for k, v in run_dict.items():
                dataset_metric_dict[k].append(float(v))
            if is_hyper_search:
                return run_dict['val_ACC']
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


if __name__ == "__main__":
    main()
