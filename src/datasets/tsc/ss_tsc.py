import os
import zipfile
import numpy as np
import torch

from hydra.utils import to_absolute_path
from aeon.datasets import load_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch import tensor
from torch.utils.data import TensorDataset

from datasets.ssl_dataset import SemiDataModule


class SemiSupervisedTSC(SemiDataModule):
    def __init__(
            self,
            file_name,
            download_dir,
            download_url,
            is_data_missing,
            num_workers,
            batch_size,
            num_labeled,
            num_val,
            validation_split_seed,
            is_ssl,
            dataset_name,
            cross_val_split_index
    ):

        super(SemiSupervisedTSC, self).__init__(
            download_dir,
            num_workers,
            batch_size,
            num_labeled,
            num_val,
            validation_split_seed,
            is_ssl
        )
        self.train_dataset = None
        self.test_dataset = None
        self.output_size = None
        self.time_series_size = None
        self.channels = None
        self.file_path = None
        self.train_size = None
        self.dataset_name = dataset_name
        self.cross_val_split_index = cross_val_split_index
        self.save_hyperparameters()

    def _download_file(self, url, local_file_path):
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        import requests
        response = requests.get(url, stream=True, headers=headers)
        content_length = int(response.headers['Content-Length'])
        from tqdm import tqdm
        pbar = tqdm(total=content_length)
        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=max(20_000_000, content_length)):
                if chunk:
                    f.write(chunk)
                pbar.update(len(chunk))

    @staticmethod
    def _transform_labels(y_train, y_test):
        """
        Transform label to min equal zero and continuous
        For example if we have [1,3,4] --->  [0,1,2]
        """
        # no validation split
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_test = np.concatenate((y_train, y_test), axis=0)
        # fit the encoder
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels
        new_y_train_test = encoder.transform(y_train_test)
        # re-split the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test

    def _load_data(self):
        x_train, y_train = load_classification(extract_path=os.path.join(self.file_path, self.dataset_name),
                                               name=self.dataset_name,
                                               return_metadata=False, split='train')
        x_test, y_test = load_classification(extract_path=os.path.join(self.file_path, self.dataset_name),
                                             name=self.dataset_name,
                                             return_metadata=False, split='test')
        y_train, y_test = self._transform_labels(y_train, y_test)
        return x_train, x_test, y_train, y_test

    def _build_splits(self, X, y):
        np.random.seed(self.validation_split_seed)
        skf = StratifiedKFold(5, shuffle=True)
        split_indices = list(skf.split(X, y))

        raw_index, test_index = split_indices[self.cross_val_split_index]

        raw_set = X[raw_index]
        raw_target = y[raw_index]

        train_index, val_index = next(StratifiedKFold(4, shuffle=True).split(raw_set, raw_target))
        train = raw_set[train_index]
        train_targets = raw_target[train_index]

        val = raw_set[val_index]
        val_targets = raw_target[val_index]

        test = X[test_index]
        test_targets = y[test_index]

        self.n_classes = len(np.unique(test_targets))

        self.build_datasets(train, train_targets, val, val_targets, test, test_targets)

        return train.shape[0], train.shape[-2] if len(train.shape) > 2 else 1, train.shape[-1]

    def build_datasets(self, train, train_targets, val, val_targets, test, test_targets):
        train_tensors = []
        val_tensors = []
        test_tensors = []
        train_tensors.append(tensor(train, dtype=torch.float))
        val_tensors.append(tensor(val, dtype=torch.float))
        test_tensors.append(tensor(test, dtype=torch.float))
        train_tensors.append(tensor(train_targets, dtype=torch.long))
        val_tensors.append(tensor(val_targets, dtype=torch.long))
        test_tensors.append(tensor(test_targets, dtype=torch.long))
        self.train_set = TensorDataset(*train_tensors)
        self.val_set = TensorDataset(*val_tensors)
        self.test_set = TensorDataset(*test_tensors)

    def prepare_data(self):
        if not os.path.exists(to_absolute_path(self.hparams.download_dir)):
            os.makedirs(to_absolute_path(self.hparams.download_dir))
        self.file_path = os.path.join(to_absolute_path(self.hparams.download_dir), self.hparams.file_name)
        if not os.path.exists(self.file_path):
            self._download_file(self.hparams.download_url, self.file_path)

            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                zip_ref.extractall(self.hparams.download_dir)
        with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
            self.file_path = os.path.join(self.hparams.download_dir, zip_ref.filelist[0].filename)
        x_train, x_test, y_train, y_test = self._load_data()
        X = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])

        X = X.transpose(0, 2, 1)

        self.train_size, self.time_series_size, self.channels = self._build_splits(X, y)
        # self.shape = x_train.shape
        # self.n_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))


if __name__ == '__main__':
    ucr = SemiSupervisedTSC(download_dir=r'C:\Developments\GP-VIB\data\tsc\ucr',
                            download_url='https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip',
                            file_name='Univariate2018_ts.zip',
                            batch_size=64,
                            num_workers=1,
                            dataset_name='Worms',
                            validation_split_seed=42,
                            is_data_missing=False,
                            num_labeled=0.4,
                            num_val=0,
                            cross_val_split_index=0,
                            is_ssl=True)

    ucr.prepare_data()
    ucr.setup()

    t_d = ucr.train_dataloader()
    for batch in t_d:
        print(1)
        break
