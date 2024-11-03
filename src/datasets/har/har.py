import os

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pytorch_lightning as pl

from datasets.ssl_dataset import Subset, CustomSemiDataset, SemiDataModule


class HARDataModule(SemiDataModule):
    def __init__(self, download_dir, batch_size, num_labeled, dataset_name, num_workers=4, n_classes=6, num_val=1,
                 validation_split_seed=1, is_ssl=True, seed=None):
        super(HARDataModule, self).__init__(
            download_dir,
            num_workers,
            batch_size,
            num_labeled,
            num_val,
            validation_split_seed,
            is_ssl
        )
        self.dataset_name = dataset_name
        self.data_path = download_dir
        self.batch_size = batch_size
        self.num_labeled = num_labeled
        self.num_workers = num_workers
        self.n_classes = n_classes
        self.seed = seed
        self.channels = 3
        self.time_series_size = 200

    def prepare_data(self):
        # Load datasets
        train_data = torch.load(os.path.join(self.data_path, 'train.pt'))
        val_data = torch.load(os.path.join(self.data_path, 'val.pt'))
        test_data = torch.load(os.path.join(self.data_path, 'test.pt'))

        # Convert samples and labels to FloatTensor
        train_x = torch.tensor(train_data['samples'], dtype=torch.float)
        train_y = torch.tensor(train_data['labels'], dtype=torch.long)
        val_x = torch.tensor(val_data['samples'], dtype=torch.float)
        val_y = val_data['labels'].to(torch.long)
        test_x = torch.tensor(test_data['samples'], dtype=torch.float)
        test_y = test_data['labels'].to(torch.long)

        # Ensure the dimensions are correct, and permute if necessary
        self.train_set = TensorDataset(train_x.squeeze().permute(0, 2, 1), train_y)
        self.val_set = TensorDataset(val_x.squeeze().permute(0, 2, 1), val_y)
        self.test_set = TensorDataset(test_x.squeeze().permute(0, 2, 1), test_y)


if __name__ == '__main__':
    a = HARDataModule(download_dir='/home/itamar/projects/SS-GP-VIB/data/har',
                      batch_size=64,
                      num_workers=4,
                      n_classes=6,
                      seed=42,
                      num_labeled=1,
                      is_ssl=False,
                      dataset_name='HAR',
                      num_val=1,
                      validation_split_seed=42)

    a.prepare_data()
    a.setup()
