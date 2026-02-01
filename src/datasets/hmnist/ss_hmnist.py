import os
import requests
import numpy as np

from hydra.utils import to_absolute_path
from torch import tensor
from torch.utils.data import TensorDataset
from tqdm import tqdm

from datasets.ssl_dataset import SemiDataModule


class SemiSupervisedHMnist(SemiDataModule):
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
            **kwargs
    ):

        super(SemiSupervisedHMnist, self).__init__(
            download_dir,
            num_workers,
            batch_size,
            num_labeled,
            num_val,
            validation_split_seed,
            is_ssl,
        )
        self.n_classes = 10
        self.save_hyperparameters()

    def prepare_data(self):
        if not os.path.exists(to_absolute_path(self.hparams.download_dir)):
            os.makedirs(to_absolute_path(self.hparams.download_dir))
        file_path = os.path.join(to_absolute_path(self.hparams.download_dir), self.hparams.file_name)
        if not os.path.exists(file_path):
            headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
            response = requests.get(self.hparams.download_url, stream=True, headers=headers)
            content_length = int(response.headers['Content-Length'])
            pbar = tqdm(total=content_length)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=20_000_000):
                    if chunk:
                        f.write(chunk)
                    pbar.update(len(chunk))

        file_path = os.path.join(to_absolute_path(self.hparams.download_dir), self.hparams.file_name)
        data = np.load(file_path)

        # full data
        # missing and mask, 1 is an evidence for missing data, True is observed

        data_name_format = "x_{}_miss" if self.hparams.is_data_missing else 'x_{}_full'
        mask_format = "m_{}_miss"

        train_tensors = [tensor(data[data_name_format.format('train')])[:self.hparams.num_val]]
        val_tensors = [tensor(data[data_name_format.format('train')])[self.hparams.num_val:]]
        test_tensors = [tensor(data[data_name_format.format('test')])]

        train_tensors.append(tensor(data[mask_format.format('train')])[:self.hparams.num_val])
        val_tensors.append(tensor(data[mask_format.format('train')])[self.hparams.num_val:])
        test_tensors.append(tensor(data[mask_format.format('test')]))

        train_tensors.append(tensor(data['y_train'])[:self.hparams.num_val])
        val_tensors.append(tensor(data['y_train'])[self.hparams.num_val:])
        test_tensors.append(tensor(data['y_test']))

        self.train_set = TensorDataset(*train_tensors)
        self.val_set = TensorDataset(*val_tensors)
        self.test_set = TensorDataset(*test_tensors)


if __name__ == '__main__':
    d = SemiSupervisedHMnist(download_url='https://www.dropbox.com/s/xzhelx89bzpkkvq/hmnist_mnar.npz?dl=1',
                             download_dir='C:\Developments\GP-VIB\data\HMNIST',
                             file_name='hmnist_mnar.npz',
                             batch_size=64,
                             num_workers=2,
                             is_data_missing=True,
                             num_labeled=0.1,
                             num_val=5000,
                             validation_split_seed=42,
                             is_ssl=True)

    d.prepare_data()
    d.setup()
    t_d = d.train_dataloader()
    for batch in t_d:
        print(1)
        break
