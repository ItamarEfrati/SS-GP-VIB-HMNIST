import math
import random
import numpy as np

from torch.utils.data import DataLoader, Dataset, TensorDataset
import pytorch_lightning as pl


def split_train_val(train_labels, num_val, n_classes, seed=None):
    if seed is not None:
        np.random.seed(seed)
    val_indices = []
    train_indices = []
    num_total = len(train_labels)
    num_per_class = [(train_labels == c).sum().astype(int) for c in range(n_classes)]
    for c, num_class in zip(range(n_classes), num_per_class):
        val_this_class = max(int(num_val * (num_class / num_total)), 1)
        class_indices = np.where(train_labels == c)[0]
        np.random.shuffle(class_indices)
        val_indices.append(class_indices[:val_this_class])
        train_indices.append(class_indices[val_this_class:])

    return val_indices, train_indices


def split_train(train_labels, num_labeled, n_classes, seed=None):
    """
    Split the train data into the following three set:
    (1) labeled data
    (2) unlabeled data
    (3) val data

    Data distribution of the three sets are same as which of the
    original training data.

    Inputs:
        labels: (np.int) array of labels
        num_labeled: (int)
        num_val: (int)
        _n_classes: (int)


    Return:
        the three indices for the three sets
    """
    if seed is not None:
        np.random.seed(seed)

    train_indices = []
    num_total = len(train_labels)
    num_per_class = [(train_labels == c).sum().astype(int) for c in range(n_classes)]

    # obtain val indices, data evenly drawn from each class
    for c, num_class in zip(range(n_classes), num_per_class):
        class_indices = np.where(train_labels == c)[0]
        train_indices.append(class_indices)

    labeled_indices, unlabeled_indices = split_label_unlabeled(n_classes, num_labeled, num_per_class, num_total,
                                                               train_indices)

    return labeled_indices, unlabeled_indices


def split_label_unlabeled(_n_classes, num_labeled, num_per_class, num_total, train_indices):
    labeled_indices = []
    unlabeled_indices = []
    for c, num_class in zip(range(_n_classes), num_per_class):
        num_labeled_this_class = int(num_labeled * num_class)
        labeled_indices.append(train_indices[c][:num_labeled_this_class])
        unlabeled_indices.append(train_indices[c][num_labeled_this_class:])
    labeled_indices = np.hstack(labeled_indices)
    unlabeled_indices = np.hstack(unlabeled_indices)
    return labeled_indices, unlabeled_indices


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        if len(self.dataset[self.indices[idx]]) == 3:
            data, mask, label = self.dataset[self.indices[idx]]
            if self.transform is not None:
                data = self.transform(data)
            return data, mask, label
        else:
            data, label = self.dataset[self.indices[idx]]
            if self.transform is not None:
                data = self.transform(data)
            return data, label

    def __len__(self):
        return len(self.indices)


class CustomSemiDataset(Dataset):
    def __init__(self, datasets, is_ssl):
        self.is_ssl = is_ssl
        self.datasets = datasets

        self.map_indices = [[] for _ in self.datasets]
        self.min_length = min(len(d) for d in self.datasets)
        self.max_length = max(len(d) for d in self.datasets) if self.is_ssl else self.min_length

    def __getitem__(self, i):
        if self.is_ssl:
            return tuple(d[m[i]] for d, m in zip(self.datasets, self.map_indices))
        else:
            return tuple(self.datasets[0][self.map_indices[0][i]])

    def construct_map_index(self):
        """
        Construct the mapping indices for every data. Because the __len__ is larger than the size of some datset,
        the map_index is use to map the parameter "index" in __getitem__ to a valid index of each dataset.
        Because of the dataset has different length, we should maintain different indices for them.
        """

        def update_indices(original_indices, data_length, max_data_length):
            # update the sampling indices for this dataset

            # return: a list, which maps the range(max_data_length) to the val index in the dataset

            original_indices = original_indices[max_data_length:]  # remove used indices
            fill_num = max_data_length - len(original_indices)
            batch = math.ceil(fill_num / data_length)

            additional_indices = list(range(data_length)) * batch
            random.shuffle(additional_indices)

            original_indices += additional_indices

            assert (
                    len(original_indices) >= max_data_length
            ), "the length of matcing indices is too small"

            return original_indices

        # use same mapping index for all unlabeled dataset for data consistency
        # the i-th dataset is the labeled data
        if self.is_ssl:
            self.map_indices = [
                update_indices(m, len(d), self.max_length)
                for m, d in zip(self.map_indices, self.datasets)
            ]
        else:
            self.map_indices = [update_indices(self.map_indices, len(self.datasets[0]), self.min_length)]

        # use same mapping index for all unlabeled dataset for data consistency
        # the i-th dataset is the labeled data
        for i in range(1, len(self.map_indices)):
            self.map_indices[i] = self.map_indices[1]

    def __len__(self):
        # will be called every epoch
        return self.max_length


class DataModuleBase(pl.LightningDataModule):
    labeled_indices: ...
    unlabeled_indices: ...
    unlabeled_mask_indices: ...
    val_indices: ...

    def __init__(self, data_root, num_workers, batch_size, num_labeled, num_val):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_labeled = num_labeled
        self.num_val = num_val
        self.n_classes = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.num_workers = num_workers

    def train_dataloader(self):
        # get and process the data first

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        # return val dataloader

        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return val_loader

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def num_labeled_data(self):
        assert self.train_set is not None, (
                "Load train data before calling %s" % self.num_labeled_data.__name__
        )
        return len(self.labeled_indices)

    @property
    def num_unlabeled_data(self):
        assert self.train_set is not None, (
                "Load train data before calling %s" % self.num_unlabeled_data.__name__
        )
        return len(self.unlabeled_indices)

    @property
    def num_val_data(self):
        assert self.train_set is not None, (
                "Load train data before calling %s" % self.num_val_data.__name__
        )
        return len(self.val_indices)

    @property
    def num_test_data(self):
        assert self.test_set is not None, (
                "Load test data before calling %s" % self.num_test_data.__name__
        )
        return len(self.test_set)


class SemiDataModule(DataModuleBase):
    """
    Data module for semi-supervised tasks. self.prepare_data() is not implemented. For custom dataset,
    inherit this class and implement self.prepare_data().
    """

    def __init__(
            self,
            data_root,
            num_workers,
            batch_size,
            num_labeled,
            num_val,
            validation_split_seed,
            is_ssl,
    ):
        super(SemiDataModule, self).__init__(data_root, num_workers, batch_size, num_labeled, num_val)
        self.is_ssl = is_ssl
        self.validation_split_seed = validation_split_seed

    def setup(self, stage=None):
        indices = np.arange(len(self.train_set))
        y_train = np.array([self.train_set[i][-1] for i in indices], dtype=np.int64)
        self.labeled_indices, self.unlabeled_indices = split_train(y_train,
                                                                   self.num_labeled,
                                                                   self.n_classes,
                                                                   seed=self.validation_split_seed)
        if self.is_ssl:
            train_list = [Subset(self.train_set, self.labeled_indices), Subset(self.train_set, self.unlabeled_indices)]
            self.train_set = CustomSemiDataset(train_list, self.is_ssl)
        else:
            tensors = self.train_set.tensors  # tuple of N tensors

            # Index every tensor consistently
            indexed_tensors = tuple(t[self.labeled_indices] for t in tensors)

            # Rebuild the dataset
            self.train_set = TensorDataset(*indexed_tensors)

    def train_dataloader(self):
        if self.is_ssl:
            self.train_set.construct_map_index()

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
