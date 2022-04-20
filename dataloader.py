import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data
from torch.utils.data import Subset

from collections import Counter


def data_balance(X, y, **kwargs):
    print(f'Original dataset shape: {Counter(y)}')
    sm = SMOTE(**kwargs)
    X_res, y_res = sm.fit_resample(X, y)

    print(f'New dataset shape: {Counter(y_res)}')

    return X_res, y_res


def feature_select(data, **kwargs):
    pca = PCA(**kwargs)
    pca.fit(data)

    return pca

def select_X_y(df, target, all):
    X = df.drop(all, 1)
    y = df[target]
    return X, y


class DfDataset(data.Dataset):

    def __init__(self, X, y):
        print(X)
        print(y)

        X = X.iloc[:,:].values
        y = y.iloc[:].values
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloader(X, y, train, batch_size, num_workers):
    dataset = DfDataset(X, y)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=True)

    return data_loader

# TODO: create the one-to-all datasets for msfe training?
# dataframe - read the column for admissiontime_multiclass (5 class), deathtime_multiclass (5 class), or death (0/1)
# run entire dataset through and get confusion matrix
# some helper to create each one-to-all confusion matrix
# loss for each class is calculated separately, then summed