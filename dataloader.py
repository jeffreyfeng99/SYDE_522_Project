import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


def data_balance(X, y, **kwargs):
    sm = SMOTE(**kwargs)
    X_res, y_res = sm.fit_resample(X, y)

    return X_res, y_res


def feature_select(data, **kwargs):
    pca = PCA(**kwargs)
    pca.fit(data)

    return pca


# TODO: create dataloader here for DNN

# TODO: create the one-to-all datasets for msfe training?
# dataframe - read the column for admissiontime_multiclass (5 class), deathtime_multiclass (5 class), or death (0/1)
# run entire dataset through and get confusion matrix
# some helper to create each one-to-all confusion matrix
# loss for each class is calculated separately, then summed