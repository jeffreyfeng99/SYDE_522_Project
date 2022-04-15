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


# TODO: Also create dataloader here for DNN?
# TODO: create the one-to-all datasets for msfe training?