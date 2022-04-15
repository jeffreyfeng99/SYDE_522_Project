import argparse
import os
import importlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from config import *
from models.build_model import *
from models import losses
import create_equivalent_dataset as ceqd
from dataloader import *


def train():
    pass


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Experimenting flow')
    parser.add_argument('--data_balance', action='store_true', default=False,
                        help='enables SMOTE oversampling')
    parser.add_argument('--pca', action='store_true', default=False,
                        help='enables PCA for feature selection')
    parser.add_argument('--model', type=str.lower, default='none', choices=['none', 'svm', 'kmeans', 'dnn'], metavar='ML',
                        help='chooses model for prediction (none [default], svm, kmeans, dnn')
    parser.add_argument('--loss', type=str.lower, default='none', choices=['none', 'ce', 'focal', 'msfe'], metavar='L',
                        help='chooses loss function for training (none [default], ce, focal, msfe')
    args = parser.parse_args()

    if args.model != 'none':
        model = Predictor(args.model)
    else:
        print('need a model to run')
        exit()

    # TODO create dataset
    print('dataset loading')
    # datasets, dataset_test = dataset_read()
    # TODO separate dataset into X, y or return it that way

    # if args.data_balance:
        # TODO set up any SMOTE kwargs
        # X_res, y_res = data_balance(X_train, y_train, **smote_kwargs)

    # if args.pca:
        # TODO set up any PCA kwargs
        # pca = feature_select(data, **pca_kwargs)

    if args.loss == 'ce':
        loss = F.cross_entropy(**ce_kwargs)
    elif args.loss == 'focal':
        loss = losses.FocalLoss()
    elif args.loss == 'msfe':
        loss = losses.MSFELoss()
    else:
        pass










