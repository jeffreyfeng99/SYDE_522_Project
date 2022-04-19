import argparse
import os
import importlib
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn

from models.build_model import *
from models import losses
import create_equivalent_dataset as ceqd
from dataloader import *
from config import *


def train_deep(model, loss, train_dataloader, test_dataloader):
    model.cuda()
    criterion = loss.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(num_epochs):

        # train for one epoch
        train_epoch(train_dataloader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = validate(test_dataloader, model, criterion, args)

        # scheduler.step()


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    correct = 0
    for i, (X, y) in enumerate(tqdm(train_loader)):

        X = X.cuda(args.gpu, non_blocking=True)
        y = y.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(X)
        loss = criterion(output, y)

        # measure accuracy and record loss
        correct += (output)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'epoch: {epoch}, [iter: {i} / all {len(train_loader)}], loss: {loss.cpu().data.numpy()}')


def validate(val_loader, model, criterion, epoch):

    # switch to evaluate mode
    model.eval()
    correct, total = 0., 0.
    with torch.no_grad():
        for i, (X, y) in enumerate(val_loader):
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            # compute output
            output = model(X)
            loss = criterion(output, y)

            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f'epoch: {epoch}, validation, accuracy: {100*correct/total}%, loss: {loss.cpu().data.numpy()}')



def train_ml(model, X, y):
    kf = KFold(n_splits=k, shuffle=True, random_state=rand_state)

    acc_score = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        if args.data_balance:
            X_train, y_train = data_balance(X_train, y_train, **smote_kwargs)

        if args.pca:
            pca = feature_select(X_train, **pca_kwargs)
            X_train = pca.components_

        model.fit(X_train, y_train)
        pred_values = model.predict(X_test)

        acc = accuracy_score(pred_values, y_test)
        acc_score.append(acc)

    avg_acc_score = sum(acc_score) / k

    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main(args):
    problem = args.problem
    model = None
    deep = None
    if args.model != 'none':
        model, deep = Predictor(args.model)
    else:
        print('need a model to run')
        exit()

    # if rand_state is None:
    rand_state = random.randint(1, 10000)

    print('setting up data')
    df = pd.read_csv(dataset)  # TODO instead of reading from csv, can just get directly from ceqd
    X, y = select_X_y(df, problem)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rand_state)
    print('starting training')
    if not deep:
        train_ml(model, X, y)
    else:
        train_dataloader = create_dataloader(X_train, y_train, problem)
        test_dataloader = create_dataloader(X_test, y_test, problem)
        loss = Loss(args.loss)

        train_deep(model, loss, train_dataloader, test_dataloader)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Experimenting flow')
    parser.add_argument('--problem', type=str.lower, default='readmissiontime_multiclass',
                        choices=['readmissiontime_multiclass', 'deathtime_multiclass',
                                 'readmissiontime', 'deathtime', 'death'],
                        help='enables SMOTE oversampling')
    parser.add_argument('--data_balance', action='store_true', default=False,
                        help='enables SMOTE oversampling')
    parser.add_argument('--pca', action='store_true', default=False,
                        help='enables PCA for feature selection')
    parser.add_argument('--model', type=str.lower, default='none', choices=['none', 'svm', 'kmeans', 'dnn'],
                        metavar='ML',
                        help='chooses model for prediction (none [default], svm, kmeans, dnn')
    parser.add_argument('--loss', type=str.lower, default='none', choices=['none', 'ce', 'focal', 'msfe'], metavar='L',
                        help='chooses loss function for training (none [default], ce, focal, msfe')
    args = parser.parse_args()

    main(args)






