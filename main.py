import argparse
import os
import importlib
import random
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn

from models.build_model import *
import create_equivalent_dataset as ceqd
from dataloader import *
from config import *


def train_deep(model, loss, train_dataloader, test_dataloader, gpu, train_json, val_json):
    model.cuda()
    criterion = loss.cuda()

    optimizer = optim.SGD(model.parameters(), lr,
                        momentum=momentum,
                        weight_decay=weight_decay)

    # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(num_epochs):

        # train for one epoch
        train_epoch(train_dataloader, model, criterion, optimizer, epoch, gpu, train_json)

        # evaluate on validation set
        acc1 = validate(test_dataloader, model, criterion, epoch, val_json)

        # scheduler.step()


def train_epoch(train_loader, model, criterion, optimizer, epoch, gpu, json):
    # switch to train mode
    model.train()

    correct, total = 0., 0.
    for i, (X, y) in enumerate(tqdm(train_loader)):

        X = X.cuda()
        y = y.cuda()

        # compute output
        output = model(X)
        loss = criterion(output, y)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        json[str(epoch)] = {
            'iter': i,
            'accuracy': float(100*correct/total),
            'loss': float(loss.cpu().data.numpy())
        }        

        print(f'epoch: {epoch}, [iter: {i} / all {len(train_loader)}], accuracy: {100*correct/total}%, loss: {loss.cpu().data.numpy()}')


def validate(val_loader, model, criterion, epoch, json):

    # switch to evaluate mode
    model.eval()
    correct, total = 0., 0.
    pred_all, y_all = [], []
    with torch.no_grad():
        for i, (X, y) in enumerate(val_loader):
            X = X.cuda()
            y = y.cuda()

            # compute output
            output = model(X)
            loss = criterion(output, y)

            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            pred_all.extend(predicted.cpu().numpy())
            y_all.extend(y.cpu().numpy())

    json[str(epoch)] = {
                'iter': i,
                'accuracy': float(100*correct/total),
                'loss': float(loss.cpu().data.numpy()),
                'confusion_matrix': [row.tolist() for row in confusion_matrix(y_all, pred_all)]
            } 

    print(f'epoch: {epoch}, validation, accuracy: {100*correct/total}%, loss: {loss.cpu().data.numpy()}')


def train_ml(model, input_data, k, hp_tune=None, filename=None, json=None):
    
    X_train, y_train, X_test, y_test = tuple(input_data)

    if hp_tune == "svm":
        grid = GridSearchCV(model, svm_param_grid, cv=k, refit=True, verbose=2)
        grid.fit(X_train, y_train)
        pred_values = grid.predict(X_test)
        filename += f"_kernel-{grid.best_params_['kernel']}_gamma-{grid.best_params_['gamma']}_C-{grid.best_params_['C']}"
        
        #TODO save confusion matrix plot
        json['0'] = {}
        json['0']['accuracy'] = float(accuracy_score(pred_values, y_test))
        json['0']['confusion_matrix'] = [row.tolist() for row in confusion_matrix(y_test, pred_values)]
        
    else:
        # kf = KFold(n_splits=k, shuffle=True, random_state=rand_state)
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rand_state)

        best_model = model
        best_acc = 0.
        acc_score = []

        for k, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):#kf.split(X):
            X_tr, X_val = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
            y_tr, y_val = y_train[train_index], y_train[test_index]

            model.fit(X_tr, y_tr)
            pred_values = model.predict(X_val)
            acc = accuracy_score(pred_values, y_val)
            acc_score.append(acc)

            if acc > best_acc:
                best_model = model

            json[str(k)] = {}
            json[str(k)]['accuracy'] = float(acc)
            json[str(k)]['confusion_matrix'] = [row.tolist() for row in confusion_matrix(y_val, pred_values)]
            
        avg_acc_score = sum(acc_score) / k

        test_vals = best_model.predict(X_test)
        test_acc = accuracy_score(test_vals, y_test)
        json['test'] = {}
        json['test']['test_accuracy'] = float(test_acc)
        json['test']['confusion_matrix'] = [row.tolist() for row in confusion_matrix(y_test, test_vals)]

        print('accuracy of each fold - {}'.format(acc_score))
        print('Avg accuracy : {}'.format(avg_acc_score))
        print(f'Test accuracy : {test_acc}')


def main(args):
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        gpu = args.gpu

    target_problem = args.problem

    print('setting up json log files')
    train_json = {}
    val_json = {}

    date = datetime.now().strftime("%m%d%Y")
    train_dataset_name = train_dataset.split('/')[-1].split('.csv')[0]
    val_dataset_name = val_dataset.split('/')[-1].split('.csv')[0]
    train_json_base_filename = f"train_{date}_dataset-{train_dataset_name}_balanced-{args.data_balance}"
    val_json_base_filename = f"val_{date}_dataset-{val_dataset_name}_balanced-{args.data_balance}"

    rand_state = random.randint(1, 10000)

    print('setting up data')
    df = pd.read_csv(train_dataset)
    X, y = select_X_y(df, target_problem, all_problems.keys())
    if cross_dataset:
        train_df = df  # TODO instead of reading from csv, can just get directly from ceqd
        val_df = pd.read_csv(val_dataset)
        X_train, y_train = X, y
        X_test, y_test = select_X_y(val_df, target_problem, all_problems.keys())
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    if args.data_balance:
        X_train, y_train = data_balance(X_train, y_train, **smote_kwargs)
        X_test, y_test = data_balance(X_test, y_test, **smote_kwargs)

    input_data = [X_train, y_train, X_test, y_test]
    
    print('setting up model')
    
    model = None
    deep = None
    if args.model != 'none':
        model, deep, train_json_base_filename, val_json_base_filename = Predictor(args.model, len(X.columns), all_problems[target_problem], 
                                            train_json_base_filename, val_json_base_filename)
    else:
        print('need a model to run')
        exit()
    
    print('starting training')
    if not deep:
        train_json_base_filename += f".json"
        val_json_base_filename += f".json"
        train_ml(model, input_data, k, args.model, val_json_base_filename, val_json)
    else:
        train_dataloader = create_dataloader(X_train, y_train, True, batch_size, num_workers)
        test_dataloader = create_dataloader(X_test, y_test, False, batch_size, num_workers)
        loss = Loss(args.loss)
        train_json_base_filename += f"_loss-{args.loss}.json"
        val_json_base_filename += f"_loss-{args.loss}.json"

        train_deep(model, loss, train_dataloader, test_dataloader, gpu, train_json, val_json)

    os.makedirs(os.path.join(output_json_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_json_root, 'val'), exist_ok=True)
    train_json_filepath = os.path.join(output_json_root, 'train', train_json_base_filename).replace('\\', '/')
    val_json_filepath = os.path.join(output_json_root, 'val', val_json_base_filename).replace('\\', '/')
    with open(train_json_filepath, 'w') as train_write, open(val_json_filepath, 'w') as val_write:
        json.dump(train_json, train_write, indent=4)
        json.dump(val_json, val_write, indent=4)

    # TODO set up analyze.py to find best train acc and val acc for dnn, 
    # TODO save the pretty seaborn confusion matrices as pngs
    # TODO set up sh script to run all the models


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Experimenting flow')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--problem', type=str.lower, default='readmissiontime_multiclass',
                        choices=['readmissiontime_multiclass', 'deathtime_multiclass',
                                 'readmissiontime', 'deathtime', 'death'],
                        help='chooses target problem to learn')
    parser.add_argument('--data_balance', action='store_true', default=False,
                        help='enables SMOTE oversampling')
    parser.add_argument('--model', type=str.lower, default='none', choices=['none', 'svm', 'kmeans', 'dnn'],
                        metavar='ML',
                        help='chooses model for prediction (none [default], svm, kmeans, dnn')
    parser.add_argument('--loss', type=str.lower, default='none', choices=['none', 'ce', 'focal'], metavar='L',
                        help='chooses loss function for training (none [default], ce, focal)')
    args = parser.parse_args()

    main(args)






