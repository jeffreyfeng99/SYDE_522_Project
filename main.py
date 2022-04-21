import argparse
import os, sys
import random
import json
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, auc
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


logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',
                    filemode='w')


def train_deep(model, loss, train_dataloader, val_dataloader, full_test_dataloader, train_fold_json, val_fold_json):
    model.cuda()
    criterion = loss.cuda()

    optimizer = optim.SGD(model.parameters(), lr,
                        momentum=momentum,
                        weight_decay=weight_decay)

    # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(num_epochs):
        train_fold_json.append({str(epoch): []})
        val_fold_json.append({str(epoch): []})

        # train for one epoch
        model.train()
        # train_epoch(train_dataloader, model, criterion, epoch, optimizer, train_json[str(fold)][epoch][str(epoch)])
        loss, _, _, _, _, _, epoch_list = epoch_pass(train_dataloader, model, criterion, epoch, optimizer,
                                                 epoch_list=train_fold_json[epoch][str(epoch)])
        train_fold_json[epoch][str(epoch)] = epoch_list

        # evaluate on validation set and test on test set
        val_fold_json[epoch][str(epoch)] = validate(val_dataloader, full_test_dataloader, model, criterion,
                                                          epoch, epoch_list=val_fold_json[epoch][str(epoch)])

        # scheduler.step()

    return train_fold_json, val_fold_json
#
#
# def train_epoch(train_loader, model, criterion, optimizer, epoch, epoch_list):
#     # switch to train mode
#     model.train()
#
#     loss, _, _, _, _, epoch_list = inference(train_loader, model, criterion, epoch, optimizer, epoch_list=epoch_list)


def validate(val_loader, test_loader, model, criterion, epoch, epoch_list):

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for loader in [val_loader, test_loader]:
            loss, correct, total, pred_all, y_all, probs, _ = epoch_pass(loader, model, criterion, epoch, train=False,
                                                                  epoch_list=epoch_list)

            lr_precision, lr_recall, _ = precision_recall_curve(y_all, probs)

            epoch_list.append({
                'val': {
                    'accuracy': float(100 * correct / total),
                    'auc_pr': float(auc(lr_recall, lr_precision)),
                    'precisions': [lr_precision.tolist()],
                    'recalls': lr_recall.tolist(),
                    'loss': float(loss.cpu().data.numpy()),
                    'confusion_matrix': [row.tolist() for row in confusion_matrix(y_all, pred_all)]
                },
                'test': {
                    'accuracy': float(100 * correct / total),
                    'auc_pr': float(auc(lr_recall, lr_precision)),
                    'precisions': [lr_precision.tolist()],
                    'recalls': lr_recall.tolist(),
                    'loss': float(loss.cpu().data.numpy()),
                    'confusion_matrix': [row.tolist() for row in confusion_matrix(y_all, pred_all)]
                }
            })

    print(f'epoch: {epoch}, validation, accuracy: {100*correct/total}%, loss: {loss.cpu().data.numpy()}')
    return epoch_list


def epoch_pass(loader, model, criterion, epoch, optimizer=None, train=True, epoch_list=None):
    correct, total = 0., 0.
    acc_all, loss_all, pred_all, y_all, prob_all = [], [], [], [], []
    for i, (X, y) in enumerate(tqdm(loader)):
        X = X.cuda()
        y = y.cuda()

        # compute output
        output = model(X)
        loss = criterion(output, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()


        acc_all.append(float(100 * correct / total))
        loss_all.append(float(loss.cpu().data.numpy()))
        pred_all.extend(predicted.cpu().numpy())
        y_all.extend(y.cpu().numpy())
        prob_all.extend(output.cpu().numpy())

        print(f'epoch: {epoch}, [iter: {i} / all {len(loader)}], accuracy: {100 * correct / total}%, loss: {loss.cpu().data.numpy()}')

    lr_precision, lr_recall, _ = precision_recall_curve(y_all, output)
    if train:
        epoch_list.append({
            'accuracy': np.mean(acc_all),
            'auc_pr': float(auc(lr_recall, lr_precision)),
            'precisions': [lr_precision.tolist()],
            'recalls': lr_recall.tolist(),
            'loss': np.mean(loss_all),
            'confusion_matrix': [row.tolist() for row in confusion_matrix(y_all, pred_all)]
        })
    else:
        print(f'epoch: {epoch}, validation, accuracy: {100 * correct / total}%, loss: {loss.cpu().data.numpy()}')

    return loss, correct, total, pred_all, y_all, output, epoch_list


def train_ml(model, input_data, k, hp_tune=None, log_file=None, filename=None, json=None):
    
    X_train, y_train, X_test, y_test = tuple(input_data)

    if hp_tune == "svm":
        svm_gridsearch_out = sys.stdout
        sys.stdout = log_file
    
        print('starting grid search')
        grid = GridSearchCV(model, svm_param_grid, cv=k, refit=True, verbose=3, n_jobs=1)
        grid.fit(X_train, y_train)
        pred_values = grid.predict(X_test)
        filename += f"_kernel-{grid.best_params_['kernel']}_gamma-{grid.best_params_['gamma']}_C-{grid.best_params_['C']}"
        sys.stdout = svm_gridsearch_out
        log_file.close()
        
        print(f'Best SVM params found: {grid.best_params_}')

        lr_precision, lr_recall, _ = precision_recall_curve(y_test, grid.predict_proba(X_test)[:, 1])

        json['0'] = {}
        json['0']['auc_pr'] = float(auc(lr_recall, lr_precision))
        json['0']['precisions'] = [lr_precision.tolist()]
        json['0']['recalls'] = lr_recall.tolist()
        json['0']['accuracy'] = float(accuracy_score(pred_values, y_test))
        json['0']['confusion_matrix'] = [row.tolist() for row in confusion_matrix(y_test, pred_values)]
        
    else:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rand_state)

        best_model = model
        best_acc = 0.
        acc_score = []

        for k, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):#kf.split(X):
            X_tr, X_val = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[test_index]

            model.fit(X_tr, y_tr)
            pred_values = model.predict(X_val)
            acc = accuracy_score(pred_values, y_val)
            acc_score.append(acc)

            if acc > best_acc:
                best_model = model

            lr_precision, lr_recall, _ = precision_recall_curve(y_val, model.predict(X_val))
            json[str(k)] = {}
            json[str(k)]['auc_pr'] = float(auc(lr_recall, lr_precision))
            json[str(k)]['precisions'] = [lr_precision.tolist()]
            json[str(k)]['recalls'] = lr_recall.tolist()
            json[str(k)]['accuracy'] = float(acc)
            json[str(k)]['confusion_matrix'] = [row.tolist() for row in confusion_matrix(y_val, pred_values)]
            
        avg_acc_score = sum(acc_score) / k

        test_vals = best_model.predict(X_test)
        test_acc = accuracy_score(test_vals, y_test)
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, best_model.predict(X_test))
        json['test'] = {}
        json['test']['auc_pr'] = float(auc(lr_recall, lr_precision))
        json['test']['precisions'] = [lr_precision.tolist()]
        json['test']['recalls'] = lr_recall.tolist()
        json['test']['accuracy'] = float(test_acc)
        json['test']['confusion_matrix'] = [row.tolist() for row in confusion_matrix(y_test, test_vals)]

        print('accuracy of each fold - {}'.format(acc_score))
        print('Avg accuracy : {}'.format(avg_acc_score))
        print(f'Test accuracy : {test_acc}')

    return filename, json


def main(args):
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        gpu = args.gpu

    target_problem = args.problem

    print(f'train dataset chosen: {args.train_dataset}')
    train_dataset_name = args.train_dataset
    train_dataset = os.path.join(dataset_root, train_dataset_name+'.csv')
    d = int(args.cross and args.train_dataset == datasets[0])
    if args.train_dataset == datasets[-1]:
        d = -1
    print(f'validation dataset chosen: {datasets[d]}')
    val_dataset_name = datasets[d]
    val_dataset = os.path.join(dataset_root, val_dataset_name+'.csv')

    print('setting up json log files')
    train_json = {}
    val_json = {}
    os.makedirs(os.path.join(output_json_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_json_root, 'val'), exist_ok=True)

    time = datetime.now().strftime("%H-%M-%S")
    train_json_base_filename = f"train_{time}_dataset-{train_dataset_name}_cross-{int(args.cross)}_balanced-{args.data_balance}"
    val_json_base_filename = f"val_{time}_dataset-{val_dataset_name}_cross-{int(args.cross)}_balanced-{args.data_balance}"

    rand_state = random.randint(1, 10000)

    print('setting up data')
    df = pd.read_csv(train_dataset)
    X, y = select_X_y(df, target_problem, all_problems.keys())
    if args.cross:
        # train_df = df  # TODO instead of reading from csv, can just get directly from ceqd
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
        model, deep, train_json_base_filename, val_json_base_filename = \
            Predictor(args.model, len(X.columns), all_problems[target_problem],
                      train_json_base_filename, val_json_base_filename)
    else:
        print('need a model to run')
        exit()
    
    print('starting training')
    if not deep:
        log_file = None
        if args.model == "svm":
            log_filepath = os.path.join(output_json_root, 'val', val_json_base_filename+'_ALL_RUNS.txt').replace('\\', '/')
            log_file = open(log_filepath, "w")
            
        val_json_base_filename, val_json = train_ml(model, input_data, k, args.model, log_file, val_json_base_filename, val_json)
        train_json_base_filename += f".json"
        val_json_base_filename += f".json"
    else:
        train_json_base_filename += f"_loss-{args.loss}.json"
        val_json_base_filename += f"_loss-{args.loss}.json"

        full_test_dataloader = create_dataloader(X_test, y_test, False, batch_size, num_workers)

        torch.save(model.state_dict(), blank_model_path)

        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rand_state)
        for fold, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
            # reset model on each fold
            model.load_state_dict(torch.load(blank_model_path))

            # Dividing data into folds
            X_train_fold = X_train[train_index]
            X_val_fold = X_train[test_index]
            y_train_fold = y_train[train_index]
            y_val_fold = y_train[test_index]

            train_dataloader = create_dataloader(X_train_fold, y_train_fold, True, batch_size, num_workers)
            val_dataloader = create_dataloader(X_val_fold, y_val_fold, False, batch_size, num_workers)

            loss = Loss(args.loss)
            print(f'\n\n--------------Running fold {fold}--------------')
            train_json[str(fold)] = []
            val_json[str(fold)] = []
            train_json[str(fold)], val_json[str(fold)] = train_deep(model, loss, train_dataloader, val_dataloader,
                                                                    full_test_dataloader,
                                                                    train_json[str(fold)], val_json[str(fold)])

    train_json_filepath = os.path.join(output_json_root, 'train', train_json_base_filename).replace('\\', '/')
    val_json_filepath = os.path.join(output_json_root, 'val', val_json_base_filename).replace('\\', '/')
    with open(train_json_filepath, 'w') as train_write, open(val_json_filepath, 'w') as val_write:
        json.dump(train_json, train_write, indent=4)
        json.dump(val_json, val_write, indent=4)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Experimenting flow')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--problem', type=str.lower, default='readmissiontime_multiclass',
                        choices=['readmissiontime_multiclass', 'deathtime_multiclass',
                                 'readmissiontime', 'deathtime', 'death'],
                        help='chooses target problem to learn')
    parser.add_argument('--train_dataset', type=str.lower, default='normalized_uci_df',
                        choices=['normalized_uci_df', 'normalized_zigong_df', 'normalized_uci_and_zigong_df'],
                        help='chooses which dataset to train on')
    parser.add_argument('--cross', action='store_true', default=False,
                        help='cross=True will choose the other dataset for test')
    parser.add_argument('--data_balance', action='store_true', default=False,
                        help='enables SMOTE oversampling')
    parser.add_argument('--model', type=str.lower, default='none', choices=['none', 'svm', 'kmeans', 'dnn'],
                        metavar='ML',
                        help='chooses model for prediction (none [default], svm, kmeans, dnn')
    parser.add_argument('--loss', type=str.lower, default='none', choices=['none', 'ce', 'focal'], metavar='L',
                        help='chooses loss function for training (none [default], ce, focal)')
    args = parser.parse_args()


    main(args)






