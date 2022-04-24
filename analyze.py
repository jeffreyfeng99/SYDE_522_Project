from cProfile import label
from distutils.log import info
import os
import argparse
import json
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

#TODO: get precision, recall, f1 from conf matrix
#TODO: output precision recall curves
#TODO: combine precision, recall, f1, auc_pr into the bar graphs

"""
svm dict format
train = {}
val = {
    "0": {
        "auc_pr": float,
        "precisions": (list of) list of floats,
        "recalls": (list of) list of floats,
        "auc_pr": float,
        "confusion_matrix": list of lists of int (each row corresponds to truth)
    }
}

kmeans dict format
train = {}
val = {
    "fold_i": {
        "auc_pr": float,
        "precisions": (list of) list of floats,
        "recalls": (list of) list of floats,
        "auc_pr": float,
        "confusion_matrix": list of lists of int (each row corresponds to truth)
    },
    "test": { this is fold_i model w best auc_pr predicting on test set
        "auc_pr": float,
        "precisions": (list of) list of floats,
        "recalls": (list of) list of floats,
        "auc_pr": float,
        "confusion_matrix": list of lists of int (each row corresponds to truth)
    }
}

dnn dict format
train = {
    "fold_i": [
        "epoch_j": { mean across epoch
            'auc_pr': float,
            'auc_pr': float,
            'precisions': list of floats,
            'recalls': list of floats,
            'loss': float,
            'confusion_matrix': list of list of lists of int (each row corresponds to truth)
        }
    ]
}

val = {
    "fold_i": [
        "epoch_j: {
            "val": {
                'auc_pr': float,
                'auc_pr': float,
                'precisions': list of floats,
                'recalls': list of floats,
                'loss': float,
                'confusion_matrix': list of list of lists of int (each row corresponds to truth)
            },
            "test": {
                'auc_pr': float,
                'auc_pr': float,
                'precisions': list of floats,
                'recalls': list of floats,
                'loss': float,
                'confusion_matrix': list of list of lists of int (each row corresponds to truth)
            }
        }
    ]
}
"""

def output_bar_graphs(pth, data):
    # x-axis for models
    # group bars for prec, recall, f1, and auc_pr per model
    # y-axis just 0 to 1

    plt.bar(x_axis +0.20, Python, width=0.2, label = 'Python')
    plt.bar(x_axis +0.20*2, Java, width=0.2, label = 'Java')
    plt.bar(x_axis +0.20*3, Php, width=0.2, label = 'Php')

    fig, ax = plt.subplots()
    fig.savefig(pth)

def div0( a, b, fill=0 ):
    """ a / b, divide by 0 -> `fill`
        div0( [-1, 0, 1], 0, fill=np.nan) -> [nan nan nan]
        div0( 1, 0, fill=np.inf ) -> inf
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
    if np.isscalar( c ):
        return c if np.isfinite( c ) \
            else fill
    else:
        c[ ~ np.isfinite( c )] = fill
        return c

def process_matrix(matrices):
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F-Measure = (2 * Precision * Recall) / (Precision + Recall)
    output = {'precision': [],
            'recall': [],
            'specificity': [],
            'f1': []}

    for data in matrices:
        if data is None:
            output['precision'].append(np.nan)
            output['recall'].append(np.nan)
            output['specificity'].append(np.nan)
            output['f1'].append(np.nan)
        else:
            precision = div0(data[1][1], data[1][1] + data[1][0])
            recall = div0(data[1][1], data[1][1] + data[0][1])

            output['precision'].append(precision)
            output['recall'].append(recall)
            output['specificity'].append(div0(data[0][0], data[0][0] + data[1][0]))
            output['f1'].append(div0(2*precision*recall,precision + recall))

    return output

class PRCurves(object):
    def __init__(self):
        self.legend = []
        self.precs = []
        self.recalls = []

    def output_pr_curves(self, pth, all_prec, all_recall, data, label_f1_score=True):
        
        self.legend.append(pth)
        self.precs.append(all_prec)
        self.recalls.append(all_recall)
        
        # if label_f1_score:
        #     print(precision, recall, f1_score)
        #     ax.axis([0., 1., 0., 1.])
        #     ax.text(0.25, 0.25, f'{f1_score}', bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 5})
    
    def save_pr_curves(self, pth):
        fig, ax = plt.subplots()

        for i in range(len(self.precs)):
            ax.plot(self.recalls[i], self.precs[i], marker='.')

        ax.set_title('Precision-Recall Curves of Best Models')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        fig.savefig(pth)
        plt.close(fig)


def output_confmat(pth, data):
    df_cm = pd.DataFrame(data, index = ['negative', 'positive'],
                columns = ['pred negative', 'pred positive'])
    cm = sn.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig(pth) 
    plt.close(fig)

class DataTracker(object):

    def __init__(self, keys=['dataset','balanced','model','cross'], metrics=['acc','auc_pr','precision','recall','f1','specificity']):

        self.names = {key:[] for key in keys}
        self.groups = []
        self.metrics = {metric:[] for metric in metrics}
    
    def _parse_name(self, name):
        # this is hardcoded
        split_name = name.split('_')
        temp_keys = {}
        
        for i in range(len(split_name)):
            key = split_name[i].split('-')[0]
            if key in self.names.keys():
                temp_keys[key] = split_name[i].split('-')[1] + split_name[i+1].split('-')[0] ############# temprory fix for naming with underscores
        
        return temp_keys

    def add_item(self, name, metrics, grouping=['max_fold','avg_fold','test','train']):

        parsed_name = self._parse_name(name) 

        created_model_identifiers = False

        for metric in metrics.keys():
            for i in range(len(metrics[metric])):
                
                if not created_model_identifiers:
                    for key in parsed_name.keys():
                        self.names[key].append(parsed_name[key])
                    
                    self.groups.append(grouping[i])
                self.metrics[metric].append(metrics[metric][i])
            
            created_model_identifiers = True
    
    def output(self, output_dir, metric='acc'):
        sn.set_theme(style="whitegrid")

        df = pd.DataFrame(self.names)
        df['groups'] = self.groups
        df['metrics'] = self.metrics[metric]

        df = df.sort_values(by=['model','groups'])

        dsets = df['dataset'].unique()

        for dset in dsets:
            dset_specific = df.loc[df['dataset'] == dset]

            # bar = sn.catplot(x='model', y='metrics', hue='groups', data=dset_specific, kind='bar',
            #                 ci="sd", palette="dark", alpha=.6, height=6)
            # pth = f'dataset-{dset}-{metric}.jpg'

            # fig = bar.figure
            # fig.savefig(os.path.join(output_dir,pth)) 
            # plt.close(fig)
            
            balances = dset_specific['balanced'].unique()

            for balance in balances:
                balance_specific = dset_specific.loc[dset_specific['balanced'] == balance]

                bar = sn.catplot(x='model', y='metrics', hue='groups', data=balance_specific, kind='bar',
                            ci="sd", palette="dark", alpha=.6, height=6)
                plt.ylim(0,1)
                plt.xlabel('model')
                plt.ylabel(metric)

                pth = f'dataset-{dset}_balanced-{balance}-{metric}.jpg'

                fig = bar.figure

                fig.savefig(os.path.join(output_dir,pth)) 
                plt.close(fig)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimenting flow')
    parser.add_argument('--json_dir', type=str, default='./output/04232022')
    parser.add_argument('--output_dir', type=str, default='./output_figures_test')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    info_tracker = DataTracker()
    pr_tracker = PRCurves()
    
    for root, dirs, files in os.walk(args.json_dir):

        for file in files:
            if file.endswith('.txt') and 'svm' in file:
                file_path = os.path.join(root,file)
                file = os.path.splitext(file)[0]
                best_params = dict((el, None) for el in ['kernel', 'gamma', 'C'])
                for key in file.split('_'):
                    if key.split('-')[0] in best_params.keys():
                        best_params[key.split('-')[0]] = key.split('-')[-1]

                with open(file_path, 'r') as gridsearch_txt:
                    best_params_fold_scores = []
                    best_params_strings = [f"{param}={best_params[param]}" for param in best_params.keys()]
                    for line in gridsearch_txt.readlines():
                        if all(param in line for param in best_params_strings):
                            score_str = next(s for s in line.split(' ') if 'score=' in s)
                            best_params_fold_scores.append(float(score_str.split('=')[-1]))
                        if len(best_params_fold_scores) == 5:
                            break
                
                avg_fold_auc_pr = np.mean(best_params_fold_scores)
                max_fold_idx = np.argmax(best_params_fold_scores)
                max_fold_auc_pr = best_params_fold_scores[max_fold_idx]
                
                conf_matrix_analyses = process_matrix([None, None, None])
                conf_matrix_analyses['auc_pr'] = [max_fold_auc_pr, avg_fold_auc_pr, np.nan]
                info_tracker.add_item(os.path.splitext(file)[0], conf_matrix_analyses, grouping=['max_fold', 'avg_fold', 'train'])


            if file.endswith('.json'):
                file_path = os.path.join(root,file)
                json_file = json.load(open(file_path,'r'))

                if len(json_file.keys()) <= 0:
                    continue

                if 'val' in file:
                    fold_names = []
                    fold_acc = []
                    best_epoch_names = []
                    trn_pths = []

                    if 'dnn' in file:
                        for fold in json_file.keys():

                            #find best epoch
                            epoch_names = []
                            epoch_acc = []
                            for i, epoch_dict in enumerate(json_file[fold]):
                                # epoch_names.append(epoch)
                                epoch_acc.append(epoch_dict[str(i)]['val']['auc_pr'])
                            
                            best_epoch = np.argmax(epoch_acc)
                            best_epoch_names.append(best_epoch)

                            fold_names.append(fold)
                            fold_acc.append(json_file[fold][best_epoch][str(best_epoch)]['val']['auc_pr'])
                            
                            trn_pth = os.path.join(os.path.split(root)[0], 'train', file.replace('val','train'))
                            trn_pths.append(trn_pth)
                            
                            
                        avg_fold_acc = np.mean(fold_acc)
                        max_fold_idx = np.argmax(fold_acc)
                        max_fold_acc = fold_acc[max_fold_idx]
                        max_fold_mtx = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['val']['confusion_matrix']
                        max_fold_precs = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['val']['precisions']
                        max_fold_recalls = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['val']['recalls']

                        trn_file = json.load(open(trn_pths[max_fold_idx],'r'))                      
                        max_fold_trn_acc = trn_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['auc_pr']
                        max_fold_trn_mtx = trn_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['confusion_matrix']
                        max_fold_trn_precs = trn_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['precisions']
                        max_fold_trn_recalls = trn_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['recalls']

                        tst_acc = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['test']['auc_pr']
                        tst_mtx = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['test']['confusion_matrix']
                        tst_precs = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['test']['precisions']
                        tst_recalls = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['test']['recalls']

                        # info_tracker.add_item(os.path.splitext(file)[0], {'acc': (max_fold_acc,avg_fold_acc,tst_acc,max_fold_trn_acc)})
                        conf_matrix_analyses = process_matrix([max_fold_mtx, None, tst_mtx, max_fold_trn_mtx])
                        conf_matrix_analyses['auc_pr'] = (max_fold_acc,avg_fold_acc,tst_acc,max_fold_trn_acc)
                        info_tracker.add_item(os.path.splitext(file)[0], conf_matrix_analyses)

                        # output best fold acc, avg fold accs, best test acc, trn_acc
                        
                    else:
                        if 'kmeans' in file:
                            for fold in json_file.keys():
                                if 'test' not in fold:
                                    fold_names.append(fold)
                                    fold_acc.append(json_file[fold]['auc_pr'])
                            
                            avg_fold_acc = np.mean(fold_acc)
                            max_fold_idx = np.argmax(fold_acc)
                            max_fold_acc = fold_acc[max_fold_idx]
                            max_fold_mtx = json_file[fold_names[max_fold_idx]]['confusion_matrix']
                            max_fold_precs = json_file[fold_names[max_fold_idx]]['precisions']
                            max_fold_recalls = json_file[fold_names[max_fold_idx]]['recalls']

                            tst_acc = json_file['test']['auc_pr']
                            tst_mtx = json_file['test']['confusion_matrix']
                            tst_precs = json_file['test']['precisions']
                            tst_recalls = json_file['test']['recalls']

                            conf_matrix_analyses = process_matrix([max_fold_mtx, None, tst_mtx, None])
                            conf_matrix_analyses['auc_pr'] = (max_fold_acc,avg_fold_acc,tst_acc,np.nan)
                            info_tracker.add_item(os.path.splitext(file)[0], conf_matrix_analyses)
                        else:
                            max_fold_mtx = json_file['0']['confusion_matrix']
                            max_fold_precs = json_file['0']['precisions']
                            max_fold_recalls = json_file['0']['recalls']
                            
                            tst_acc = json_file['0']['auc_pr']
                            tst_mtx = json_file['0']['confusion_matrix']
                            tst_precs = json_file['0']['precisions']
                            tst_recalls = json_file['0']['recalls']

                            conf_matrix_analyses = process_matrix([tst_mtx])
                            conf_matrix_analyses['auc_pr'] = [tst_acc]
                            info_tracker.add_item(os.path.splitext(file)[0], conf_matrix_analyses, grouping=['test'])

                    output_confmat(os.path.join(args.output_dir, os.path.splitext(file)[0] + '_maxfold_confmat.jpg'), max_fold_mtx)
                    output_confmat(os.path.join(args.output_dir, os.path.splitext(file)[0] + '_test_confmat.jpg'), tst_mtx)
                    maxfold_scores = pr_tracker.output_pr_curves(os.path.join(args.output_dir, os.path.splitext(file)[0] + '_maxfold_prcurve.jpg'), max_fold_precs, max_fold_recalls, max_fold_mtx)
                    test_scores = pr_tracker.output_pr_curves(os.path.join(args.output_dir, os.path.splitext(file)[0] + '_test_prcurve.jpg'), tst_precs, tst_recalls,tst_mtx)

    info_tracker.output(args.output_dir,metric='auc_pr')
    info_tracker.output(args.output_dir,metric='f1')
    info_tracker.output(args.output_dir,metric='specificity')
    pr_tracker.save_pr_curves(os.path.join(args.output_dir,'test.jpg'))
    


                
                
                
