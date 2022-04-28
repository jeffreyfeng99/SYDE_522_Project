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
    def __init__(self, keys=['dataset','balanced','model','cross']):

        self.keys = keys
        self.headers = {key:[] for key in self.keys}
        self.legend = []
        self.precs = []
        self.recalls = []
        self.accs = []

    def _parse_name(self, name):
        # this is hardcoded
        split_name = name.split('_')
        temp_keys = {}
        
        for i in range(len(split_name)):
            key = split_name[i].split('-')[0]
            if key in self.headers.keys():
                temp_keys[key] = split_name[i].split('-')[1]
        
        return temp_keys
        
    def add_pr_curves(self, pth, all_prec, all_recall, data, acc=None):
        parsed_name = self._parse_name(pth) 

        for key in parsed_name.keys():
            if key in self.headers.keys():
                self.headers[key].append(parsed_name[key])

        self.legend.append(pth)
        self.precs.append(all_prec)
        self.recalls.append(all_recall)
        self.accs.append(acc)
    
    def save_pr_curves(self, pth):
        df = pd.DataFrame(self.headers)
        df['legend'] = self.legend
        df['precs'] = self.precs
        df['recalls'] = self.recalls
        df['accs'] = self.accs

        df = df.sort_values(by=['dataset','model','balanced'])
        dsets = df['dataset'].unique()

        

        for dset in dsets:
            temp_df = df.loc[df['dataset'] == dset]        

            fig, ax = plt.subplots(figsize=(10,6))
            
            for i in range(len(temp_df.precs.tolist())):
                ax.plot(temp_df.recalls.tolist()[i], temp_df.precs.tolist()[i])

            ax.set_title('Precision-Recall Curves')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')

            temp_legend = temp_df.model.tolist()
            temp_balanced = temp_df.balanced.tolist()
            for i in range(len(temp_balanced)):
                if 'True' in temp_balanced[i]:
                    temp_legend[i] += ', SMOTE'

            plt.legend(temp_legend, bbox_to_anchor=(1.04,1), loc="upper left")
            plt.tight_layout()
            fig.savefig(pth + f'{dset}.jpg')
            plt.close(fig)

def output_confmat(pth, data):
    df_cm = pd.DataFrame(data, index = ['negative', 'positive'],
                columns = ['pred negative', 'pred positive'])
    cm = sn.heatmap(df_cm, annot=True)
    fig = cm.figure

    print(data, pth, fig)
    fig.savefig(pth) 
    plt.close(fig)

class DataTracker(object):

    def __init__(self, keys=['dataset','balanced','model','cross'], metrics=['acc','auc_pr','precision','recall','f1','specificity','f1_avg','sp_avg']):

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
                temp_keys[key] = split_name[i].split('-')[1]
        
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
        df = df.loc[(df['groups'] == 'test')]

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
        
    def output_csv(self, output_dir):
        df = pd.DataFrame(self.names)
        df['groups'] = self.groups

        for key, val in self.metrics.items():
            if len(self.metrics[key]) > 0:
                df[key] = self.metrics[key]

        df = df.sort_values(by=['model','groups'])

        for group in ['max_fold','avg_fold','test','train']:
            temp_df = df.loc[(df['groups'] == group)]

            pth = f'data_summary_{group}.csv'

            temp_df.to_csv(os.path.join(output_dir,pth), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimenting flow')
    parser.add_argument('--json_dir', type=str, default='./output/04272022_fullrunv1_combined_focalfix')
    parser.add_argument('--output_dir', type=str, default='./output_analyze/04272022_fullrunv1_combined_focalfix')
    parser.add_argument('--choose_test', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    info_tracker = DataTracker()
    fold_pr_tracker = PRCurves()
    test_pr_tracker = PRCurves()
    
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
                    best_params_fold_scores = dict((el, []) for el in ['aucpr', 'f1-score', 'spc'])
                    best_params_strings = [f"{param}={best_params[param]}" for param in best_params.keys()]
                    for line in gridsearch_txt.readlines():
                        if all(param in line for param in best_params_strings):
                            test_str = [t.strip('()').split('=')[-1] for t in line.split(' ') if 'test=' in t]
                            # score_str = next(s for s in line.split(' ') if 'score=' in s)
                            for k, v in zip(best_params_fold_scores.keys(), test_str):
                                best_params_fold_scores[k].append(float(v))
                        if len(best_params_fold_scores) == 5:
                            break

                avg_fold_auc_pr = np.mean(best_params_fold_scores['aucpr'])
                avg_f1 = np.mean(best_params_fold_scores['f1-score'])
                avg_sp = np.mean(best_params_fold_scores['spc'])
                max_fold_idx = np.argmax(best_params_fold_scores['aucpr'])
                max_fold_auc_pr = best_params_fold_scores['aucpr'][max_fold_idx]
                
                conf_matrix_analyses = process_matrix([None, None, None])
                conf_matrix_analyses['auc_pr'] = [max_fold_auc_pr, avg_fold_auc_pr, np.nan]
                conf_matrix_analyses['f1_avg'] = (None,avg_f1,None)
                conf_matrix_analyses['sp_avg'] = (None,avg_sp,None)
                info_tracker.add_item(os.path.splitext(file)[0], conf_matrix_analyses, grouping=['max_fold', 'avg_fold', 'train'])


            elif file.endswith('.json'):
                os.chdir('./')
                file_path = os.path.join(root,file).replace('\\','/')

                print(file_path, os.path.isfile(file_path))
                json_file = json.load(open(file_path,'r'))

                if len(json_file.keys()) <= 0:
                    continue

                if 'val' in file:
                    fold_names = []
                    fold_acc = []
                    best_epoch_names = []
                    trn_pths = []
                    test_acc = []
                    best_test_epoch_names = []
                    fold_f1s = []
                    fold_sps = []

                    if 'dnn' in file:
                        for fold in json_file.keys():
                            
                            if 'test' in fold:
                                continue
                            #find best epoch
                            epoch_names = []
                            epoch_acc = []
                            epoch_test_acc = []
                            for i, epoch_dict in enumerate(json_file[fold]):
                                # epoch_names.append(epoch)
                                epoch_acc.append(epoch_dict[str(i)]['val']['auc_pr'])
                                epoch_test_acc.append(epoch_dict[str(i)]['test']['auc_pr'])

                            
                            best_epoch = np.argmax(epoch_acc)
                            best_epoch_names.append(best_epoch)

                            best_test_epoch = np.argmax(epoch_test_acc)
                            best_test_epoch_names.append(best_test_epoch)
                            test_acc.append(json_file[fold][best_test_epoch][str(best_test_epoch)]['test']['auc_pr'])

                            confmat = process_matrix([json_file[fold][best_test_epoch][str(best_test_epoch)]['val']['confusion_matrix']])
                            fold_f1s.append(confmat['f1'][0])
                            fold_sps.append(confmat['specificity'][0])

                            fold_names.append(fold)
                            fold_acc.append(json_file[fold][best_epoch][str(best_epoch)]['val']['auc_pr'])
                            
                            trn_pth = os.path.join(os.path.split(root)[0], 'train', file.replace('val','train'))
                            trn_pths.append(trn_pth)
                        

                        if args.choose_test:
                            max_fold_idx = np.argmax(test_acc)
                            best_epoch_names = best_test_epoch_names
                        else:
                            max_fold_idx = np.argmax(fold_acc)
                            
                        max_fold_acc = fold_acc[max_fold_idx]
                        avg_fold_acc = np.mean(fold_acc)
                        avg_f1 = np.mean(fold_f1s)
                        avg_sp = np.mean(fold_sps)

                        max_fold_mtx = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['val']['confusion_matrix']
                        max_fold_precs = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['val']['precisions']
                        max_fold_recalls = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['val']['recalls']

                        # trn_file = json.load(open(trn_pths[max_fold_idx],'r'))                      
                        # max_fold_trn_acc = trn_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['auc_pr']
                        # max_fold_trn_mtx = trn_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['confusion_matrix']
                        # max_fold_trn_precs = trn_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['precisions']
                        # max_fold_trn_recalls = trn_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['recalls']

                        tst_acc = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['test']['auc_pr']
                        tst_mtx = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['test']['confusion_matrix']
                        tst_precs = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['test']['precisions']
                        tst_recalls = json_file[fold_names[max_fold_idx]][best_epoch_names[max_fold_idx]][str(best_epoch_names[max_fold_idx])]['test']['recalls']

                        # info_tracker.add_item(os.path.splitext(file)[0], {'acc': (max_fold_acc,avg_fold_acc,tst_acc,max_fold_trn_acc)})
                        conf_matrix_analyses = process_matrix([max_fold_mtx, None, tst_mtx, None])
                        conf_matrix_analyses['auc_pr'] = (max_fold_acc,avg_fold_acc,tst_acc,None)
                        conf_matrix_analyses['f1_avg'] = (None,avg_f1,None,None)
                        conf_matrix_analyses['sp_avg'] = (None,avg_sp,None,None)
                        info_tracker.add_item(os.path.splitext(file)[0], conf_matrix_analyses)

                        # output best fold acc, avg fold accs, best test acc, trn_acc
                        
                    else:
                        if 'kmeans' in file:
                            for fold in json_file.keys():
                                if 'test' not in fold:
                                    fold_names.append(fold)
                                    fold_acc.append(json_file[fold]['auc_pr'])

                                    confmat = process_matrix([json_file[fold]['confusion_matrix']])
                                    fold_f1s.append(confmat['f1'][0])
                                    fold_sps.append(confmat['specificity'][0])
                            
                            avg_fold_acc = np.mean(fold_acc)
                            avg_sp = np.mean(fold_sps)
                            avg_f1 = np.mean(fold_f1s)

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
                            conf_matrix_analyses['f1_avg'] = (None,avg_f1,None,None)
                            conf_matrix_analyses['sp_avg'] = (None,avg_sp,None,None)
                            info_tracker.add_item(os.path.splitext(file)[0], conf_matrix_analyses)
                        else:
                            try:
                                max_fold_mtx = json_file['0']['confusion_matrix']
                                max_fold_precs = json_file['0']['precisions']
                                max_fold_recalls = json_file['0']['recalls']
                                
                                tst_acc = json_file['0']['auc_pr']
                                tst_mtx = json_file['0']['confusion_matrix']
                                tst_precs = json_file['0']['precisions']
                                tst_recalls = json_file['0']['recalls']
                            except:
                                max_fold_mtx = json_file['test']['confusion_matrix']
                                max_fold_precs = json_file['test']['precisions']
                                max_fold_recalls = json_file['test']['recalls']
                                
                                tst_acc = json_file['test']['auc_pr']
                                tst_mtx = json_file['test']['confusion_matrix']
                                tst_precs = json_file['test']['precisions']
                                tst_recalls = json_file['test']['recalls']

                            conf_matrix_analyses = process_matrix([tst_mtx])
                            conf_matrix_analyses['auc_pr'] = [tst_acc]
                            conf_matrix_analyses['f1_avg'] = [np.nan]
                            conf_matrix_analyses['sp_avg'] = [np.nan]
                            info_tracker.add_item(os.path.splitext(file)[0], conf_matrix_analyses, grouping=['test'])

                    # output_confmat(os.path.join(args.output_dir, os.path.splitext(file)[0] + '_maxfold_confmat.jpg'), max_fold_mtx)
                    # output_confmat(os.path.join(args.output_dir, os.path.splitext(file)[0] + '_test_confmat.jpg'), tst_mtx)
                    fold_pr_tracker.add_pr_curves(os.path.splitext(file)[0], max_fold_precs, max_fold_recalls, max_fold_mtx)
                    test_pr_tracker.add_pr_curves(os.path.splitext(file)[0], tst_precs, tst_recalls, tst_mtx, acc=tst_acc)

    # info_tracker.output(args.output_dir,metric='auc_pr')
    # info_tracker.output(args.output_dir,metric='f1')
    # info_tracker.output(args.output_dir,metric='specificity')
    info_tracker.output_csv(args.output_dir)
    fold_pr_tracker.save_pr_curves(os.path.join(args.output_dir,'fold'))
    test_pr_tracker.save_pr_curves(os.path.join(args.output_dir,'test'))



                
                
                
