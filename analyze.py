from distutils.log import info
import os
import argparse
import json
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def output_confmat(pth, data):
    df_cm = pd.DataFrame(data, index = ['negative', 'positive'],
                columns = ['pred negative', 'pred positive'])
    cm = sn.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig(pth) 

class DataTracker(object):

    def __init__(self, keys=['dataset','balanced','model','cross']):

        self.keys = keys
        self.names = {}
        for key in keys:
            self.names[key] = []

        self.groups = []
        self.metrics = []
    
    def _parse_name(self, name):
        # this is hardcoded

        split_name = name.split('-')

        temp_keys = {}
        
        for i in range(len(split_name)):
            key = split_name.split('-')[0]
            if key in self.keys:
                temp_keys[key] = split_name.split('-')[1]
        
        return temp_keys

    def add_item(self, name, metrics, grouping=['max_fold','avg_fold','test','train']):
        assert len(grouping) == len(metrics)

        parsed_name = self._parse_name(name) 

        for i in range(len(metrics)):

            for key in parsed_name.keys():
                self.names[key].append(parsed_name[key])
            
            self.groups.append(grouping[i])
            self.metrics.append(metrics[i])
    
    def output(self, output_dir):

        df = pd.DataFrame(self.names)
        df['groups'] = self.groups
        df['metrics'] = self.metrics

        dsets = df['datasets'].unique()

        for dset in dsets:
            dset_specific = df.loc[df['datasets'] == dset]
            
            crosses = dset_specific['cross'].unique()

            for cross in crosses:
                cross_specific = dset_specific.loc[dset_specific['datasets'] == cross]

                bar = sn.catplot(x='model', y='metrics', hue='groups', data=cross_specific, kind='bar')
                
                pth = f'dataset-{dset}_cross-{cross}.jpg'

                fig = bar.get_figure()
                fig.savefig(os.path.join(output_dir,pth)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimenting flow')
    parser.add_argument('--json_dir', type=str, default='./output_jsons')
    parser.add_argument('--output_dir', type=str, default='./output_figures')

    args = parser.parse_args()

    info_tracker = DataTracker()
    
    for root, dirs, files in os.walk(args.json_dir):

        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root,file)
                json_file = json.load(open(file_path,'r'))

                if len(json_file.keys()) <= 0:
                    continue

                if 'val' in file:
                    fold_names = []
                    fold_acc = []
                    best_epoch_names = []
                    trn_acc = []

                    if 'dnn' in file:
                        for fold in json_file.keys():

                            #find best epoch
                            epoch_names = []
                            epoch_acc = []
                            for epoch in json_file[fold]['validation']:
                                epoch_names.append(epoch)
                                epoch_acc.append(json_file[fold]['validation'][epoch]['accuracy'])
                            
                            best_epoch = np.argmax(epoch_acc)
                            best_epoch_names.append(epoch_names[best_epoch])

                            fold_names.append(fold)
                            fold_acc.append(json_file[fold]['validation'][best_epoch]['accuracy'])

                            trn_pth = os.path.join(os.path.split(root)[0], 'train', file.replace('val','train'))
                            trn_file = json.load(open(trn_pth,'r'))
                            trn_acc.append(trn_file[fold][best_epoch]['accuracy'])
                        
                        avg_fold_acc = np.mean(fold_acc)
                        max_fold_idx = np.argmax(fold_acc)
                        max_fold_acc = fold_acc(max_fold_idx)
                        max_fold_mtx = json_file[fold_names[max_fold_idx]]['validation'][best_epoch_names[max_fold_idx]]['confusion_matrix']

                        max_fold_acc_trn = trn_acc(max_fold_idx)

                        tst_acc = json_file[fold_names[max_fold_idx]]['test'][best_epoch_names[max_fold_idx]]['accuracy']
                        tst_mtx = json_file[fold_names[max_fold_idx]]['test'][best_epoch_names[max_fold_idx]]['confusion_matrix']
                        
                        info_tracker.add_item(os.path.basename(file), (max_fold_acc,avg_fold_acc,tst_acc,max_fold_acc_trn))

                        # output best fold acc, avg fold accs, best test acc, trn_acc
                        

                    else:
                        for fold in json_file.keys():
                            if 'test' not in fold:
                                fold_names.append(fold)
                                fold_acc.append(json_file[fold]['accuracy'])
                        
                        avg_fold_acc = np.mean(fold_acc)
                        max_fold_idx = np.argmax(fold_acc)
                        max_fold_acc = fold_acc(max_fold_idx)
                        max_fold_mtx = json_file[fold_names[max_fold_idx]]['confusion_matrix']

                        tst_acc = json_file['test']['accuracy']
                        tst_mtx = json_file['test']['confusion_matrix']

                        info_tracker.add_item(os.path.basename(file), (max_fold_acc,avg_fold_acc,tst_acc,np.nan))

                    output_confmat(os.path.join(args.output_dir, os.path.basename(file) + '_maxfold.jpg'), max_fold_mtx)
                    output_confmat(os.path.join(args.output_dir, os.path.basename(file) + '_test.jpg'), tst_mtx)

    info_tracker.output(args.output_dir)


                
                
                
