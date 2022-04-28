import os

import numpy as np


if __name__ == "__main__":

    json_dir = './output/04272022_svm/jsons'
    
    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file.endswith('.txt') and 'svm' in file:
                print(f'we are in {file}\n')
                file_path = os.path.join(root,file)
                file = os.path.splitext(file)[0]
                best_params = dict((el, None) for el in ['kernel', 'gamma', 'C'])
                for key in file.split('_'):
                    if key.split('-')[0] in best_params.keys():
                        best_params[key.split('-')[0]] = key.split('-')[-1]
                
                print(best_params)

                with open(file_path, 'r') as gridsearch_txt:
                    best_params_fold_scores = dict((el, []) for el in ['aucpr', 'f1-score', 'spc'])
                    best_params_strings = [f"{param}={best_params[param]}" for param in best_params.keys()]
                    for line in gridsearch_txt.readlines():
                        if all(param in line for param in best_params_strings):
                            # print(line)
                            test_str = [t.strip('()').split('=')[-1] for t in line.split(' ') if 'test=' in t]
                            for k, v in zip(best_params_fold_scores.keys(), test_str):
                                best_params_fold_scores[k].append(float(v))
                            # score_str = next(s for s in line.split(' ') if 'score=' in s)
                            # best_params_fold_scores.append(float(score_str.split('=')[-1]))
                        if len(best_params_fold_scores) == 5:
                            break

                avg_fold_auc_pr = np.mean(best_params_fold_scores['aucpr'])
                avg_fold_f1 = np.mean(best_params_fold_scores['f1-score'])
                avg_fold_spc = np.mean(best_params_fold_scores['spc'])
                max_fold_idx = np.argmax(best_params_fold_scores)
                max_fold_auc_pr = best_params_fold_scores['aucpr'][max_fold_idx]
                max_fold_f1 = best_params_fold_scores['f1-score'][max_fold_idx]
                max_fold_spc = best_params_fold_scores['spc'][max_fold_idx]

                print(f'AVERAGE SCORES')
                print(f'aucpr: {avg_fold_auc_pr}, f1: {avg_fold_f1}, spc: {avg_fold_spc}\n')
                print(f'MAX SCORES')
                print(f'aucpr: {max_fold_auc_pr}, f1: {max_fold_f1}, spc: {max_fold_spc}\n\n')
