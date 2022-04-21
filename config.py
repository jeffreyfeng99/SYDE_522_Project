from datetime import datetime

# general
dataset_root = 'normalized_datasets'
datasets = ['normalized_uci_df', 'normalized_zigong_df', 'normalized_uci_and_zigong_df']

all_problems = {'readmissiontime_multiclass': 4, # TODO look into other imblearn that can work on smol ds
                'deathtime_multiclass': 5,
                'readmissiontime': 'all', 
                'deathtime': 'all', 
                'death': 2}

output_json_root = f'output_jsons_test/{datetime.now().strftime("%m%d%Y")}'

blank_model_path = 'models/blank_model.pth'

k = 5
rand_state = None

# smote
smote_kwargs = {}


# dnn
lr = 0.05
momentum = 0.9
weight_decay = 0.0005
num_epochs = 50
batch_size = 16
num_workers = 4

ce_kwargs = {'label_smoothing': 0.}


# focal
focal_kwargs = {'alpha': 2., 'reduction': 'mean'}


# kmeans


# svm

svm_param_grid = {
    'kernel': ['linear', 'rbf'],
    'gamma': [0.01, 0.1, 1.0, 10.],
    'C': [0.01, 0.1, 1.0, 10.]
}