from models import dnn
from models import kmeans
from models import losses
from config import *

from sklearn import svm
from sklearn.cluster import KMeans
import torch.nn as nn


def Predictor(name, n_features=9, n_classes=5, train_json_base_filename="", val_json_base_filename=""):
    train_json_base_filename += f"_model-{name}"
    val_json_base_filename += f"_model-{name}"
 
    deep = False
    if name == 'dnn':
        deep = True
        train_json_base_filename += f"_lr-{lr}_mom-{momentum}"
        val_json_base_filename += f"_lr-{lr}_mom-{momentum}"
        return dnn.DNNet(n_features=n_features, n_classes=n_classes), deep, train_json_base_filename, val_json_base_filename
    elif name == 'kmeans':
        val_json_base_filename += f"_nc-{n_classes}"
        return KMeans(n_clusters=n_classes), deep, train_json_base_filename, val_json_base_filename
    elif name == 'svm':
        return svm.SVC(probability=True), deep, train_json_base_filename, val_json_base_filename
    
    

def Loss(name):
    if name == 'ce':
        return nn.CrossEntropyLoss(**ce_kwargs)
    elif name == 'focal':
        return losses.FocalLoss(**focal_kwargs)
    elif name == 'msfe':
        return losses.MSFELoss()
    else:
        raise ValueError(f'loss function {name} is not implemented')

