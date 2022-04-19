from models import dnn
from models import kmeans
from models import svm
from models import losses
from config import *

import torch.nn.functional as F


def Predictor(name):
    deep = False
    if name == 'dnn':
        deep = True
        return dnn.DNNet(), deep
    elif name == 'kmeans':
        return kmeans.KMeans_(N_CLUSTERS), deep
    elif name == 'svm':
        return svm.SVM_(KERNEL, GAMMA, C), deep

def Loss(name):
    if name == 'ce':
        return F.cross_entropy(**ce_kwargs)
    elif name == 'focal':
        return losses.FocalLoss()
    elif name == 'msfe':
        return losses.MSFELoss()
    else:
        raise ValueError(f'loss function {name} is not implemented')

