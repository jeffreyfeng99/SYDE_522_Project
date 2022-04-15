import dnn
import kmeans
import svm
from ..config import *


def Predictor(name):
    if name == 'dnn':
        return dnn.DNNet()
    elif name == 'kmeans':
        return kmeans.KMeans_(N_CLUSTERS)
    elif name == 'svm':
        return svm.SVM_(KERNEL, GAMMA, C)