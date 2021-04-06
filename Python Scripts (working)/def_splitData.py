# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:27:25 2021

@author: carte
"""

from itertools import cycle
from scipy import interp
from scipy.stats import multivariate_normal
from sklearn import svm
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd





# plt.style.use('bmh')
# arr1 = np.load('data_train.npy')
# for r in arr1:
#     row = []
#     for c in r:
#         row.append(c[0])
#     arr.append(row)
# arr = np.array(arr)
# print(arr.shape)

# arr2 = np.load('labels_train.npy')
trainingData = np.load('data_train.npy')
labels = np.load('labels_train.npy')

def splitData(trainingData, labels):
    arr= trainingData.T
    arr2 = labels
    # Min-max scaling the data
    scaler = MinMaxScaler()
    normalized_arr2 = scaler.fit_transform(arr)
    # Normalizing the data using Normalizer ()
    scaler = Normalizer()
    normalized_arr_n = scaler.fit_transform(arr)
    # Normalizing the data using StandardScaler
    scaler = StandardScaler()
    normalized_arr_ss = scaler.fit_transform(arr)
    # Split the data
    #train-test split as 70% train and 30% test using normalized data with Min-max scaling the data
    X_train, X_test, y_train, y_test = train_test_split(normalized_arr2, arr2, test_size=0.30, random_state=42) 
    print('X_train: ',X_train.shape, 'X_test: ',X_test.shape, 'y_train: ',y_train.shape, 'y_test: ',y_test.shape)
    y_binarized_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_binarized_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_classes = y_binarized_train.shape[1]

    return(X_train,X_test,y_train,y_test,y_binarized_train,y_binarized_test,n_classes)

