# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 19:31:33 2021

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



# Model 3: SVM Model with RBF Kernel
C = [1500, 2000, 2500]

def RBF_Kernel(X_train, y_train, X_test, y_test, C):
    svc = svm.SVC()
    # Seems like we can improve performance including higher values for c greater than 1500
    parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': C} 
    clf = GridSearchCV(svc, parameters, scoring='accuracy')
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print("Mean cross-validated score of the best_estimator:")
    print()
    print(clf.best_score_)
    # Accuracy for different settings on development set
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, RBF_y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, RBF_y_pred))
        print()
    # Overall accuracy
    RBF_acc = accuracy_score(y_true, RBF_y_pred)
    print('Model 3 overall accuracy',accuracy_score(y_true, RBF_y_pred))
    #Confusion Matrix, created with nice visualization:)
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues)
    print(disp.confusion_matrix)
    plt.show()
    return RBF_acc, RBF_y_pred






