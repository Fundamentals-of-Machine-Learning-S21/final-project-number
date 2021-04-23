# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 19:17:43 2021

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



def splitData(trainingData, labels):
    arr= trainingData.T
    arr2 = labels
    # Min-max scaling the data
    scaler1 = MinMaxScaler()
    normalized_arr2 = scaler1.fit_transform(arr)
    # Normalizing the data using Normalizer ()
    scaler2 = Normalizer()
    normalized_arr_n = scaler2.fit_transform(arr)
    # Normalizing the data using StandardScaler
    scaler3 = StandardScaler()
    normalized_arr_ss = scaler3.fit_transform(arr)
    # Split the data
    #train-test split as 70% train and 30% test using normalized data with Min-max scaling the data
    X_train, X_test, y_train, y_test = train_test_split(normalized_arr2, arr2, test_size=0.30, random_state=42) 
    print('X_train: ',X_train.shape, 'X_test: ',X_test.shape, 'y_train: ',y_train.shape, 'y_test: ',y_test.shape)
    y_binarized_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_binarized_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_classes = y_binarized_train.shape[1]
    return X_train,X_test,y_train,y_test,y_binarized_train

def naiveBayesClassifier(X_train,X_test,y_train,y_test,y_binarized_train):
    # Fit Naive Bayes Classifier on training data
    gnb = GaussianNB()
    classifier_gnb = gnb.fit(X_train, y_train)
    nBC_y_pred = classifier_gnb.predict(X_test)
    nBC_acc = accuracy_score(y_test, nBC_y_pred)
    print('k-NN overall accuracy',accuracy_score(y_test, nBC_y_pred))
    #Confusion Matrix, created with nice visualization:)
    disp = plot_confusion_matrix(gnb, X_test, y_test,
                                 cmap=plt.cm.Blues)
    print(disp.confusion_matrix)
    plt.show()
    classifier = OneVsRestClassifier(GaussianNB())
    nBC_y_score = classifier.fit(X_train, y_binarized_train).predict_proba(X_test)
    return nBC_y_pred, nBC_y_score, nBC_acc

def ROC_curve(y_train, y_test):
    y_binarized_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_binarized_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_classes = y_binarized_train.shape[1]
    # each class against the others
    classifier = OneVsRestClassifier(GaussianNB())
    y_score = classifier.fit(X_train, y_binarized_train).predict_proba(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_binarized_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ## Plot of a ROC curve for a specific class (class label 2)
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    ## Plot of a ROC curve for all classes with overall micro and macro averages
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 10))
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

    colors = cycle(mcolors.TABLEAU_COLORS)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]), alpha = 0.5)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


# Import Data from Directory
trainingData = np.load('data_train.npy')
print(trainingData.shape)
labels = np.load('labels_train.npy')
print(labels.shape)

# Split the Data
X_train,X_test,y_train,y_test,y_binarized_train = splitData(trainingData, labels)

# Run Naive Bayes Classifier
nBC_y_pred, nBC_y_score, nBC_acc = naiveBayesClassifier(X_train,X_test,y_train,y_test,y_binarized_train)

# Build ROV Curves
ROC_curve(y_train, y_test)












