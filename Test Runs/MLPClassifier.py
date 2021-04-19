# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:31:24 2021

@author: carte
"""
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np 
import numpy.matlib
import pandas as pd 

from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# data_preprocesssed = np.load('data_preprocessed.npy')
# data_preprocesssed = np.load('data_preprocessed_LDA.npy', allow_pickle=True)
data_preprocesssed = np.load('data_preprocessed_PCA.npy')
print('raw data shape', data_preprocesssed.shape)
labels = np.load('labels_train.npy')
print(labels.shape)

X = data_preprocesssed
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

net = MLPClassifier(hidden_layer_sizes=(1000, 1000), max_iter=10000, alpha=0.13,
                    solver='sgd', random_state=42, learning_rate = 'adaptive',
                    learning_rate_init=.08)

net.fit(X_train, y_train)

print('TRAINING PERFORMANCE')
y_predtrain = net.predict(X_train)
accuracy_train = np.round(accuracy_score(y_train, y_predtrain),2)
print('Accuracy in the Train set= ', accuracy_train*100, '%')
# print(classification_report(y_train, y_predtrain))
print('----------------------------------------------------------')

print('TEST PERFORMANCE')
y_predtest = net.predict(X_test)
accuracy_test = np.round(accuracy_score(y_test, y_predtest),2)
print('Accuracy in the Test set= ', accuracy_test*100, '%')
# print(classification_report(y_test, y_predtest))
print('number of iterations before convergence: ', net.n_iter_)
print('Confusion matrix in Test:')
print(confusion_matrix(y_test, y_predtest))






# data_preprocesssed2 = np.load('data_preprocessed.npy')
# # data_preprocesssed2 = np.load('data_preprocessed_LDA.npy', allow_pickle=True)
# # data_preprocesssed2 = np.load('data_preprocessed_PCA.npy')
# print('raw data shape', data_preprocesssed2.shape)
# labels = np.load('labels_train.npy')
# print(labels.shape)

# X2 = data_preprocesssed2
# y = labels

# X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=42)

# net = MLPClassifier(hidden_layer_sizes=(250, 200), max_iter=10000, alpha=0.01,
#                     solver='sgd', random_state=42, learning_rate = 'adaptive',
#                     learning_rate_init=.0001)

# net.fit(X_train2, y_train2)

# print('TRAINING PERFORMANCE')
# y_predtrain2 = net.predict(X_train2)
# accuracy_train = np.round(accuracy_score(y_train2, y_predtrain2),2)
# print('Accuracy in the Train set= ', accuracy_train*100, '%')
# # print(classification_report(y_train, y_predtrain))
# print('----------------------------------------------------------')

# print('TEST PERFORMANCE')
# y_predtest2 = net.predict(X_test2)
# accuracy_test = np.round(accuracy_score(y_test2, y_predtest2),2)
# print('Accuracy in the Test set= ', accuracy_test*100, '%')
# # print(classification_report(y_test, y_predtest))
# print('number of iterations before convergence: ', net.n_iter_)
# print('Confusion matrix in Test:')
# print(confusion_matrix(y_test2, y_predtest2))









# net_hidden_layers = [20, 20] #consider a set of parameters
# net_learning_rate = [0.010, 0.020] #consider a set of parameters
# epochs = [5000, 10000]  #consider a set of parameters
# # epochs = [20000, 100000]  #consider a set of parameters

# for i in net_hidden_layers:
#     for j in net_learning_rate:
#         for k in epochs:
#             net.set_params(hidden_layer_sizes = i, learning_rate_init = j, max_iter = k)
#             net.fit(X_train, y_train)
#             y_pred = net.predict(X_train)

#             acc_score = accuracy_score(y_train, y_pred)
#             print('-----------------------------------')
#             print('Hidden Layer Architecture: ', i)
#             print('Learning Rate: ', j)
#             print('Number of Epochs: ', k)
#             print('Accuracy = ', np.round(acc_score*100,2),'%')
#             print('Test Accuracy =', np.round(net.score(X_test, y_test)*100,2),'%')
#             print('-----------------------------------')



# import matplotlib.pyplot as plt
# from itertools import cycle

# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
# from sklearn.metrics import roc_auc_score
            

# y_binarized_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# y_binarized_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# n_classes = y_binarized_train.shape[1]

# y_train

# y_binarized_train

# # each class against the others
# classifier = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(200, 150), max_iter=10000, alpha=0.13,
#                     solver='sgd', random_state=42, learning_rate = 'adaptive',
#                     learning_rate_init=.08))

# y_score = classifier.fit(X_train, y_binarized_train).predict_proba(X_test)

# y_score

# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_binarized_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# ## Plot of a ROC curve for a specific class (class label 2)
# plt.figure()
# lw = 2
# plt.plot(fpr[2], tpr[2], color='darkorange',
#           lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()


# import matplotlib.colors as mcolors
# ## Plot of a ROC curve for all classes with overall micro and macro averages
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# # Finally average it and compute AUC
# mean_tpr /= n_classes

# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# # Plot all ROC curves
# plt.figure(figsize=(10, 10))
# plt.plot(fpr["micro"], tpr["micro"],
#           label='micro-average ROC curve (area = {0:0.2f})'
#                 ''.format(roc_auc["micro"]),
#           color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#           label='macro-average ROC curve (area = {0:0.2f})'
#                 ''.format(roc_auc["macro"]),
#           color='navy', linestyle=':', linewidth=4)

# colors = cycle(mcolors.TABLEAU_COLORS)
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#               label='ROC curve of class {0} (area = {1:0.2f})'
#               ''.format(i, roc_auc[i]), alpha = 0.5)

# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()
