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

data_preprocesssed = np.load('data_preprocessed.npy')
print('raw data shape', data_preprocesssed.shape)
labels = np.load('labels_train.npy')
print(labels.shape)


from sklearn.neural_network import MLPClassifier

X = data_preprocesssed
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# net = MLPClassifier(activation='tanh', hidden_layer_sizes=(120,80,50), solver='adam',
# net = MLPClassifier(activation='tanh', hidden_layer_sizes=(200,100), solver='adam',
#                     learning_rate = 'adaptive', learning_rate_init=0.00025, max_iter=10000, 
#                     shuffle = True, random_state=42, n_iter_no_change = 1000)

net = MLPClassifier(hidden_layer_sizes=(330, 330,330), max_iter=10000, alpha=0.13,
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
            

