# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:03:08 2021

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


# Outlines
def outline(input_data, outline_kernel):
    data_outlines = []
    for i in range(len(input_data[1])):
        img = input_data[:,i]
        kernel = np.ones((outline_kernel,outline_kernel),np.uint8)
        outlines_img = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
        data_outlines.append(outlines_img)
    np.save('data_outlines', data_outlines)
    data_outlines = np.array(data_outlines)
    
    print('Outline kernel value set to: '+ str(outline_kernel))
    # print('Outline image output data size: '+ str(data_outlines.shape)) 
    return data_outlines

# Thinning
def thinned(input_data, thin_kernel):
    data_eroded = []
    for i in range(len(input_data[1])):
        img = input_data[:,i]
        # img = img.reshape(300,300)
        kernel_value = thin_kernel
        kernel = np.ones((kernel_value,kernel_value),np.uint8)
        eroded_img = cv2.erode(img,kernel,iterations = 1)
        data_eroded.append(eroded_img)

    data_eroded = np.array(data_eroded)
    print('Thinning kernel value set to: '+ str(thin_kernel))
    # print('Thinning image output data size: '+ str(data_eroded.shape)) 
    return data_eroded

# Resizing - if one PreP before resize, input_data1. if two, input data2
def resized(input_data, dimensions):
    Data_train_resized = []
    input_data2 = np.array(input_data)
    input_data1 = input_data2.T[0]
    for i in range(len(data_train[1])):
        img = input_data1[:,i]
        # img = input_data2[:,i]
        img = img.reshape(300,300)
        dim = (dimensions,dimensions)
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        resized_img = resized_img.reshape(np.square(dimensions))
        Data_train_resized.append(resized_img)
    
    data_resized = np.array(Data_train_resized).T
    resolution2 = np.sqrt(data_resized.shape[0])
    print('Resized Training Data ='+str(data_resized.shape[1])+' samples '+'at '+ str(resolution2) + 'x' + str(resolution2)+' resolution')
    return data_resized

def preprocess_Data(input_data, dimensions, thin_kernel, outline_kernel):
    Thinned = thinned(input_data, thin_kernel)     
    Resized = resized(Thinned, dimensions)
    Outlined = outline(Resized, outline_kernel)       
    return Outlined

# PCA
def PrePro_PCA(X_PCA):
    # Split the data
    # X_PCA = Preprocessed_data.T
    X_PCA = Preprocessed_data

    scaler = Normalizer()
    print('Scaler used on the raw data is: ', scaler)
    X_PCA = scaler.fit_transform(X_PCA)
    
    # Number of Components required to preserve 90% of the data with PCA
    pca = PCA(0.9)
    pca.fit(X_PCA)
    print('minimum number of principal components you need to preserve in order to explain at least 90% of the data is: ',
          pca.n_components_)
    
    n_components = pca.n_components_
    pca = PCA(n_components=n_components)
    X_PCA = pca.fit_transform(X_PCA)
    
    return n_components, pca, X_PCA

# k-NN classifier
def k_NN(X_train, y_train, X_test, y_test, function):
    # k-NN classifier
    knn = KNeighborsClassifier(3)
    knn.fit(function.transform(X_train), y_train)
    # print('train shapes: ', X_train.shape, y_train.shape)
    # print('test shapes: ', X_test.shape, y_test.shape)
    acc_knn_train = knn.score(function.transform(X_train), y_train)
    acc_knn_test = knn.score(function.transform(X_test), y_test)
    
    print('training data k-NN, PCA accuracy: ','%.3f'%(acc_knn_train))
    print('testing data k-NN, PCA accuracy: ','%.3f'%(acc_knn_test))
    return 


data_train = np.load('data_train.npy')
print('raw data shape', data_train.shape)
labels = np.load('labels_train.npy')
print(labels.shape)

dimensions = 20
thin_kernel = 60
outline_kernel = 4

Training_Data = data_train

Preprocessed_data = preprocess_Data(Training_Data, dimensions, thin_kernel, outline_kernel)

Preprocessed_data = np.reshape(Preprocessed_data, (3360, -1))
print('pre-processed data shape', Preprocessed_data.shape)
print(labels.shape)

#run PCA, retaining 90%
n_components, pca, X_PCA = PrePro_PCA(Preprocessed_data)


from sklearn.neural_network import MLPClassifier

X = X_PCA
y = labels

net = MLPClassifier(activation='tanh',
                    n_iter_no_change = 1000)

net_hidden_layers = [20, 20] #consider a set of parameters
net_learning_rate = [0.001, 0.005, 0.010] #consider a set of parameters
epochs = [10000, 15000, 20000]  #consider a set of parameters

for i in net_hidden_layers:
    for j in net_learning_rate:
        for k in epochs:
            net.set_params(hidden_layer_sizes = i, learning_rate_init = j, max_iter = k)
            net.fit(X, y)
            y_pred = net.predict(X)

            acc_score = accuracy_score(y, y_pred)
            print('-----------------------------------')
            print('Hidden Layer Architecture: ', i)
            print('Learning Rate: ', j)
            print('Number of Epochs: ', k)
            print('Accuracy = ', np.round(acc_score*100,2),'%')
            print('-----------------------------------')









