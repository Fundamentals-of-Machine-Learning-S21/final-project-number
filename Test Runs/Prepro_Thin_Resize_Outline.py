# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:28:11 2021

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
    # np.save('data_outlines', data_outlines)
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

def PrePro_LDA(X, y):
    LDA_var = []
    for i in range(10):
        n_components = i
        lda_numbers = LDA(n_components=n_components)
        lda_numbers.fit(X, y)
        total_var = lda_numbers.explained_variance_ratio_.sum() * 100
        LDA_var.append(total_var)
    LDA_var = np.array(LDA_var)

    #print(np.where(LDA_var>=90))
    print('minimum number of principal components you need to preserve in order to explain at least 90% of the data is: ',
          np.amin(np.where(LDA_var>=90)))

    n_components = np.amin(np.where(LDA_var>=90))
    # n_components = 9
    lda = LDA(n_components=n_components)
    X_LDA = lda.fit_transform(X, y)
    return lda, X_LDA

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

#no PCA or LDA
np.save('data_preprocessed', Preprocessed_data)

#run PCA, retaining 90%
n_components, pca, X_PCA = PrePro_PCA(Preprocessed_data)
np.save('data_preprocessed_PCA', X_PCA)

#run LDA, retaining 90%
lda, X_LDA = PrePro_LDA(Preprocessed_data, labels)
np.save('data_preprocessed_LDA', X_LDA)















