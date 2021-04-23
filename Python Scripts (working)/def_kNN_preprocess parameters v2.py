# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 12:34:14 2021

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
import numpy.random as npr
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter

# https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7 

# Resize
def resized(input_data, dimensions):
    Data_train_resized = []
    input_data = np.array(input_data)
    input_data2 = input_data.T[0]
    for i in range(len(data_train[1])):
        img = input_data2[:,i]
        # img = input_data[:,i]
        img = img.reshape(300,300)
        dim = (dimensions,dimensions)
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        resized_img = resized_img.reshape(np.square(dimensions))
        Data_train_resized.append(resized_img)
    
    data_resized = np.array(Data_train_resized).T
    resolution2 = np.sqrt(data_resized.shape[0])
    print('Resized Training Data ='+str(data_resized.shape[1])+' samples '+'at '+ str(resolution2) + 'x' + str(resolution2)+' resolution')

    # for i in range(1):
    #     rnd_sample = npr.permutation(np.where(labels==i)[0])
    #     fig=plt.figure(figsize=(15,15))
    #     for j in range(25):
    #         fig.add_subplot(5,5,j+1)
    #         plt.imshow(256-data_resized[:,rnd_sample[j]].reshape((dimensions,dimensions)),cmap='gray')
    #         plt.axis('off');plt.title('Digit RESIZE '+str(int(labels[rnd_sample[j]])),size=15)
            
    return data_resized

# Binarization
def threshold(input_data):
    data_thresh = []
    for i in range(len(input_data[1])):
        img = input_data[:,i] 
        thresh_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY,11,5)
        # ret, thresh_img = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)
        data_thresh.append(thresh_img)
    # np.save('data_thresh', data_thresh)
    data_thresh = np.array(data_thresh)
    return data_thresh

# Skew Correction
def skew_Correct(input_data, skew_limit, skew_delta):
    data_skew = []
    for i in range(len(input_data[1])):
        img = input_data[:,i] 
        
        def find_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            hist = np.sum(data, axis=1)
            score = np.sum((hist[1:] - hist[:-1]) ** 2)
            return hist, score
        
        delta = skew_delta
        limit = skew_limit
        angles = np.arange(-limit, limit+delta, delta)
        scores = []
        for angle in angles:
            hist, score = find_score(img, angle)
            scores.append(score)
        
        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]
        # print('Best angle: ', best_angle)
        img_skew = inter.rotate(img, best_angle, reshape=False, order=0)
        data_skew.append(img_skew)
    data_skew = np.array(data_skew)
    return data_skew

# Thinning
def thinned(input_data, thin_kernel):
    data_eroded = []
    for i in range(len(input_data[1])):
        img = input_data[:,i]
        # img = img.reshape(300,300)
        kernel_value = thin_kernel
        kernel = np.ones((kernel_value,kernel_value),np.uint8)
        eroded_img = cv2.erode(img,kernel,iterations = 3)
        data_eroded.append(eroded_img)

    data_eroded = np.array(data_eroded)
    return data_eroded

# Outlines
def outline(input_data, outline_kernel):
    data_outlines = []
    for i in range(len(data_train[1])):
        img = data_train[:,i]
        kernel = np.ones((outline_kernel,outline_kernel),np.uint8)
        outlines_img = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
        data_outlines.append(outlines_img)
    np.save('data_outlines', data_outlines)
    data_outlines = np.array(data_outlines)
    
    print('Outline kernel value set to: '+ str(outline_kernel))
    return data_outlines

# Noise Removal
def opened(input_data, open_kernel):
    data_opened = []
    for i in range(len(data_train[1])):
        img = data_train[:,i]
        # img = img.reshape(300,300)
        kernel_value = open_kernel
        kernel = np.ones((kernel_value,kernel_value),np.uint8)
        opened_img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
        data_opened.append(opened_img)
    # np.save('data_opened', data_opened)
    data_opened = np.array(data_opened)
    return data_opened

def preprocess_Data(input_data, dimensions, skew_limit, skew_delta,  thin_kernel, outline_kernel, open_kernel):

    Thresholding = threshold(input_data)
    SkewCorrect = skew_Correct(Thresholding, skew_limit, skew_delta)
    Thinned = thinned(SkewCorrect, thin_kernel)
    Outlined = outline(Thinned, outline_kernel)
    Opened = opened(Outlined, open_kernel)
    Resized = resized(Opened, dimensions)
    
    return Resized


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
    return(X_train,X_test,y_train,y_test)

def kNN(X_train,X_test,y_train,y_test):
    # create new a knn model
    knn2 = KNeighborsClassifier()
    # create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(3, 25)}
    # use gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
    # fit model to data
    knn_gscv.fit(X_train, y_train)
    # check top performing n_neighbors value
    best_k=knn_gscv.best_params_
    best_k
    # check mean score for the top performing value of n_neighbors
    knn_gscv.best_score_
    knn_gscv.best_params_
    knn = KNeighborsClassifier(n_neighbors = best_k['n_neighbors'])
    knn.fit(X_train,y_train)
    kNN_y_pred =knn.predict(X_test)
    # check accuracy of our model on the test data
    kNN_score = knn.score(X_test, y_test)
    print('k-NN score',kNN_score)
    # overall accuracy on test data
    kNN_acc = accuracy_score(y_test, kNN_y_pred)
    print('k-NN overall accuracy',kNN_acc)
    # Confusion matrix on test data
    confusion_matrix(y_test, kNN_y_pred)
    #Confusion Matrix, created with nice visualization:)
    disp = plot_confusion_matrix(knn, X_test, y_test,
                                 cmap=plt.cm.Blues)
    print(disp.confusion_matrix)
    plt.show()
    return kNN_y_pred, kNN_acc, kNN_score, best_k

def ROC_curve(X_train, X_test, y_train, y_test, best_k):
    y_binarized_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_binarized_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_classes = y_binarized_train.shape[1]
    # each class against the others
    classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = best_k['n_neighbors']))
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
data_train = np.load('data_train.npy')
print('raw data shape', data_train.shape)
labels = np.load('labels_train.npy')
print(labels.shape)

"""
Input pre-processing parameters here
"""
Training_Data = data_train
dimensions = 30
skew_limit = 30
skew_delta = 1
thin_kernel = 20
outline_kernel = 20
open_kernel = 20  

Preprocessed_data = preprocess_Data(Training_Data, skew_limit, skew_delta, thin_kernel, outline_kernel, dimensions, open_kernel)
trainingData = Preprocessed_data
print('pre-processed data shape', trainingData.shape)
print(labels.shape)

# Split the Data
X_train,X_test,y_train,y_test = splitData(trainingData, labels)

# Run k-NN
kNN_y_pred, kNN_acc, kNN_score, best_k = kNN(X_train,X_test,y_train,y_test)
print('accuracy = ', kNN_score, ', ',kNN_acc )

# Build ROV Curves
#ROC_curve(y_train, y_test, best_k)




