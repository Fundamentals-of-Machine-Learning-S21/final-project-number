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
    # input_data2 = input_data.T[0]
    for i in range(len(data_train[1])):
        # img = input_data2[:,i]
        img = input_data[:,i]
        img = img.reshape(300,300)
        dim = (dimensions,dimensions)
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        resized_img = resized_img.reshape(np.square(dimensions))
        Data_train_resized.append(resized_img)
    data_resized = np.array(Data_train_resized).T
    resolution2 = np.sqrt(data_resized.shape[0])
    print('Resized Training Data ='+str(data_resized.shape[1])+' samples '+'at '+ str(resolution2) + 'x' + str(resolution2)+' resolution')
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
    print('Thresholding: Adaptive, binary')
    return data_thresh

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
    print('Thinning kernel value set to: '+ str(thin_kernel))
    # print('Thinning image output data size: '+ str(data_eroded.shape)) 
    return data_eroded

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


# Noise Removal
def opened(input_data, open_kernel):
    data_opened = []
    for i in range(len(input_data[1])):
        img = input_data[:,i]
        # img = img.reshape(300,300)
        kernel_value = open_kernel
        kernel = np.ones((kernel_value,kernel_value),np.uint8)
        opened_img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
        data_opened.append(opened_img)
    # np.save('data_opened', data_opened)
    data_opened = np.array(data_opened)
    print('Opening kernel value set to: '+ str(open_kernel))
    print('Opened image output data size: '+ str(data_opened.shape))    
    return data_opened

# Skew Correction
def skew_Correct(input_data, dimensions):
    data_skew = []
    for i in range(len(input_data[1])):
        skew_img = input_data[:,i] 
        m = cv2.moments(skew_img)
        if abs(m['mu02']) < 1e-2:
            return skew_img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*dimensions*skew], [0, 1, 0]])
        skew_img = cv2.warpAffine(skew_img, M, (dimensions, dimensions), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)    
        skew_img = np.reshape(skew_img, dimensions**2)
        data_skew.append(skew_img)
    data_skew = np.array(data_skew)
    # print('Skew image output size set to: '+ str(skew_img.shape))
    print('Skew image output data size: '+ str(data_skew.shape))
    return data_skew

def preprocess_Data(input_data, dimensions, thin_kernel, outline_kernel, open_kernel):

    # Resized = resized(input_data, dimensions)
    Thresholding = threshold(input_data)
    # Thinned = thinned(Thresholding, thin_kernel)
    Outlined = outline(Thresholding, outline_kernel)
    # Opened = opened(Outlined, open_kernel)
    Resized = resized(Outlined, dimensions)
    # SkewCorrect = skew_Correct(Resized, dimensions)    
    # Thresholding2 = threshold(Resized)
    # # Thinned2 = thinned(Thresholding2, thin_kernel)
    # Outlined2 = outline(Thresholding2, outline_kernel)
    # Opened2 = opened(Outlined2, open_kernel)    
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

# Model 3: SVM Model with RBF Kernel

def SVM_RBF(X_train,X_test,y_train,y_test):
    ## Seems like we can improve performance including higher values for c greater than 1500
    svc = svm.SVC()
    parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1500, 2000, 2500]} 
    clf = GridSearchCV(svc, parameters, scoring='accuracy')
    clf.fit(X_train, y_train)
    # print("Best parameters set found on development set:")
    # print()
    print(clf.best_params_)
    # print("Mean cross-validated score of the best_estimator:")
    # print()
    print(clf.best_score_)
    #Accuracy for different settings on development set
    # print("Grid scores on development set:")
    # print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        print()
    # print("Detailed classification report:")
    # print()
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # print()
    y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))
    # print()
    ## Overall accuracy
    print('SVMoverall accuracy',accuracy_score(y_true, y_pred))
    SVM_RBF_Acc = accuracy_score(y_true, y_pred)
    #Confusion Matrix, created with nice visualization:)
    # disp = plot_confusion_matrix(clf, X_test, y_test,
    #                              cmap=plt.cm.Blues)
    # print(disp.confusion_matrix)
    plt.show()
    return SVM_RBF_Acc
    
def ROC_curve(y_train, y_test, X_train, X_test, ):
    y_binarized_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_binarized_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_classes = y_binarized_train.shape[1]
    # each class against the others
    classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', C=2000, gamma=0.0001, probability=True,
                                 random_state=0))
    y_score = classifier.fit(X_train, y_binarized_train).decision_function(X_test)
    y_score
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
# """
# Input pre-processing parameters here
# """
# Training_Data = data_train
# dimensions = 35
# thin_kernel = 5
# outline_kernel = 10
# open_kernel = 3

# Preprocessed_data = preprocess_Data(Training_Data, dimensions, thin_kernel, outline_kernel, open_kernel)
# trainingData = Preprocessed_data
# trainingData = trainingData.reshape((trainingData.shape[2]*trainingData.shape[1]), trainingData.shape[0])
# trainingData = trainingData.T

# # trainingData=Preprocessed_data.T  # correcting the data
# # for r in Preprocessed_data:
# #     row = []
# #     for c in r:
# #         row.append(c[0])
# #     trainingData = np.append(trainingData, row)
# # trainingData = np.array(trainingData)
# print('pre-processed data shape', trainingData.shape)
# print(labels.shape)

# # Split the Data
# X_train,X_test,y_train,y_test = splitData(trainingData, labels)

# # Run SVM RBF
# SVM_RBF_Acc = SVM_RBF(X_train,X_test,y_train,y_test)
# print('accuracy =', SVM_RBF_Acc)

# Build ROV Curves
#ROC_curve(y_train, y_test, best_k)
K_range = range(35, 65, 5)

all_SVM_Scores = np.zeros(1)
resize_SVM_Scores = np.zeros(1)
# resize_SVM_Scores2 = np.zeros(len(Resize_range))
for i in K_range:
    Training_Data = data_train
    outline_kernel = i+2
    dimensions = 35
    thin_kernel = i+2
    open_kernel = i+2

    Preprocessed_data = preprocess_Data(Training_Data, dimensions, thin_kernel, outline_kernel, open_kernel)
    trainingData = Preprocessed_data
    # trainingData = trainingData.reshape((trainingData.shape[2]*trainingData.shape[1]), trainingData.shape[0])
    # trainingData = trainingData.T

    print('pre-processed data shape', trainingData.shape)
    print(labels.shape)
    # Split the Data
    X_train,X_test,y_train,y_test = splitData(trainingData, labels)
    # Run k-NN
    SVM_RBF_Acc = SVM_RBF(X_train,X_test,y_train,y_test)
    # Build ROV Curves
    #ROC_curve(y_train, y_test, best_k)
    resize_SVM_Scores = np.vstack((resize_SVM_Scores, SVM_RBF_Acc))
all_SVM_Scores = resize_SVM_Scores[1:,:]


TopScore = np.amax(all_SVM_Scores)
TopScore_loc = np.where(all_SVM_Scores == (np.amax(all_SVM_Scores)))

print('Scores: Kernel Ranges X Resize Ranges', all_SVM_Scores.shape)
print('Highest Accuracy', TopScore)
print('Highest Accuracy Location', TopScore_loc)





