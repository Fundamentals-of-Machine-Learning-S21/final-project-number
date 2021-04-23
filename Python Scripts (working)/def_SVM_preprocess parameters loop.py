# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:27:26 2021

@author: carte
"""

from scipy import interp
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import matplotlib.colors as mcolors
import numpy.random as npr
import cv2

def threshold(input_data):
    data_thresh = []
    for i in range(len(input_data[1])):
        img = input_data[:,i] 
        thresh_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY,11,5)
        data_thresh.append(thresh_img)
    # np.save('data_thresh', data_thresh)
    data_thresh = np.array(data_thresh)

    # for i in range(1):
    #     rnd_sample = npr.permutation(np.where(labels==i)[0])
    #     fig=plt.figure(figsize=(15,15))
    #     for j in range(25):
    #         fig.add_subplot(5,5,j+1)
    #         plt.imshow(256-data_thresh[rnd_sample[j],:,:].reshape((300,300)),cmap='gray')
    #         plt.axis('off');plt.title('Digit (thresh) '+str(int(labels[rnd_sample[j]])),size=15)
    
    return data_thresh

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

    # for i in range(1):
    #     rnd_sample = npr.permutation(np.where(labels==i)[0])
    #     fig=plt.figure(figsize=(15,15))
    #     for j in range(25):
    #         fig.add_subplot(5,5,j+1)
    #         plt.imshow(256-data_outlines[rnd_sample[j],:,:].reshape((300,300)),cmap='gray')
    #         plt.axis('off');plt.title('Digit outline '+str(int(labels[rnd_sample[j]])),size=15)
            
    return data_outlines

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

def preprocess_Data(input_data, outline_kernel, dimensions):
    Thresholding = threshold(input_data)
    Outlined = outline(Thresholding, outline_kernel)
    Resized = resized(Outlined, dimensions)
    
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

"""
Input pre-processing parameters here
"""
# Training_Data = data_train
# outline_kernel = 4
# dimensions = 50

        
# Preprocessed_data = preprocess_Data(Training_Data, outline_kernel, dimensions)
# trainingData = Preprocessed_data
# print('pre-processed data shape', trainingData.shape)
# print(labels.shape)
# Split the Data
# X_train,X_test,y_train,y_test = splitData(trainingData, labels)
# Run SVM
# SVM_RBF_Acc = SVM_RBF(X_train,X_test,y_train,y_test)
# Build ROV Curves
#ROC_curve(y_train, y_test, best_k)

# KERNEL AND RESIZE PARAMETER LOOP
K_range = range(20, 85, 5)
Resize_range = 35

all_SVM_Scores = np.zeros(1)
resize_SVM_Scores = np.zeros(1)
# resize_SVM_Scores2 = np.zeros(len(Resize_range))
for i in K_range:
    Training_Data = data_train
    outline_kernel = i+2
    for j in Resize_range:
        dimensions = j
        Preprocessed_data = preprocess_Data(Training_Data, outline_kernel, dimensions)
        # Detele .T and column select if doing more than threshold 
        trainingData=Preprocessed_data  # correcting the data
        # trainingData=Preprocessed_data.T  # correcting the data
        # for r in Preprocessed_data:
            #     row = []
            #     for c in r:
                #         row.append(c[0])
                #     trainingData = np.append(trainingData, row)
                # trainingData = np.array(trainingData)
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


np.reshape(all_SVM_Scores, ((len(K_range), len(Resize_range))))
print('Scores: Kernel Ranges X Resize Ranges')
TopScore = np.amax(all_SVM_Scores)
TopScore_loc = np.where(all_SVM_Scores == (np.amax(all_SVM_Scores)))

print('Scores: Kernel Ranges X Resize Ranges', all_SVM_Scores.shape)
print('Highest Accuracy', TopScore)
print('Highest Accuracy Location', TopScore_loc)










