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


# Resize
def resized(input_data, dimensions):
    Data_train_resized = []
    input_data = np.array(input_data)
    input_data2 = input_data.T[0]
    for i in range(len(data_train[1])):
        # img = input_data2[:,i]
        img = input_data[:,i]
        img = img.reshape(300,300)
        dim = (dimensions,dimensions)
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        resized_img = resized_img.reshape(np.square(dimensions))
        Data_train_resized.append(resized_img)

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

def preprocess_Data(input_data, dimensions, skew_limit, skew_delta,  thin_kernel, outline_kernel, open_kernel):

    Resized = resized(input_data, dimensions)
    Thresholding = threshold(Resized)
    SkewCorrect = skew_Correct(Thresholding, skew_limit, skew_delta)
    Thinned = thinned(SkewCorrect, thin_kernel)
    Outlined = outline(Thinned, outline_kernel)
    Opened = opened(Outlined, open_kernel)
    
    return Opened

# Import Data from Directory
data_train = np.load('data_train.npy')
print('raw data shape', data_train.shape)
labels = np.load('labels_train.npy')
print(labels.shape)

"""
Input pre-processing parameters here
"""
Training_Data = data_train
dimensions = 20
skew_limit = 15
skew_delta = 1
thin_kernel = 5
outline_kernel = 4
open_kernel_value = 3

Preprocessed_data = preprocess_Data(Training_Data, outline_kernel, dimensions)




