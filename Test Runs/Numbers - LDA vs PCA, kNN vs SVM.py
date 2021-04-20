# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:27:02 2021

@author: carte
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# conda install -c plotly plotly
import numpy.matlib
from sklearn.neighbors import KNeighborsClassifier
import cv2
from sklearn.model_selection import GridSearchCV
from sklearn import svm


# Resizing - if one PreP before resize, input_data1. if two, input data2
def resized(input_data, dimensions):
    Data_train_resized = []
    input_data2 = np.array(input_data)
    input_data1 = input_data2.T[0]
    for i in range(len(data_train[1])):
        # img = input_data1[:,i]
        img = input_data2[:,i]
        img = img.reshape(300,300)
        dim = (dimensions,dimensions)
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        resized_img = resized_img.reshape(np.square(dimensions))
        Data_train_resized.append(resized_img)
    
    data_resized = np.array(Data_train_resized).T
    resolution2 = np.sqrt(data_resized.shape[0])
    print('Resized Training Data ='+str(data_resized.shape[1])+' samples '+'at '+ str(resolution2) + 'x' + str(resolution2)+' resolution')
    return data_resized

def resized2(input_data, dimensions, dimensions2):
    Data_train_resized = []
    input_data2 = np.array(input_data)
    input_data1 = input_data2.T[0]
    for i in range(len(data_train[1])):
        img = input_data1[:,i]
        # img = input_data2[:,i]
        img = img.reshape(dimensions,dimensions)
        dim = (dimensions2,dimensions2)
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        resized_img = resized_img.reshape(np.square(dimensions2))
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
    # print('Opened image output data size: '+ str(data_opened.shape))    
    return data_opened

# Closing
def closed(input_data, close_kernel):
    data_closed = []
    for i in range(len(input_data[1])):
        img = input_data[:,i]
        # img = img.reshape(300,300)
        kernel_value = close_kernel
        kernel = np.ones((kernel_value,kernel_value),np.uint8)
        closed_img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
        data_closed.append(closed_img)
    # np.save('data_closed', data_closed)
    data_closed = np.array(data_closed)
    print('Closing kernel value set to: '+ str(close_kernel))
    return data_closed

def preprocess_Data(input_data, dimensions, 
                                    thin_kernel, open_kernel, outline_kernel, dimensions2, 
                                    thin_kernel2, open_kernel2, outline_kernel2):
    # Resized = resized(input_data, dimensions)
    # Thresholding = threshold(input_data)
    # Opened = opened(input_data, open_kernel)    
    # Closed = closed(Opened, close_kernel)
    # Outlined = outline(input_data, outline_kernel)
    Thinned = thinned(input_data, thin_kernel)
    Thinned2 = thinned(Thinned, thin_kernel2)
    # Outlined = outline(input_data, outline_kernel)
    # Thresholding = threshold(Outlined)
    # SkewCorrect = skew_Correct(Thinned, dimensions)      
    Resized = resized(Thinned2, dimensions)
    Outlined = outline(Resized, outline_kernel2)    
    # Resized2 = resized2(Outlined, dimensions, dimensions2)
    # Outlined2 = outline(Resized, outline_kernel2)
    # Thinned2 = thinned(Resized2, thin_kernel2)
    # Opened2 = opened(Resized, open_kernel2) 
    # Outlined = outline(Thresholding, outline_kernel2)
    # Thresholding = threshold(Resized)
    # SkewCorrect = skew_Correct(Resized, dimensions)      
    return Outlined

# def preprocess_Data2(input_data, dimensions, thin_kernel, 
#                     dimensions2, outline_kernel2, 
#                     dimensions3, thin_kernel3, open_kernel3,
#                     dimensions4, outline_kernel4, 
#                     dimensions5, thin_kernel5, open_kernel5):
    
#     Thinned = thinned(input_data, thin_kernel)
#     Resized = resized(Thinned, dimensions)
#     Outlined2 = outline(Resized, outline_kernel2) 
#     Resized2 = resized2(Outlined2, dimensions, dimensions2)
#     # Thinned3 = thinned(Resized2, thin_kernel3)
#     Opened3 = opened(Resized2, open_kernel3)     
#     Resized3 = resized2(Opened3, dimensions2, dimensions3)
#     Outlined4 = outline(Resized3, outline_kernel4)
#     Resized4 = resized2(Outlined4, dimensions3, dimensions4)
#     Thinned5 = thinned(Resized4, thin_kernel5)
#     # Opened5 = opened(Resized4, open_kernel5)     
    
#     return Thinned5

# Import Data from Directory
data_train = np.load('data_train.npy')
print('raw data shape', data_train.shape)
labels = np.load('labels_train.npy')
print(labels.shape)


# thin_kernel = 60
# dimensions = 200
# outline_kernel2 = 40
# dimensions2 = 120
# thin_kernel3 = 24
# open_kernel3 = 24
# dimensions3 = 40
# outline_kernel4 = 8 
# dimensions4 = 20
# thin_kernel5 = 1
# open_kernel5 = 1
# dimensions5 = 15

dimensions = 20
dimensions2 = 20
thin_kernel = 60
open_kernel = 60
outline_kernel = 60
dimensions2 = 20
thin_kernel2 = 1
open_kernel2 = 1
outline_kernel2 = 15

Training_Data = data_train

# Preprocessed_data = preprocess_Data2(Training_Data, dimensions, thin_kernel, 
#                     dimensions2, outline_kernel2, 
#                     dimensions3, thin_kernel3, open_kernel3,
#                     dimensions4, outline_kernel4, 
#                     dimensions5, thin_kernel5, open_kernel5)

Preprocessed_data = preprocess_Data(Training_Data, dimensions, 
                                    thin_kernel, open_kernel, outline_kernel, dimensions2, 
                                    thin_kernel2, open_kernel2, outline_kernel2)
# Preprocessed_data = preprocess_Data(Training_Data, outline_kernel, dimensions)

# Split the data
# X = data_resized.T

Preprocessed_data = np.reshape(Preprocessed_data, (3360, -1))
print('pre-processed data shape', Preprocessed_data.shape)
print(labels.shape)

# When using x1 PPmethod, .T input data. When using x2, do NOT .T input data

# X = Preprocessed_data.T
X = Preprocessed_data
y = labels
# Scaler is set here for all iterations of re-loading and re-splitting the data
# scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler = Normalizer()
print('Scaler used on the raw data is: ', scaler)
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print('Training shape', X_train.shape, y_train.shape)
# print('Testing shape', X_test.shape, y_test.shape)


    ## Seems like we can improve performance including higher values for c greater than 1500
svc = svm.SVC()
parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [15000, 20000, 25000]} 
clf = GridSearchCV(svc, parameters, scoring='accuracy')
clf.fit(X_train, y_train)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#         print()

y_true, y_pred = y_test, clf.predict(X_test)
SVM_RBF_Acc = accuracy_score(y_true, y_pred)
print('SVM Overall accuracy','%.3f'%(SVM_RBF_Acc))



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

print('training data k-NN accuracy: ','%.3f'%(knn.score(X_train, y_train)))
print('testing data k-NN accuracy: ','%.3f'%(knn.score(X_test, y_test)))


# Model 4: Logistic Discrimination (with L2 Regularizaiton)
grid_values = {'penalty': ['l2'], 'C': [0.001,0.01,0.1,1,2.5,5]}
lr = LogisticRegression(max_iter=1500)
clf = GridSearchCV(lr, grid_values, scoring='accuracy')
clf.fit(X_train, y_train)
y_true, y_pred = y_test, clf.predict(X_test)
print('LR overall accuracy: ', '%.3f'%(accuracy_score(y_true, y_pred)))

# PCA

# Split the data
# X_PCA = Preprocessed_data.T
X_PCA = Preprocessed_data
y_PCA = labels
X_PCA = scaler.fit_transform(X_PCA)
X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = train_test_split(X_PCA, y_PCA, test_size=0.3, random_state=42)

# Number of Components required to preserve 90% of the data with PCA
pca = PCA(0.9)
pca.fit(X_train_PCA)
print('minimum number of principal components you need to preserve in order to explain at least 90% of the data is: ',
      pca.n_components_)

n_components = pca.n_components_
pca = PCA(n_components=n_components)
pca.fit_transform(X_train_PCA)

svc_PCA = svm.SVC()
parameters_PCA = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [15000, 20000, 25000]} 
clf_PCA = GridSearchCV(svc_PCA, parameters_PCA, scoring='accuracy')
clf_PCA.fit(X_train_PCA, y_train_PCA)
means_PCA = clf_PCA.cv_results_['mean_test_score']
stds_PCA = clf_PCA.cv_results_['std_test_score']
# for mean, std, params in zip(means_PCA, stds_PCA, clf_PCA.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#         print()

y_true_PCA, y_pred_PCA = y_test_PCA, clf.predict(X_test_PCA)
SVM_RBF_Acc_PCA = accuracy_score(y_true_PCA, y_pred_PCA)
print('SVM Overall accuracy, PCA','%.3f'%(SVM_RBF_Acc_PCA))

knn = KNeighborsClassifier(3)
knn.fit(pca.transform(X_train_PCA), y_train_PCA)
# print('train shapes: ', X_train_PCA.shape, y_train_PCA.shape)
# print('test shapes: ', X_test_PCA.shape, y_test_PCA.shape)
acc_knn_train = knn.score(pca.transform(X_train_PCA), y_train_PCA)
acc_knn_test = knn.score(pca.transform(X_test_PCA), y_test_PCA)

print('training data k-NN, PCA accuracy: ','%.3f'%(acc_knn_train))
print('testing data k-NN, PCA accuracy: ','%.3f'%(acc_knn_test))

# Model 4: Logistic Discrimination (with L2 Regularizaiton)
grid_values_PCA = {'penalty': ['l2'], 'C': [0.001,0.01,0.1,1,2.5,5]}
lr_PCA = LogisticRegression(max_iter=1500)
clf_PCA = GridSearchCV(lr_PCA, grid_values_PCA, scoring='accuracy')
clf_PCA.fit(X_train_PCA, y_train_PCA)
y_true_PCA, y_pred_PCA = y_test_PCA, clf_PCA.predict(X_test_PCA)
print('LR overall accuracy,  PCA: ', '%.3f'%(accuracy_score(y_true_PCA, y_pred_PCA)))


# LDA
# Split the data
X_LDA = Preprocessed_data
# X_LDA = Preprocessed_data.T
y_LDA = labels
X_LDA = scaler.fit_transform(X_LDA)

X_train_LDA, X_test_LDA, y_train_LDA, y_test_LDA = train_test_split(X_LDA, y, test_size=0.3, random_state=42)

# Number of Components required to preserve 90% of the data with LDA
LDA_var = []
for i in range(10):
    n_components = i
    lda_numbers = LDA(n_components=n_components)
    lda_numbers.fit(X_train_LDA, y_train_LDA)
    total_var = lda_numbers.explained_variance_ratio_.sum() * 100
    LDA_var.append(total_var)
LDA_var = np.array(LDA_var)

#print(np.where(LDA_var>=90))
print('minimum number of principal components you need to preserve in order to explain at least 90% of the data is: ',
      np.amin(np.where(LDA_var>=90)))

n_components = np.amin(np.where(LDA_var>=90))
# n_components = 9
lda = LDA(n_components=n_components)
lda.fit_transform(X_train_LDA, y_train_LDA)

svc_LDA = svm.SVC()
parameters_LDA = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                      'C': [15000, 20000, 25000]} 
clf_LDA = GridSearchCV(svc_LDA, parameters_LDA, scoring='accuracy')
clf_LDA.fit(X_train_LDA, y_train_LDA)
means_LDA = clf_LDA.cv_results_['mean_test_score']
stds_LDA = clf_LDA.cv_results_['std_test_score']
# for mean, std, params in zip(means_LDA, stds_LDA, clf_LDA.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#         print()

y_true_LDA, y_pred_LDA = y_test_LDA, clf.predict(X_test_LDA)
SVM_RBF_Acc_LDA = accuracy_score(y_true_LDA, y_pred_LDA)
print('SVM Overall accuracy, LDA','%.3f'%(SVM_RBF_Acc_LDA))

knn = KNeighborsClassifier(3)
knn.fit(lda.transform(X_train_LDA), y_train_LDA)
# acc_knn_train = knn.score(lda.transform(X_train_LDA), y_train_LDA)
# print('train shapes: ', X_train_LDA.shape, y_train_LDA.shape)
# print('test shapes: ', X_test_LDA.shape, y_test_LDA.shape)
acc_knn_test = knn.score(lda.transform(X_test_LDA), y_test_LDA)

print('training data k-NN, LDA accuracy: ','%.3f'%(acc_knn_train))
print('testing data k-NN, LDA accuracy: ','%.3f'%(acc_knn_test))


# Model 4: Logistic Discrimination (with L2 Regularizaiton)
grid_values_LDA = {'penalty': ['l2'], 'C': [0.001,0.01,0.1,1,2.5,5]}
lr_LDA = LogisticRegression(max_iter=1500)
clf_LDA = GridSearchCV(lr_LDA, grid_values_LDA, scoring='accuracy')
clf_LDA.fit(X_train_LDA, y_train_LDA)
y_true_LDA, y_pred_LDA = y_test_LDA, clf_LDA.predict(X_test_LDA)
print('LR overall accuracy, LDA: ', '%.3f'%(accuracy_score(y_true_LDA, y_pred_LDA)))