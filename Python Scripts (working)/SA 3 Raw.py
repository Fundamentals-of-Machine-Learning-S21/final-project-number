# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')

arr1 = np.load('preprocessed_data_thresh_close_resize.npy')
print(arr1.shape)
arr=arr1.T  # correcting the data
# for r in arr1:
#     row = []
#     for c in r:
#         row.append(c[0])
#     arr.append(row)
# arr = np.array(arr)
# print(arr.shape)

arr2 = np.load('labels_train.npy')
arr2.shape

# Pre-processing the data, this includes data normalization

print('Min: %.3f, Max: %.3f' % (arr.min(), arr.max()))
print('Mean: %.3f, Standarddeviation: %.3f' % (np.mean(arr), np.std(arr)))

# Min-max scaling the data
scaler = MinMaxScaler()
normalized_arr2 = scaler.fit_transform(arr)
print(normalized_arr2)
#Data after min-max scaling of the data
print('Min: %.3f, Max: %.3f' % (normalized_arr2.min(), normalized_arr2.max()))
print('Mean: %.3f, Standarddeviation: %.3f' % (np.mean(normalized_arr2), np.std(normalized_arr2)))

# Normalizing the data using Normalizer ()
scaler = Normalizer()
normalized_arr_n = scaler.fit_transform(arr)
print(normalized_arr_n)
#Data after sample-wise L2 normalizing
print('Min: %.3f, Max: %.3f' % (normalized_arr_n.min(), normalized_arr_n.max()))
print('Mean: %.3f, Standarddeviation: %.3f' % (np.mean(normalized_arr_n), np.std(normalized_arr_n)))

# Normalizing the data using StandardScaler
scaler = StandardScaler()
normalized_arr_ss = scaler.fit_transform(arr)
print(normalized_arr_ss)
#Data after sample-wise L2 normalizing
print('Min: %.3f, Max: %.3f' % (normalized_arr_ss.min(), normalized_arr_ss.max()))
print('Mean: %.3f, Standarddeviation: %.3f' % (np.mean(normalized_arr_ss), np.std(normalized_arr_ss)))

normalized_arr2.shape,normalized_arr_n.shape,normalized_arr_ss.shape  # checking the shape

# Split the data

##train-test split as 70% train and 30% test using normalized data with Min-max scaling the data
X_train, X_test, y_train, y_test = train_test_split(normalized_arr2, arr2, test_size=0.30, random_state=42) 

print('X_train: ',X_train.shape, 'X_test: ',X_test.shape, 'y_train: ',y_train.shape, 'y_test: ',y_test.shape)

# Model 1: k-NN

from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(3, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X_train, y_train)

#check top performing n_neighbors value
best_k=knn_gscv.best_params_
best_k

#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_

knn_gscv.best_params_

knn = KNeighborsClassifier(n_neighbors = best_k['n_neighbors'])

knn.fit(X_train,y_train)

y_pred =knn.predict(X_test)

#check accuracy of our model on the test data
knn.score(X_test, y_test)

## Overall accuracy on test data
accuracy_score(y_test, y_pred)

## Confusion matrix on test data
confusion_matrix(y_test, y_pred)

#Confusion Matrix, created with nice visualization:)

from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(knn, X_test, y_test,
                                 cmap=plt.cm.Blues)

print(disp.confusion_matrix)

plt.show()

#ROC curve

y_score_knn = knn.fit(X_train, y_train).predict_proba(X_test)

y_score_knn.shape

y_test.shape

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

y_binarized_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_binarized_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = y_binarized_train.shape[1]

y_train

y_binarized_train

# each class against the others
classifier = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = best_k['n_neighbors']))

y_score = classifier.fit(X_train, y_binarized_train).predict_proba(X_test)

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


import matplotlib.colors as mcolors
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

# Model 2: Naive Bayes Classifier

#Fit Naive Bayes Classifier on training data
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
classifier_gnb = gnb.fit(X_train, y_train)
y_pred = classifier_gnb.predict(X_test)

## Overall accuracy on test data
accuracy_score(y_test, y_pred)

#Confusion Matrix, created with nice visualization:)

from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(gnb, X_test, y_test,
                                 cmap=plt.cm.Blues)
print(disp.confusion_matrix)
plt.show()

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

import matplotlib.colors as mcolors
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

# Model 3: SVM Model with RBF Kernel

from sklearn import svm
from sklearn.model_selection import GridSearchCV

## Seems like we can improve performance including higher values for c greater than 1500
svc = svm.SVC()
parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1500, 2000, 2500]} 
clf = GridSearchCV(svc, parameters, scoring='accuracy')
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)

print("Mean cross-validated score of the best_estimator:")
print()
print(clf.best_score_)

#Accuracy for different settings on development set
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
print()

from sklearn.metrics import classification_report
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

## Overall accuracy
print('overall accuracy',accuracy_score(y_true, y_pred))

#Confusion Matrix, created with nice visualization:)

from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues)

print(disp.confusion_matrix)

plt.show()

## ROC curves for the overall classifier and for each class
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
#from scipy import interp
from sklearn.metrics import roc_auc_score

y_binarized_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_binarized_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = y_binarized_train.shape[1]

# Predict each class against the otherS
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

import matplotlib.colors as mcolors
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

# Model 4: Logistic Discrimination (with L2 Regularizaiton)

from sklearn.linear_model import LogisticRegression
grid_values = {'penalty': ['l2'], 'C': [0.001,0.01,0.1,1,2.5,5]}
lr = LogisticRegression(max_iter=1500)
clf = GridSearchCV(lr, grid_values, scoring='accuracy')
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)

print("Mean cross-validated score of the best_estimator:")
print()
print(clf.best_score_)

y_true, y_pred = y_test, clf.predict(X_test)

## Overall accuracy on test data
print('Model 4 overall accuracy: ', accuracy_score(y_true, y_pred))

#Confusion Matrix, created with nice visualization:)

from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues)

print(disp.confusion_matrix)

plt.show()

## ROC curves for the overall classifier and for each class
# Learn to predict each class against the other
classifier = OneVsRestClassifier(LogisticRegression(penalty = 'l2', C=1, max_iter=1500))
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

import matplotlib.colors as mcolors
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

# MODEL 5. Fisher's Linear Discriminant Analysis (LDA) Model.

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv)
# summarize result

np.mean(scores) 

np.std(scores)

# fit model
lda = model.fit(X_train, y_train)
# make a prediction
y_pred = model.predict(X_test)

y_pred

## Overall accuracy on test data
accuracy_score(y_test, y_pred)

#Confusion Matrix, created with nice visualization:)

disp = plot_confusion_matrix(model, X_test, y_test,
                                 cmap=plt.cm.Blues)

print(disp.confusion_matrix)

plt.show()

# each class against the others
classifier = OneVsRestClassifier(LinearDiscriminantAnalysis())
y_score = classifier.fit(X_train, y_binarized_train).predict_proba(X_test)
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

import matplotlib.colors as mcolors
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

































