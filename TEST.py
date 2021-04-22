# -*- coding: utf-8 -*-
"""
@author: Carter Kelly

EEL5840 Fund. Machine Learning - Final Project
Group: Number$

This is the TESTING code for number classification. For more information, please read our README at 
https://github.com/Fundamentals-of-Machine-Learning-S21/final-project-number/blob/main/README.md

"""

import cv2
import numpy as np 
import numpy.matlib

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
import itertools
import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from sklearn.metrics import confusion_matrix
from keras.models import load_model
# start timer
start = time.time()

# Input training data and labels here

data_test = np.load('data_test.npy')   # testing data input (should be remaining 30-20% of the total data)
print('raw data shape', data_test.shape)
labels = np.load('labels_test.npy')    # input for labels associated with testing data
print(labels.shape)

# Preprocessing Dimensions

thin_kernel = 60        # size of the kernel used by the cv2.erode function
dimensions = 20         # new number of pixels (height, width) that the input data will be resized to
outline_kernel = 4      # the size of the kernel used for the application of a morphological gradient

# c-NN Hyperparameters

num_classes = 10        # Final 10 neurons in CNN, each represents a label 0-9
img_rows, img_cols = dimensions, dimensions
batch_size = 300        # number of samples processed before the model is updated
epochs = 750            # number of complete passes through the training datase
cNN_NN_layer1 = 750     # number of neurons in first hidden layer in neural network
cNN_NN_layer2 = 750     # number of neurons in second hidden layer in neural network



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
    for i in range(len(Testing_Data[1])):
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

# assign variable to input data
Testing_Data = data_test

Preprocessed_data = preprocess_Data(Testing_Data, dimensions, thin_kernel, outline_kernel)
# Preprocessed_data = np.reshape(Preprocessed_data, (3360, -1))

print('pre-processed data shape', Preprocessed_data.shape)
print(labels.shape)
np.save('test_data_preprocessed', Preprocessed_data)

# Load and scale testing data (DOES NOT TRAIN TESTING DATA)

# pull in testing data
def load_testingData():
    data_preprocesssed = np.load('data_preprocessed.npy')
    X_testing = data_preprocesssed # (3360, 400)
    y_testing = labels # (3360,)
    return X_testing, y_testing

# scale the input data to match requirements of CNN (20x20 square)
def scale_testingData(X_testing, y_testing):
    X_testing = X_testing.reshape(X_testing.shape[0],img_rows,img_cols,1)
    print('x_test shape:', X_testing.shape)
    print(X_testing.shape[0], 'train samples')
    y_testing = keras.utils.to_categorical(y_testing, num_classes)
    return X_testing, y_testing

# Confusion matrix build function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Run model test with testing data    
def test_model():
    X_testing, y_testing = load_testingData()
    X_testing, y_testing = scale_testingData(X_testing, y_testing)
    model = load_model('Number$_CNN_Model.h5')
    score = model.evaluate(X_testing, y_testing, verbose=0)
    print('Test loss:', '%.3f' % (score[0]))
    print('Test accuracy:', '%.3f' % (score[1]*100),'%')
    Y_pred = model.predict(X_testing)
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    Y_true = np.argmax(y_testing,axis = 1) 
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    plot_confusion_matrix(confusion_mtx, classes = range(10)) 
    labels_predicted = Y_pred_classes
    np.save('labels_predicted', labels_predicted)
    return labels_predicted, score
    
labels_predicted, score = test_model()


end = time.time()
print("It took "+str(int(end - start))+" seconds to run the test data!") 












