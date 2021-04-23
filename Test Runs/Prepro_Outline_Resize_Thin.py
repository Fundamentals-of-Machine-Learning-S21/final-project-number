# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 18:49:59 2021

@author: carte
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 16:28:11 2021

@author: Carter Kelly

Sources used to build code:
    https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
    https://datascienceplus.com/handwritten-digit-recognition-with-cnn/#:~:text=CNN%20is%20primarily%20used%20in,them%20in%20a%20certain%20category.&text=For%20this%2C%20we%20will%20use,test%20set%20of%2010%2C000%20examples.
    https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
    https://www.mdpi.com/1424-8220/20/12/3344/pdf
    https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
"""
print('outline-resize-thin')
import cv2
import numpy as np 
import numpy.matlib
from sklearn.model_selection import train_test_split

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
print('thin-resize-outline')
# Input training data and labels here

data_train = np.load('data_train.npy')
print('raw data shape', data_train.shape)
labels = np.load('labels_train.npy')
print(labels.shape)

# Preprocessing Dimensions

thin_kernel = 4        # size of the kernel used by the cv2.erode function
dimensions = 20         # new number of pixels (height, width) that the input data will be resized to
outline_kernel = 60      # the size of the kernel used for the application of a morphological gradient

# c-NN Hyperparameters

num_classes = 10        # Final 10 neurons in CNN, each represents a label 0-9
img_rows, img_cols = dimensions, dimensions
batch_size = 300        # number of samples processed before the model is updated
epochs = 750            # number of complete passes through the training datase
cNN_NN_layer1 = 750     # number of neurons in first hidden layer in neural network
cNN_NN_layer2 = 750     # number of neurons in second hidden layer in neural network


start = time.time()
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
    return data_eroded

# Resizing - if one PreP before resize, input_data1. if two, input data2
def resized(input_data, dimensions):
    Data_train_resized = []
    input_data2 = np.array(input_data)
    input_data1 = input_data2.T[0]
    for i in range(len(Training_Data[1])):
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

def preprocess_Data(input_data, dimensions, thin_kernel, outline_kernel):
    Outlined = outline(input_data, outline_kernel)
    # Thinned = thinned(input_data, thin_kernel)     
    Resized = resized(Outlined, dimensions)
    Thinned = thinned(Resized, thin_kernel)
    # Outlined = outline(Resized, outline_kernel)       
    return Thinned

Training_Data = data_train

Preprocessed_data = preprocess_Data(Training_Data, dimensions, thin_kernel, outline_kernel)
Preprocessed_data = np.reshape(Preprocessed_data, (3360, -1))

print('pre-processed data shape', Preprocessed_data.shape)
print(labels.shape)
np.save('data_preprocessed', Preprocessed_data)