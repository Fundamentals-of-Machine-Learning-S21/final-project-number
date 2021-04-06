# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:03:52 2021

@author: carte
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:57:28 2021

@author: carte
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import cv2


# Loading Data
data_train = np.load('data_train.npy')
labels_train = np.load('labels_train.npy')


def opened(input_data, open_kernel_value):
    data_opened = []
    for i in range(len(input_data[1])):
        img = input_data[:,i]
        kernel = np.ones((open_kernel_value,open_kernel_value),np.uint8)
        opened_img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
        data_opened.append(opened_img)
    # np.save('data_opened', data_opened)
    data_opened = np.array(data_opened)

    print('Opening kernel value set to: '+ str(open_kernel_value))

    for i in range(1):
        rnd_sample = npr.permutation(np.where(labels_train==i)[0])
        fig=plt.figure(figsize=(15,15))
        for j in range(25):
            fig.add_subplot(5,5,j+1)
            plt.imshow(256-data_opened[rnd_sample[j],:,:].reshape((300,300)),cmap='gray')
            plt.axis('off');plt.title('Digit (open) '+str(int(labels_train[rnd_sample[j]])),size=15)
            
    return data_opened
            
def closed(input_data, close_kernel_value):
    data_closed = []
    for i in range(len(input_data[1])):
        img = input_data[:,i]
        kernel = np.ones((close_kernel_value,close_kernel_value),np.uint8)
        closed_img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
        data_closed.append(closed_img)
    # np.save('data_closed', data_closed)
    data_closed3 = np.array(data_closed)

    print('Closing kernel value set to: '+ str(close_kernel_value))

    for i in range(1):
        rnd_sample = npr.permutation(np.where(labels_train==i)[0])
        fig=plt.figure(figsize=(15,15))
        for j in range(25):
            fig.add_subplot(5,5,j+1)
            plt.imshow(256-data_closed3[:,rnd_sample[j],:].reshape((300,300)),cmap='gray')
            plt.axis('off');plt.title('Digit (close)'+str(int(labels_train[rnd_sample[j]])),size=15)
            
    return data_closed
    
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

    for i in range(1):
        rnd_sample = npr.permutation(np.where(labels_train==i)[0])
        fig=plt.figure(figsize=(15,15))
        for j in range(25):
            fig.add_subplot(5,5,j+1)
            plt.imshow(256-data_resized[:,rnd_sample[j]].reshape((dimensions,dimensions)),cmap='gray')
            plt.axis('off');plt.title('Digit v4'+str(int(labels_train[rnd_sample[j]])),size=15)
            
    return data_resized
        
def preprocess_Data(input_data, open_kernel_value, close_kernel_value, dimensions):
    Opened = opened(input_data, open_kernel_value)
    Closed = closed(Opened, close_kernel_value)
    Resized = resized(Closed, dimensions)
    
    return Resized


"""
Input parameters here
"""
Training_Data = data_train
open_kernel_value = 7
close_kernel_value = 3
dimensions = 50

        
Preprocessed_data = preprocess_Data(Training_Data, open_kernel_value, close_kernel_value, dimensions)

for i in range(10):
    rnd_sample = npr.permutation(np.where(labels_train==i)[0])
    fig=plt.figure(figsize=(15,15))
    for j in range(25):
        fig.add_subplot(5,5,j+1)
        plt.imshow(256-Preprocessed_data[:,rnd_sample[j]].reshape((dimensions,dimensions)),cmap='gray')
        plt.axis('off');plt.title('Digit '+str(int(labels_train[rnd_sample[j]])),size=15)


        
        
        
        
        
        
        
        