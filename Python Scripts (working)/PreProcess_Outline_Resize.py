# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:15:03 2021

@author: carte
"""
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import cv2


# Loading Data
data_train = np.load('data_train.npy')
labels_train = np.load('labels_train.npy')

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

    for i in range(1):
        rnd_sample = npr.permutation(np.where(labels_train==i)[0])
        fig=plt.figure(figsize=(15,15))
        for j in range(25):
            fig.add_subplot(5,5,j+1)
            plt.imshow(256-data_outlines[rnd_sample[j],:,:].reshape((300,300)),cmap='gray')
            plt.axis('off');plt.title('Digit '+str(int(labels_train[rnd_sample[j]])),size=15)
            
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

    for i in range(1):
        rnd_sample = npr.permutation(np.where(labels_train==i)[0])
        fig=plt.figure(figsize=(15,15))
        for j in range(25):
            fig.add_subplot(5,5,j+1)
            plt.imshow(256-data_resized[:,rnd_sample[j]].reshape((dimensions,dimensions)),cmap='gray')
            plt.axis('off');plt.title('Digit v4'+str(int(labels_train[rnd_sample[j]])),size=15)
            
    return data_resized

def preprocess_Data(input_data, outline_kernel, dimensions):
    Thresholding = threshold(input_data)
    Outlined = outline(Thresholding, outline_kernel)
    Resized = resized(Outlined, dimensions)
    
    return Resized


"""
Input parameters here
"""
Training_Data = data_train
outline_kernel = 4
dimensions = 50

        
Preprocessed_data = preprocess_Data(Training_Data, outline_kernel, dimensions)

for i in range(10):
    rnd_sample = npr.permutation(np.where(labels_train==i)[0])
    fig=plt.figure(figsize=(15,15))
    for j in range(25):
        fig.add_subplot(5,5,j+1)
        plt.imshow(Preprocessed_data[:,rnd_sample[j]].reshape((dimensions,dimensions)),cmap='gray')
        plt.axis('off');plt.title('Digit '+str(int(labels_train[rnd_sample[j]])),size=15)