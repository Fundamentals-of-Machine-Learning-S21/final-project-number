# -*- coding: utf-8 -*-
"""
@author: Carter Kelly

EEL5840 Fund. Machine Learning - Final Project
Group: Number$

This is the TRAINING code for number classification. For more information, please read our README at 
https://github.com/Fundamentals-of-Machine-Learning-S21/final-project-number/blob/main/README.md

"""

import cv2
import numpy as np 
import numpy.matlib
import time

import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')


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

data_train = np.load('data_train.npy')
print('raw data shape', data_train.shape)
labels = np.load('labels_train.npy')
print(labels.shape)

# Preprocessing Dimensions

thin_kernel = 60
dimensions = 20
outline_kernel = 4

# c-NN Hyperparameters

num_classes = 10
img_rows, img_cols = dimensions, dimensions
batch_size = 300
epochs = 750
cNN_NN_layer1 = 750
cNN_NN_layer2 = 750





# Thinning
def thinned(input_data, thin_kernel):
    data_eroded = []
    for i in range(len(input_data[1])):
        img = input_data[:,i]
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
    data_outlines = np.array(data_outlines)
    
    print('Outline kernel value set to: '+ str(outline_kernel))
    return data_outlines

def preprocess_Data(input_data, dimensions, thin_kernel, outline_kernel):
    Thinned = thinned(input_data, thin_kernel)     
    Resized = resized(Thinned, dimensions)
    Outlined = outline(Resized, outline_kernel)       
    return Outlined

# assign variable to input data
Training_Data = data_train

# preprocess the data
Preprocessed_data = preprocess_Data(Training_Data, dimensions, thin_kernel, outline_kernel)
# Preprocessed_data = np.reshape(Preprocessed_data, (3360, -1))

print('pre-processed data shape', Preprocessed_data.shape)
print(labels.shape)
np.save('data_preprocessed', Preprocessed_data)

# Train the CNN

# pull in training data
def load_trainingData():
    data_preprocesssed = np.load('data_preprocessed.npy')
    X_training = data_preprocesssed # (3360, 400)
    y_training = labels # (3360,)
    return X_training, y_training

# scale the input data to match requirements of CNN (20x20 square)
def scale_trainingData(X_training, y_training):
    X_training = X_training.reshape(X_training.shape[0],img_rows,img_cols,1)
    print('x_train shape:', X_training.shape)
    print(X_training.shape[0], 'train samples')
    y_training = keras.utils.to_categorical(y_training, num_classes)
    return X_training, y_training

# Define CNN Model
def cNN_model():
    model = Sequential()
    # Layer 1
    model.add(Conv2D(20, (5, 5),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 2
    model.add(Conv2D(32, (5, 5),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))
    # Layer 3
    model.add(Conv2D(64, (3, 3),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 4
    model.add(Conv2D(128, (3, 3),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))
    # Layer 5
    model.add(Conv2D(64, (1, 1),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 6
    model.add(Conv2D(32, (1, 1),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Layer 7
    model.add(Conv2D(20, (1, 1),
                  kernel_initializer='he_uniform', 
                  kernel_regularizer = regularizers.l2(0.01),
                  input_shape=(img_rows,img_cols,1)))
    model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    # Run through MLP (x2 layers)
    model.add(Flatten())
    model.add(Dense(cNN_NN_layer1, activation='relu'))
    model.add(Dropout(0.25)) # reduce overfitting
    model.add(Dense(cNN_NN_layer2, activation='relu'))
    model.add(Dropout(0.5)) # reduce overfitting
    model.add(BatchNormalization())
    # Flatten down in to x10 dimensions (digits 0-9)
    model.add(Dense(10, activation='softmax'))

    # Optimize the model (best option: Adagrad with lr of 0.07)
    # opt = Adam(learning_rate=0.004)
    opt = Adagrad(learning_rate=0.07)
    # opt = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    # Compile the model
    model.compile(optimizer=opt, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
    return model


def train_model():
    X_training, y_training = load_trainingData()
    X_training, y_training = scale_trainingData(X_training, y_training)
    datagen = ImageDataGenerator(rotation_range=15, zoom_range = 0.2, width_shift_range=0.2, height_shift_range=0.2)  
    datagen.fit(X_training)
    model = cNN_model()
    model.fit(datagen.flow(X_training, y_training,
          batch_size=batch_size),
          epochs=epochs,
          verbose=0,
          steps_per_epoch=X_training.shape[0] // batch_size)
    model.save('Number$_CNN_Model.h5') 
    

train_model()

end = time.time()
print("It took "+str(int(end - start))+" seconds to train the program!") 
















