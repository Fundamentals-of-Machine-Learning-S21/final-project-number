# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:51:11 2021

@author: carte
"""

"""
Sources used to build code:
    https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
    https://datascienceplus.com/handwritten-digit-recognition-with-cnn/#:~:text=CNN%20is%20primarily%20used%20in,them%20in%20a%20certain%20category.&text=For%20this%2C%20we%20will%20use,test%20set%20of%2010%2C000%20examples.
    https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
    https://www.mdpi.com/1424-8220/20/12/3344/pdf
    https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
"""

import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
import itertools


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import layers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras import models
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

def load_data():
    data_preprocesssed = np.load('data_preprocessed.npy')
    data_preprocessed_LDA = np.load('data_preprocessed_LDA.npy', allow_pickle=True)
    data_preprocessed_PCA = np.load('data_preprocessed_PCA.npy')
    labels = np.load('labels_train.npy')
    X = data_preprocesssed # (3360, 400)
    # X = data_preprocessed_LDA # (3360, 7 -> 9)
    # X = data_preprocessed_PCA # (3360, 73 -> 81)
    y = labels # (3360,)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
    
def scale_data(X_train, X_test, y_train, y_test):
    num_classes = 10
    img_rows, img_cols = 20, 20
    X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
    X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return X_train, X_test, y_train, y_test

def cNN_model():
    img_rows, img_cols = 20, 20
    model = Sequential()
    model.add(Conv2D(16, (5, 5),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(16, (3, 3),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (5, 5),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))

    model.add(Conv2D(64, (3, 3),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (1, 1),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))

    model.add(Conv2D(128, (3, 3),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))
    model.add(Dropout(0.25)) # reduce overfitting

    model.add(Conv2D(64, (1, 1),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (1, 1),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(20, (1, 1),
                  kernel_initializer='he_uniform', 
                  kernel_regularizer = regularizers.l2(0.01),
                  input_shape=(img_rows,img_cols,1)))
    model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))
    model.add(Flatten())
    model.add(Activation('sigmoid'))

    model.add(Flatten())
    model.add(Dense(1500, activation='relu'))
    model.add(Dropout(0.25)) # reduce overfitting
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5)) # reduce overfitting
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    
    # opt = Adam(learning_rate=0.004)
    opt = Adagrad(learning_rate=0.07)
    # opt = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=opt, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
    return model

def summary_plots(history):
    # Plot the loss and accuracy curves for training and validation 
    fig, ax = plt.subplots(2,1)
    #ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    #ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='b',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.show()

def run_model():
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test, y_train, y_test = scale_data(X_train, X_test, y_train, y_test)
    datagen = ImageDataGenerator(rotation_range=15, zoom_range = 0.2, width_shift_range=0.2, height_shift_range=0.2)  
    datagen.fit(X_train)
    model = model = cNN_model()
    history = model.fit(datagen.flow(X_train, y_train,
          batch_size=batch_size),
          epochs=epochs,
          verbose=0,
          steps_per_epoch=X_train.shape[0] // batch_size,
          # callbacks=[annealer],
          validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    summary_plots(history)
    model.save('cNN_model.h5')

    
def test_model():
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test, y_train, y_test = scale_data(X_train, X_test, y_train, y_test)
    model = model = cNN_model()
    # evaluate model on test dataset
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))
 
# entry point, run the test harness
batch_size = 200
epochs = 400

run_model()
# save_model()







