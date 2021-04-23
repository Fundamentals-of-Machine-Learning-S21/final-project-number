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

import cv2
import numpy as np 
import numpy.matlib
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
import itertools


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from sklearn.metrics import confusion_matrix
from keras.models import load_model

# Input training data and labels here

data_train = np.load('data_train.npy')
print('raw data shape', data_train.shape)
labels = np.load('labels_train.npy')
print(labels.shape)

# Preprocessing Dimensions

dimensions = 20
thin_kernel = 60
outline_kernel = 4

# c-NN Hyperparameters

batch_size = 200
epochs = 400
cNN_NN_layer1 = 1000
cNN_NN_layer2 = 500


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
    for i in range(len(data_train[1])):
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

Training_Data = data_train

Preprocessed_data = preprocess_Data(Training_Data, dimensions, thin_kernel, outline_kernel)
Preprocessed_data = np.reshape(Preprocessed_data, (3360, -1))

print('pre-processed data shape', Preprocessed_data.shape)
print(labels.shape)
np.save('data_preprocessed', Preprocessed_data)


def load_data():
    data_preprocesssed = np.load('data_preprocessed.npy')
    labels = np.load('labels_train.npy')
    X = data_preprocesssed # (3360, 400)
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
    model.add(Dense(cNN_NN_layer1, activation='relu'))
    model.add(Dropout(0.25)) # reduce overfitting
    model.add(Dense(cNN_NN_layer2, activation='relu'))
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

def confusionMatrix(X_test, y_test):
    model = cNN_model()
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    Y_true = np.argmax(y_test,axis = 1) 
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    plot_confusion_matrix(confusion_mtx, classes = range(10)) 

def run_model():
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test, y_train, y_test = scale_data(X_train, X_test, y_train, y_test)
    datagen = ImageDataGenerator(rotation_range=15, zoom_range = 0.2, width_shift_range=0.2, height_shift_range=0.2)  
    datagen.fit(X_train)
    model = cNN_model()
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
    model = load_model('final_model.h5')
    # evaluate model on test dataset
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print('MODEL TEST','> %.3f' % (acc * 100.0))

run_model()


















