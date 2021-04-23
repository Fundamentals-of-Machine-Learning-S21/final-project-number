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
#87.6%

import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
import itertools
import time


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import layers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras import models
from sklearn.metrics import confusion_matrix       

num_classes = 10
batch_size = 300
epochs = 500
cNN_NN_layer1 = 750
cNN_NN_layer2 = 750


start = time.time()

data_preprocesssed = np.load('data_preprocessed.npy')
labels = np.load('labels_train.npy')
X = data_preprocesssed # (3360, 400)
# X = data_preprocessed_LDA # (3360, 7 -> 9)
# X = data_preprocessed_PCA # (3360, 73 -> 81)
y = labels # (3360,)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# NOTE: preprocessing was adjusted to ensure components were square numbers
#   PCA: increase from 73 to 81, LDA increase from 7 to 9

img_rows, img_cols = int(np.sqrt(len(X[1]))),int(np.sqrt(len(X[1])))

X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
## ADD BACK
# model.add(Conv2D(8, (3, 3),
#                   kernel_initializer='he_uniform', 
#                   input_shape=(img_rows,img_cols,1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))


# model.add(Conv2D(16, (5, 5),
#                   kernel_initializer='he_uniform', 
#                   input_shape=(img_rows,img_cols,1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

model.add(Conv2D(20, (5, 5),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(Conv2D(16, (3, 3),
#                   kernel_initializer='he_uniform', 
#                   input_shape=(img_rows,img_cols,1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

model.add(Conv2D(32, (5, 5),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# model.add(Conv2D(32, (3, 3),
#                   kernel_initializer='he_uniform', 
#                   input_shape=(img_rows,img_cols,1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))
## ADD BACK
# model.add(Conv2D(32, (1, 1),
#                   kernel_initializer='he_uniform', 
#                   input_shape=(img_rows,img_cols,1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))

model.add(Conv2D(64, (3, 3),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
## ADD BACK
# model.add(Conv2D(64, (1, 1),
#                   kernel_initializer='he_uniform', 
#                   input_shape=(img_rows,img_cols,1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))

model.add(Conv2D(128, (3, 3),
                  kernel_initializer='he_uniform', 
                  input_shape=(img_rows,img_cols,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))

# model.add(Conv2D(128, (1, 1),
#                   kernel_initializer='he_uniform', 
#                   input_shape=(img_rows,img_cols,1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2), strides = 1, padding = 'same', data_format='channels_last'))
# model.add(Dropout(0.25))


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

datagen = ImageDataGenerator(
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2)  # randomly shift images vertically (fraction of total height)

datagen.fit(X_train)
# annealer = LearningRateScheduler(lambda x: 0.07 * 0.9 ** x)

history = model.fit(datagen.flow(X_train, y_train,
          batch_size=batch_size),
          epochs=epochs,
          verbose=0,
          steps_per_epoch=X_train.shape[0] // batch_size,
          # callbacks=[annealer],
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)

print('NN hidden layer sizes: ', cNN_NN_layer1, cNN_NN_layer2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


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
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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

# Predict the values from the validation dataset
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 

end = time.time()
print("It took "+str(int(end - start))+" seconds to solve the system!") 

