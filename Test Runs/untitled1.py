# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:23:25 2021

@author: carte
"""

# cnn model with batch normalization for mnist
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

# load train and test dataset
def load_dataset():
    data_preprocesssed = np.load('data_train.npy')
    print('raw data shape', data_preprocesssed.shape)
    labels = np.load('labels_train.npy')
    print(labels.shape)
    X = data_preprocesssed
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	# one hot encode target values
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    return X_train, X_test, y_train, y_test, X, y

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(300, 300)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, datay):
    print('x: ', dataX.shape, 'y: ', datay.shape)
    # scores, histories = list(), list()
	# prepare cross validation
    # kfold = KFold(n_folds, shuffle=True, random_state=42)
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # evaluate model
    # model = define_model()
    model = KerasClassifier(build_fn=define_model, epochs=150, batch_size=10, verbose=0)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    results = cross_val_score(model, dataX, datay, cv=kfold)
    print(results.mean())
    return results

# plot diagnostic learning curves
# def summarize_diagnostics(histories):
# 	for i in range(len(histories)):
# 		# plot loss
# 		pyplot.subplot(2, 1, 1)
# 		pyplot.title('Cross Entropy Loss')
# 		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
# 		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
# 		# plot accuracy
# 		pyplot.subplot(2, 1, 2)
# 		pyplot.title('Classification Accuracy')
# 		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
# 		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
# 	pyplot.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	X_train, X_test, y_train, y_test, X, y = load_dataset()
	# prepare pixel data
# 	X_train, X_test = prep_pixels(X_train, X_test)
	# evaluate model
	scores = evaluate_model(X_train, y_train)
	# learning curves
# 	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test_harness()