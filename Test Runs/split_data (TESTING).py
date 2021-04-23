# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:05:49 2021

@author: carte
"""
import numpy as np 
from sklearn.model_selection import train_test_split


# splits data for validation purposes

data_train = np.load('data_train.npy')
print('raw data shape', data_train.shape)
labels = np.load('labels_train.npy')
print(labels.shape)

TRAINX_train, TESTX_test, TRAINy_train, TESTy_test = train_test_split(data_train, labels, test_size=0.7, random_state=42)

# 70% of data_train for new train
np.save('TRAIN_data_train', TRAINX_train)

# 70% of labels_train for new train
np.save('TRAIN_labels_train', TRAINy_train)

# 30% of data_train for new train
np.save('TEST_data_test', TESTX_test)

# 30% of labels_train for new train
np.save('TEST_labels_test', TESTy_test)
