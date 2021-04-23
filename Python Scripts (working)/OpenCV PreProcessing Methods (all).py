# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 21:05:13 2021

@author: carte
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import cv2

plt.style.use('bmh')

# Loading Data
data_train = np.load('data_train.npy')
labels_train = np.load('labels_train.npy')
data_resized = np.load('data_resized (28x28).npy')
# data_opened = np.load('data_opened.npy')
# data_closed = np.load('data_closed.npy')

"""
To add:
    Thresholding
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
        adapt_thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\ cv2.THRESH_BINARY,11,2)
        
    and to clean it up a tlittle: 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
"""

print(data_train.shape, labels_train.shape)

# data_outlines2 = []
# for i in range(len(data_resized[1])):
#     img = data_resized[:,i]
#     kernel_value = 2
#     kernel = np.ones((kernel_value,kernel_value),np.uint8)
#     outlines_img = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
#     data_outlines2.append(outlines_img)
# np.save('PREPROCESSING_OPTION_ONE_data_resized_outlines', data_outlines2)
# data_outlines2 = np.array(data_outlines2)

# print('Outline kernel value set to: '+ str(kernel_value))

data_opened_2 = []
for i in range(len(data_resized[1])):
    img = data_resized[:,i]
    # img = img.data_resized(300,300)
    kernel_value = 7
    kernel = np.ones((kernel_value,kernel_value),np.uint8)
    opened_img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
    data_opened_2.append(opened_img)
np.save('data_opened', data_opened_2)
data_opened_2 = np.array(data_opened_2)

print('Opening kernel value set to: '+ str(kernel_value))

data_closed2 = []
for i in range(len(data_opened_2[1])):
    img = data_opened_2[:,i]
    # img = img.reshape(300,300)
    kernel_value = 3
    kernel = np.ones((kernel_value,kernel_value),np.uint8)
    closed_img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    data_closed2.append(closed_img)
np.save('PREPROCESSING_OPTION_TWO_data_resized_opened_closed', data_closed2)
data_closed2 = np.array(data_closed2)

print('Closing kernel value set to: '+ str(kernel_value))





# Erosion of raw data

# data_eroded = []
# for i in range(len(data_train[1])):
#     img = data_train[:,i]
#     # img = img.reshape(300,300)
#     kernel_value = 5
#     kernel = np.ones((kernel_value,kernel_value),np.uint8)
#     eroded_img = cv2.erode(img,kernel,iterations = 3)
#     data_eroded.append(eroded_img)
# np.save('data_dilated', data_eroded)
# data_eroded = np.array(data_eroded)

# print('Erosion kernel value set to: '+ str(kernel_value))

# for i in range(1):
#     rnd_sample = npr.permutation(np.where(labels_train==i)[0])
#     fig=plt.figure(figsize=(15,15))
#     for j in range(25):
#         fig.add_subplot(5,5,j+1)
#         plt.imshow(256-data_eroded[rnd_sample[j],:,:].reshape((300,300)),cmap='gray')
#         plt.axis('off');plt.title('Digit '+str(int(labels_train[rnd_sample[j]])),size=15)

# Dilation of raw data

# data_dilated = []
# for i in range(len(data_train[1])):
#     img = data_train[:,i]
#     # img = img.reshape(300,300)
#     kernel_value = 5
#     kernel = np.ones((kernel_value,kernel_value),np.uint8)
#     dilated_img = cv2.dilate(img,kernel,iterations = 1)
#     data_dilated.append(dilated_img)
# np.save('data_dilated', data_dilated)
# data_dilated = np.array(data_dilated)

# print('Dilation kernel value set to: '+ str(kernel_value))


# for i in range(1):
#     rnd_sample = npr.permutation(np.where(labels_train==i)[0])
#     fig=plt.figure(figsize=(15,15))
#     for j in range(25):
#         fig.add_subplot(5,5,j+1)
#         plt.imshow(256-data_dilated[rnd_sample[j],:,:].reshape((300,300)),cmap='gray')
#         plt.axis('off');plt.title('Digit '+str(int(labels_train[rnd_sample[j]])),size=15)
        
# Opening of raw data

# data_opened = []
# for i in range(len(data_train[1])):
#     img = data_train[:,i]
#     # img = img.reshape(300,300)
#     kernel_value = 7
#     kernel = np.ones((kernel_value,kernel_value),np.uint8)
#     opened_img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
#     data_opened.append(opened_img)
# np.save('data_opened', data_opened)
# data_opened = np.array(data_opened)

# print('Opening kernel value set to: '+ str(kernel_value))

# Closing of raw data

# data_closed = []
# for i in range(len(data_opened[1])):
#     img = data_opened[:,i]
#     # img = img.reshape(300,300)
#     kernel_value = 3
#     kernel = np.ones((kernel_value,kernel_value),np.uint8)
#     closed_img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
#     data_closed.append(closed_img)
# np.save('data_closed', data_closed)
# data_closed = np.array(data_closed)

# print('Closing kernel value set to: '+ str(kernel_value))


# for i in range(1):
#     rnd_sample = npr.permutation(np.where(labels_train==i)[0])
#     fig=plt.figure(figsize=(15,15))
#     for j in range(25):
#         fig.add_subplot(5,5,j+1)
#         plt.imshow(256-data_opened[rnd_sample[j],:,:].reshape((300,300)),cmap='gray')
#         plt.axis('off');plt.title('Digit '+str(int(labels_train[rnd_sample[j]])),size=15)

# Outline creation using raw data

# data_outlines = []
# for i in range(len(data_train[1])):
#     img = data_train[:,i]
#     kernel_value = 4
#     kernel = np.ones((kernel_value,kernel_value),np.uint8)
#     outlines_img = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
#     data_outlines.append(outlines_img)
# np.save('data_outlines', data_outlines)
# data_outlines = np.array(data_outlines)

# print('Outline kernel value set to: '+ str(kernel_value))

# for i in range(1):
#     rnd_sample = npr.permutation(np.where(labels_train==i)[0])
#     fig=plt.figure(figsize=(15,15))
#     for j in range(25):
#         fig.add_subplot(5,5,j+1)
#         plt.imshow(256-data_outlines[rnd_sample[j],:,:].reshape((300,300)),cmap='gray')
#         plt.axis('off');plt.title('Digit '+str(int(labels_train[rnd_sample[j]])),size=15)

# # Erosion of raw data

# data_eroded = []
# for i in range(len(data_outlines[1])):
#     img = data_outlines[:,i]
#     # img = img.reshape(300,300)
#     kernel_value = 3
#     kernel = np.ones((kernel_value,kernel_value),np.uint8)
#     eroded_img = cv2.erode(img,kernel,iterations = 3)
#     data_eroded.append(eroded_img)
# np.save('data_dilated', data_eroded)
# data_eroded = np.array(data_eroded)

# print('Erosion kenrel value set to: '+ str(kernel_value))

# for i in range(1):
#     rnd_sample = npr.permutation(np.where(labels_train==i)[0])
#     fig=plt.figure(figsize=(15,15))
#     for j in range(25):
#         fig.add_subplot(5,5,j+1)
#         plt.imshow(256-data_eroded[:,rnd_sample[j],:].reshape((300,300)),cmap='gray')
#         plt.axis('off');plt.title('Digit '+str(int(labels_train[rnd_sample[j]])),size=15)

# # Dilation of raw data

# data_dilated = []
# for i in range(len(data_outlines[1])):
#     img = data_outlines[:,i]
#     # img = img.reshape(300,300)
#     kernel_value = 5
#     kernel = np.ones((kernel_value,kernel_value),np.uint8)
#     dilated_img = cv2.dilate(img,kernel,iterations = 1)
#     data_dilated.append(dilated_img)
# np.save('data_dilated', data_dilated)
# data_dilated = np.array(data_dilated)

# print('Dilation kenrel value set to: '+ str(kernel_value))


# for i in range(1):
#     rnd_sample = npr.permutation(np.where(labels_train==i)[0])
#     fig=plt.figure(figsize=(15,15))
#     for j in range(25):
#         fig.add_subplot(5,5,j+1)
#         plt.imshow(256-data_dilated[:,rnd_sample[j],:].reshape((300,300)),cmap='gray')
#         plt.axis('off');plt.title('Digit '+str(int(labels_train[rnd_sample[j]])),size=15)

# # Closing of raw data

# data_closed = []
# for i in range(len(data_outlines[1])):
#     img = data_outlines[:,i]
#     # img = img.reshape(300,300)
#     kernel_value = 3
#     kernel = np.ones((kernel_value,kernel_value),np.uint8)
#     closed_img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
#     data_closed.append(closed_img)
# np.save('data_closed', data_closed)
# data_closed = np.array(data_closed)

# print('Closing kenrel value set to: '+ str(kernel_value))


# for i in range(1):
#     rnd_sample = npr.permutation(np.where(labels_train==i)[0])
#     fig=plt.figure(figsize=(15,15))
#     for j in range(25):
#         fig.add_subplot(5,5,j+1)
#         plt.imshow(256-data_opened[:,rnd_sample[j],:].reshape((300,300)),cmap='gray')
#         plt.axis('off');plt.title('Digit '+str(int(labels_train[rnd_sample[j]])),size=15)

# Outline creation using raw data

# # Opening of raw data

# data_opened = []
# for i in range(len(data_outlines[1])):
#     img = data_outlines[:,i]
#     # img = img.reshape(300,300)
#     kernel_value = 7
#     kernel = np.ones((kernel_value,kernel_value),np.uint8)
#     opened_img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
#     data_opened.append(opened_img)
# np.save('data_opened', data_opened)
# data_opened = np.array(data_opened)

# print('Opening kenrel value set to: '+ str(kernel_value))

# for i in range(1):
#     rnd_sample = npr.permutation(np.where(labels_train==i)[0])
#     fig=plt.figure(figsize=(15,15))
#     for j in range(25):
#         fig.add_subplot(5,5,j+1)
#         plt.imshow(256-data_opened[:,rnd_sample[j],:].reshape((300,300)),cmap='gray')
#         plt.axis('off');plt.title('Digit '+str(int(labels_train[rnd_sample[j]])),size=15)