import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# conda install -c plotly plotly
import numpy.matlib
from sklearn.neighbors import KNeighborsClassifier
import cv2
from sklearn.model_selection import GridSearchCV


# Resizing
def resized(input_data, dimensions):
    Data_train_resized = []
    input_data = np.array(input_data)
    input_data2 = input_data.T[0]
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
    return data_resized

# Binarization
def threshold(input_data):
    data_thresh = []
    for i in range(len(input_data[1])):
        img = input_data[:,i] 
        thresh_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY,11,5)
        # ret, thresh_img = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)
        data_thresh.append(thresh_img)
    # np.save('data_thresh', data_thresh)
    data_thresh = np.array(data_thresh)
    print('Thresholding: Adaptive, binary')
    return data_thresh

# Outlines
def outline(input_data, outline_kernel):
    data_outlines = []
    for i in range(len(input_data[1])):
        img = input_data[:,i]
        kernel = np.ones((outline_kernel,outline_kernel),np.uint8)
        outlines_img = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
        data_outlines.append(outlines_img)
    np.save('data_outlines', data_outlines)
    data_outlines = np.array(data_outlines)
    
    print('Outline kernel value set to: '+ str(outline_kernel))
    # print('Outline image output data size: '+ str(data_outlines.shape)) 
    return data_outlines

# Skew Correction
def skew_Correct(input_data, dimensions):
    data_skew = []
    for i in range(len(input_data[1])):
        skew_img = input_data[:,i] 
        m = cv2.moments(skew_img)
        if abs(m['mu02']) < 1e-2:
            return skew_img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*dimensions*skew], [0, 1, 0]])
        skew_img = cv2.warpAffine(skew_img, M, (dimensions, dimensions), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)    
        skew_img = np.reshape(skew_img, dimensions**2)
        data_skew.append(skew_img)
    data_skew = np.array(data_skew)
    # print('Skew image output size set to: '+ str(skew_img.shape))
    print('Skew image output data size: '+ str(data_skew.shape))
    return data_skew

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

# Noise Removal
def opened(input_data, open_kernel):
    data_opened = []
    for i in range(len(input_data[1])):
        img = input_data[:,i]
        # img = img.reshape(300,300)
        kernel_value = open_kernel
        kernel = np.ones((kernel_value,kernel_value),np.uint8)
        opened_img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
        data_opened.append(opened_img)
    # np.save('data_opened', data_opened)
    data_opened = np.array(data_opened)
    print('Opening kernel value set to: '+ str(open_kernel))
    print('Opened image output data size: '+ str(data_opened.shape))    
    return data_opened

# def preprocess_Data(input_data, outline_kernel, dimensions):
def preprocess_Data(input_data, dimensions, outline_kernel):
    # Resized = resized(input_data, dimensions)
    Thresholding = threshold(input_data)
    # Thinned = thinned(Thresholding, thin_kernel)
    Outlined = outline(Thresholding, outline_kernel)
    # Opened = opened(Thinned, open_kernel)
    Resized = resized(Outlined, dimensions)
    # SkewCorrect = skew_Correct(Resized, dimensions)   
    # SkewCorrect = skew_Correct(Resized, dimensions)    
    # Thresholding2 = threshold(Resized)
    # # Thinned2 = thinned(Thresholding2, thin_kernel)
    # Outlined2 = outline(Thresholding2, outline_kernel)
    # Opened2 = opened(Outlined2, open_kernel)    
    return Resized 

# Import Data from Directory
data_train = np.load('data_train.npy')
print('raw data shape', data_train.shape)
labels = np.load('labels_train.npy')
print(labels.shape)

# thin_kernel = 60
# open_kernel = 60
outline_kernel = 60
dimensions = 20
Training_Data = data_train

Preprocessed_data = preprocess_Data(Training_Data, dimensions, outline_kernel)
# Preprocessed_data = preprocess_Data(Training_Data, outline_kernel, dimensions)

# print('resized data shape', data_resized.shape)
# print(labels.shape)

# Split the data
# X = data_resized.T
X = Preprocessed_data.T
y = labels
# Scaler is set here for all iterations of re-loading and re-splitting the data
# scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler = Normalizer()
print('Scaler used on the raw data is: ', scaler)
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print('Training shape', X_train.shape, y_train.shape)
# print('Testing shape', X_test.shape, y_test.shape)


# No PCA or LDA

# knn2 = KNeighborsClassifier()
#     # create a dictionary of all values we want to test for n_neighbors
# param_grid = {'n_neighbors': np.arange(3, 25)}
#     # use gridsearch to test all values for n_neighbors
# knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#     # fit model to data
# knn_gscv.fit(X_train, y_train)
#     # check top performing n_neighbors value
# best_k=knn_gscv.best_params_
# print('besk k value: ', best_k)
#     # check mean score for the top performing value of n_neighbors
# knn_gscv.best_score_
# knn_gscv.

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

print('training data knn accuracy: ','%.3f'%(knn.score(X_train, y_train)))
print('testing data knn accuracy: ','%.3f'%(knn.score(X_test, y_test)))


# # PCA

# # Import Data from Directory
# data_train = np.load('data_train.npy')
# # print('raw data shape', data_train.shape)
# labels = np.load('labels_train.npy')
# # print(labels.shape)

# Training_Data = data_train

# Preprocessed_data = preprocess_Data(Training_Data, dimensions)
# # Preprocessed_data = preprocess_Data(Training_Data, outline_kernel, dimensions)

# # print('resized data shape', data_resized.shape)
# # print(labels.shape)

# # Split the data
# # X = data_resized.T
# X_PCA = Preprocessed_data.T
# y_PCA = labels
# X_PCA = scaler.fit_transform(X_PCA)
# X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = train_test_split(X_PCA, y_PCA, test_size=0.3, random_state=42)

# # Number of Components required to preserve 90% of the data with PCA
# pca = PCA(0.9)
# pca.fit(X_train_PCA)
# print('minimum number of principal components you need to preserve in order to explain at least 90% of the data is: ',
#       pca.n_components_)

# n_components = pca.n_components_
# pca = PCA(n_components=n_components)
# pca.fit_transform(X_train_PCA)

# knn = KNeighborsClassifier(3)
# knn.fit(pca.transform(X_train_PCA), y_train_PCA)
# # print('train shapes: ', X_train_PCA.shape, y_train_PCA.shape)
# # print('test shapes: ', X_test_PCA.shape, y_test_PCA.shape)
# acc_knn_train = knn.score(pca.transform(X_train_PCA), y_train_PCA)
# acc_knn_test = knn.score(pca.transform(X_test_PCA), y_test_PCA)

# print('training data knn, PCA accuracy: ','%.3f'%(acc_knn_train))
# print('testing data knn, PCA accuracy: ','%.3f'%(acc_knn_test))



# # LDA
# # Import Data from Directory
# data_train = np.load('data_train.npy')
# # print('raw data shape', data_train.shape)
# labels = np.load('labels_train.npy')
# # print(labels.shape)

# Training_Data = data_train

# Preprocessed_data = preprocess_Data(Training_Data, dimensions)
# # Preprocessed_data = preprocess_Data(Training_Data, outline_kernel, dimensions)

# # print('resized data shape', data_resized.shape)
# # print(labels.shape)

# # Split the data
# # X = data_resized.T
# X_LDA = Preprocessed_data.T
# y_LDA = labels
# X_LDA = scaler.fit_transform(X_LDA)

# X_train_LDA, X_test_LDA, y_train_LDA, y_test_LDA = train_test_split(X_LDA, y, test_size=0.3, random_state=42)

# # Number of Components required to preserve 90% of the data with LDA
# LDA_var = []
# for i in range(10):
#     n_components = i
#     lda_numbers = LDA(n_components=n_components)
#     lda_numbers.fit(X_train_LDA, y_train_LDA)
#     total_var = lda_numbers.explained_variance_ratio_.sum() * 100
#     LDA_var.append(total_var)
# LDA_var = np.array(LDA_var)

# #print(np.where(LDA_var>=90))
# print('minimum number of principal components you need to preserve in order to explain at least 90% of the data is: ',
#       np.amin(np.where(LDA_var>=90)))

# n_components = np.amin(np.where(LDA_var>=90))
# # n_components = 9
# lda = LDA(n_components=n_components)
# lda.fit_transform(X_train_LDA, y_train_LDA)

# knn = KNeighborsClassifier(3)
# knn.fit(lda.transform(X_train_LDA), y_train_LDA)
# # acc_knn_train = knn.score(lda.transform(X_train_LDA), y_train_LDA)
# # print('train shapes: ', X_train_LDA.shape, y_train_LDA.shape)
# # print('test shapes: ', X_test_LDA.shape, y_test_LDA.shape)
# acc_knn_test = knn.score(lda.transform(X_test_LDA), y_test_LDA)

# print('training data knn, LDA accuracy: ','%.3f'%(acc_knn_train))
# print('testing data knn, LDA accuracy: ','%.3f'%(acc_knn_test))

