import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# conda install -c plotly plotly
import plotly.express as px
from scipy.spatial.distance import pdist, cdist
import numpy.matlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the data

from sklearn.datasets import fetch_olivetti_faces
n_row, n_col = 3, 3
image_shape = (64, 64)
# Load faces
dataset = fetch_olivetti_faces(shuffle=True)
faces = dataset.data
labels = dataset.target
n_samples, n_features = faces.shape
print('Dataset consists of %d faces' % n_samples)


# Split the data
X = faces
y = labels
# Scaler is set here for all iterations of re-loading and re-splitting the data
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = Normalizer()
print('Scaler used on the raw data is: ', scaler)
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Training shape', X_train.shape, y_train.shape)
print('Testing shape', X_test.shape, y_test.shape)


# No PCA or LDA
knn = KNeighborsClassifier(3)
knn.fit(X_train, y_train)

print('training data knn accuracy: ','%.3f'%(knn.score(X_train, y_train)))

y_pred_test = knn.predict(X_test)
print('testing data knn accuracy: ','%.3f'%(knn.score(X_test, y_test)))


# PCA

dataset = fetch_olivetti_faces(shuffle=True)
faces = dataset.data
labels = dataset.target
n_samples, n_features = faces.shape

X = faces
y = labels
X = scaler.fit_transform(X)

# Number of Components required to preserve 90% of the data with PCA
pca = PCA(0.9)
pca.fit(X_train)
print('minimum number of principal components you need to preserve in order to explain at least 90% of the data is: ',
      pca.n_components_)

n_components = pca.n_components_
pca_faces = PCA(n_components=n_components)
X_PCA = pca_faces.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_PCA, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(3)
knn.fit(X_train, y_train)

print('training data knn accuracy: ','%.3f'%(knn.score(X_train, y_train)))
print('testing data knn accuracy: ','%.3f'%(knn.score(X_test, y_test)))



# LDA

dataset = fetch_olivetti_faces(shuffle=True)
faces = dataset.data
labels = dataset.target
n_samples, n_features = faces.shape

X = faces
y = labels
X = scaler.fit_transform(X)

# Number of Components required to preserve 90% of the data with LDA
LDA_var = []
for i in range(40):
    n_components = i
    lda_faces = LDA(n_components=n_components)
    lda_faces.fit(X_train, y_train)
    total_var = lda_faces.explained_variance_ratio_.sum() * 100
    LDA_var.append(total_var)
LDA_var = np.array(LDA_var)

#print(np.where(LDA_var>=90))
print('minimum number of principal components you need to preserve in order to explain at least 90% of the data is: ',
      np.amin(np.where(LDA_var>=90)))

n_components = np.amin(np.where(LDA_var>=90))
lda_faces = LDA(n_components=n_components)
X_LDA = lda_faces.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_LDA, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(3)
knn.fit(X_train, y_train)

print('training data knn accuracy: ','%.3f'%(knn.score(X_train, y_train)))
print('testing data knn accuracy: ','%.3f'%(knn.score(X_test, y_test)))

