#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 17:51:24 2018

@author: deadpool
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from pandas.api.types import CategoricalDtype

matplotlib.style.use('ggplot') # Look Pretty

def plotDecisionBoundary(model, X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    padding = 0.6
    resolution = 0.0025
    colors = ['royalblue','forestgreen','ghostwhite']

    # Calculate the boundaries
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    y_min -= y_range * padding
    x_max += x_range * padding
    y_max += y_range * padding

    # Create a 2D Grid Matrix. The values stored in the matrix
    # are the predictions of the class at at said location
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

    # What class does the classifier say?
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour map
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)

    # Plot the test original points as well...
    for label in range(len(np.unique(y))):
        indices = np.where(y == label)
        plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], label=str(label), alpha=0.8)

#    for i, col in enumerate(y.columns):
#        indices = np.where(y.col == 1)
#        plt.scatter(X[indices, 0], X[indices, 1], c=colors[i], label=str(i), alpha=0.8)
#    
    p = model.get_params()
    plt.axis('tight')
    plt.title('K = ' + str(p['n_neighbors']))
    


X = pd.read_csv('Datasets/wheat.data', index_col='id')
y = X[['wheat_type']]
y = pd.get_dummies(y)
X.drop(columns='wheat_type', inplace=True)
X.fillna(X.mean(), inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

#categ = CategoricalDtype(y.wheat_type.unique(), ordered=True)
#y_test = y_test.wheat_type.astype(categ).cat.codes
#y_train = y_train.wheat_type.astype(categ).cat.codes

norm = Normalizer()
norm.fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)

#pca = PCA(n_components=2, random_state=1)
#pca.fit(X_train)
#X_test = pca.transform(X_test)
#X_train = pca.transform(X_train)

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

plotDecisionBoundary(knn, X_train, y_train)

def show_score(k=9):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(knn.score(X_test, y_test))

























    
