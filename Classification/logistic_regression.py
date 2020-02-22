# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:22:54 2020

@author: Luis Galaviz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from data import get_normal_clouds, get_donut

np.random.seed(40)
def cross_entropy(y, y_pred):
    # c value is used to avoid Ln(0) which is undefined
    c = 0.0000000000001
    err = y * np.log(y_pred + c) + (1 - y) * np.log(1 - y_pred + c)
    return -err.sum()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, Y, learning_rate= 0.05):
    B = np.random.randn(X.shape[1])
    learning_rate = 0.01
    learning_curve = np.ndarray(shape=(100, 2))
    for i in range(100):
        y_pred = sigmoid(np.matmul(X, B))
        
        learning_curve[i, 0] = i
        learning_curve[i, 1] = cross_entropy(Y, y_pred)

        B -= np.matmul(X.T, y_pred - Y ) * learning_rate
    
    #plt.plot(learning_curve[:,0], learning_curve[:,1])
    #plt.title('Logistic Regression Cross Entropy Error Minimization')
    #plt.xlabel('Iterations')
    #plt.ylabel('Cross Entropy Error')
    return B

X, y = get_normal_clouds()
#X, y = get_donut()

plt.scatter(X[:50, 1], X[:50, 2], c='green', marker= 's')
plt.scatter(X[50:, 1], X[50:, 2], c='red', marker= 'v')

B = np.random.uniform(-5, 5, size= 3).T
B = train(X, y)
y_pred = sigmoid(np.matmul(X, B))
cross_entropy_err = np.round(cross_entropy(y, y_pred), decimals = 4)

xx, yy = np.meshgrid(np.arange(start = -15, stop = 15, step = 0.01),
                     np.arange(start = -15, stop = 15, step = 0.01))
x_grid = np.vstack((xx.ravel(), yy.ravel())).T
x_grid_ones = np.ones(shape=(x_grid.shape[0], 1))
x_grid = np.concatenate((x_grid_ones, x_grid), axis = 1)

# Plotting inputs
plt.contourf(xx, yy, sigmoid(np.matmul(x_grid, B).reshape(xx.shape)),
             alpha = 0.2, cmap = ListedColormap(('red', 'green')))
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title(f'Binary Classification CEE: {cross_entropy_err}')
plt.legend(['Class 0', 'Class 1'])
plt.show()



