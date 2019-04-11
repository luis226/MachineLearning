# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 00:38:09 2019

@author: Luis Galaviz
"""
import matplotlib.pyplot as plt
import numpy as np

def KNN():
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
    
    def predict(self, X):
        
        
        pass
np.random.seed(40)
N = 10
a = 1
b = 1
mean = 0
var = 0.5
# load the data
X = np.arange(0, 11, 1)
N = len(X)

y_real = a * X + b
y = y_real + np.random.normal(mean, var, size = N)



resolution = 10
neighbors = 1
test_size = 11
X_test = np.arange(0.0, 11, 0.01)
resolution = len(X_test)

y_index = np.arange(0, resolution)
y_pred = np.ndarray(shape=(resolution, ))
for i, x_i in enumerate(X_test):
    print("Iteration ", i)
    diff = np.vstack((np.abs(X - x_i), y)).T

    
    indexes = diff[:,0].argsort()
    ordered_diff = diff[indexes]
    
    nearest_neighbors = ordered_diff[0:neighbors]
    print("X_i ", x_i, "\n KNN: \n", nearest_neighbors)
    
    pred = nearest_neighbors[:, 1].mean()
    y_pred[i] = pred
#


    
plt.scatter(X, y)
plt.plot(X_test, y_pred, color = 'green', alpha = 0.7)
#plt.legend(['f(x)', 'f(x) + gaussian_error', 'KNN k = 2'])

plt.scatter(X, y, s = 10)
plt.plot(X, y_real, color = 'red', alpha = 0.7)
plt.show()
    
arr = diff[diff[:,0].argsort()]
    
#plt.plot(X_test, y_pred)
#plt.legend(['f(x)', 'f(x) + gaussian_error', 'KNN k = 2'])
