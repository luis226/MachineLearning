# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 00:38:09 2019

@author: Luis Galaviz
"""
import matplotlib.pyplot as plt
import numpy as np
from data import linear_function_gaussian_error, sinoidal_gaussian_error

class KNN:
    def __init__(self, neighbors):
        self.neighbors = neighbors
    
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
    
    def predict(self, X_pred):
        y_pred = np.ndarray(shape=(len(X_pred), ))
        y_index = np.arange(0, len(self.X_train))
        for i, x_i in enumerate(X_pred):
            #print("Iteration ", i)
            diff = np.vstack((np.abs(self.X_train - x_i), self.Y_train, y_index)).T
        
            
            indexes = diff[:,0].argsort()
            ordered_diff = diff[indexes]
            
            nearest_neighbors = ordered_diff[0:self.neighbors]
            #print("X_i ", x_i, "\n KNN: \n", nearest_neighbors)
            
            pred = nearest_neighbors[:, 1].mean()
            y_pred[i] = pred
        
        return y_pred
    
np.random.seed(40)
X = np.arange(0, 10, 0.2)
y_real, y = sinoidal_gaussian_error(X, N = 10, var = 0.2)



X_test = np.arange(0, 10, 0.01)

neighbors = 2
model = KNN(1)     
model.fit(X, y)
y_pred = model.predict(X_test)             
                  
#


    
plt.scatter(X, y)
plt.plot(X_test, y_pred, color = 'green', alpha = 0.7)
#plt.legend(['f(x)', 'f(x) + gaussian_error', 'KNN k = 2'])

plt.scatter(X, y, s = 10)
plt.plot(X, y_real, color = 'red', alpha = 0.7)
plt.show()
    

#plt.plot(X_test, y_pred)
#plt.legend(['f(x)', 'f(x) + gaussian_error', 'KNN k = 2'])
