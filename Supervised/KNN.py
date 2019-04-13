# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 00:38:09 2019

@author: Luis Galaviz
"""
import matplotlib.pyplot as plt
import numpy as np

class KNN:
    def __init__(self, neighbors = 1):
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



