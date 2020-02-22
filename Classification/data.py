# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:35:14 2020

@author: Luis Galaviz
"""

import numpy as np

def get_normal_clouds():
    X = np.ndarray(shape=(100, 2))
    ones = np.ones(shape=(100, 1))
    X = np.concatenate((ones, X), axis = 1)
    
    y = np.ndarray(shape=(100,))
    
    cov = np.array([[1, 0], [0, 1]])
    X[:50, 1:] = np.random.multivariate_normal((2, 2), cov = cov, size = 50)
    X[50:, 1:] = np.random.multivariate_normal((5, 5), cov = cov, size = 50)
    
    y[:50] = np.zeros(shape=(50,)) 
    y[50:] = np.ones(shape=(50,)) 
    
    return X, y

def get_donut():
    N = 100
    R_inner = 5
    R_outer = 10
    
    R1 = np.random.randn(N//2) + R_inner
    theta = 2*np.pi*np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
    
    R2 = np.random.randn(N//2) + R_outer
    theta = 2*np.pi*np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T
    
    X = np.concatenate([ X_inner, X_outer ])
    ones = np.ones(shape=(100, 1))
    X = np.concatenate((ones, X), axis = 1)
    
    Y = np.array([0]*(N//2) + [1]*(N//2)) 
    return X, Y