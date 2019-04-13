# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 06:26:02 2019

@author: Luis Galaviz
"""
import numpy as np

def linear_function_gaussian_error(X, intercept = 1, slope = 1, N = 100, 
                                   mean = 0, var = 0.5):

    N = len(X)
    
    y_real = slope * X + intercept
    y = y_real + np.random.normal(mean, var, size = N)
    
    return y_real, y

def sinoidal_gaussian_error(X, N = 100, 
                                   mean = 0, var = 0.5):

    N = len(X)
    
    y_real = np.sin(X)
    y = y_real + np.random.normal(mean, var, size = N)
    
    return y_real, y