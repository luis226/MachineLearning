# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:57:04 2019

@author: Luis Galaviz
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def gaussian(X, mean, var):
    return (1 / (var * np.sqrt(2*np.pi))) * np.exp(- np.square(X - mean) / (2 * var * var))

mean = 0
var = 2

X = np.arange(-10, 10, 0.1)
y = gaussian(X, 0, 2)

plt.plot(X, y, color = 'green')

v_lines = [mean, mean - var, mean - 2*var, mean - 3*var,
           mean + var, mean + 2*var, mean + 3*var,]


for line in v_lines:
    plt.axvline(x= line).set_linestyle('--')
    
y_2 = stats.norm.pdf(X, mean, var)

plt.plot(X, y_2, color = 'red')
plt.legend(['mean', 'mean - var', 'mean - 2var', 'mean - 3var',
           'mean + var', 'mean + 2var', 'mean + 3var'])