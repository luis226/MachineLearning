# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 03:51:12 2019

@author: Luis Galaviz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from error_functions import *
    
np.random.seed(20)
   
# Data parameters
real_slope = 1
real_intercept = 0
N = 10
mean = 0
var = 0.5

# Generating simulated data f(x) = ax + b + N(mean, var)
X =  np.arange(0, 10)
y_real = X * real_slope + real_intercept 
y = y_real + np.random.normal(mean, var, size = N)

# Reshape vectors to ensure operations consistency
y_real = y_real.reshape((N, 1))
y = y.reshape((N, 1))

# Adds a ones row to be able to write this on matrix notation
X = np.vstack((np.ones(shape = (N,)), X))
X_t = X.T

# Creating a vector to write the line model as matrix multiplication
intercept_pred = 0.8
slope_pred = 0.9
w_t = np.array([[intercept_pred, slope_pred]])
w = w_t.T

# Predicted values are calculated by w_T * X
y_hat = np.matmul(w_t, X).T

# Root mean squared error calculation
table_err = np.concatenate((X_t[:, 1].reshape(N, 1), y, y_hat,
                            residuals(y, y_hat), 
                            square_residuals(y, y_hat),
                            absolute_residuals(y, y_hat)), 
                           axis = 1)
df_err = pd.DataFrame(table_err, columns = ['X', 'f(x)', 'f_pred(x)', 'diff', 'SE', 'AE'])

print(df_err, '\n')
print("Square Residuals: ", mean_squared_error(y, y_hat))
print("Absolute Residuals: ", mean_absolute_error(y, y_hat))


# Plots funtion with and without noise
plt.figure(0)
plt.plot(X_t[:, 1], y_real, color = 'red', alpha = 0.7, linestyle = '--')
plt.plot(X_t[:, 1], y_hat, color = 'green', alpha = 0.7)
plt.scatter(X_t[:, 1], y, s = 8)
plt.xlabel('x')
plt.xticks(range(0, 10))
plt.ylabel('f(x)')
plt.yticks(range(0, 10))
plt.title('Comparision between predicted and real f(x)')
plt.legend(["f(x)", 'f_pred(x)', "f(x) + N(mean, var)"])
plt.grid(True)
plt.show()