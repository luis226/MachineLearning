# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 04:55:33 2019

@author: Luis Galaviz
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from KNN import KNN
from error_functions import mean_squared_error, mean_absolute_error

np.random.seed(40)
X = np.arange(0, 20)
tolerance = 2
X_all = np.linspace(np.min(X) - tolerance, np.max(X) + tolerance, num = 1000)
N = len(X)
f = lambda X: 2*X + 1
y_real = f(X)
y = y_real + np.random.normal(0, 2, size= N)



X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.4, random_state=42)

plt.grid(True)
plt.plot(X_all, f(X_all), color = 'red', linestyle = ':', alpha = 1)
plt.scatter(X_train, y_train, color = 'blue', marker = '.')
plt.scatter(X_test, y_test, color = 'orange', marker = 'x')
legend = ['f(x)']
colors = ['green', 'magenta']
max_k_plot = 3

err_table = np.ndarray((0, 5))
for i, k in enumerate(range(1, 6)):
    model = KNN(k)     
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    test_mse =  mean_squared_error(y_test, y_test_pred)
    train_mse =  mean_squared_error(y_train, y_train_pred) 
                  
    test_mae =  mean_absolute_error(y_test, y_test_pred)
    train_mae =  mean_absolute_error(y_train, y_train_pred) 
    
    row = np.array([k, train_mse, test_mse, train_mae, test_mae]).reshape((1,5))
    err_table = np.concatenate((err_table,row), axis = 0)
    
    if k < max_k_plot:
        plt.plot(X_all, model.predict(X_all), alpha = 0.8, c = colors[i])
        legend.append('k = ' + str(k))

legend.append('Train Set')
legend.append('Test Set')
plt.legend(legend)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("f(x) = 2X + 1 + Norm(0, 2)  for distinct k's")
df_err = pd.DataFrame(err_table, 
                      columns = ['k', 'train_mse','test_mse', 
                                   'train_mae', 'test_mae'])    
print(df_err)
    
